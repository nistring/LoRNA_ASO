from typing import Optional, Tuple, Union
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from configuration_hyena import StripedHyenaConfig
from stripedhyena.model import StripedHyena
from stripedhyena.utils import dotdict

from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput, CausalLMOutputWithPast
from transformers.utils import logging

import wandb
from wandb import AlertLevel

logger = logging.get_logger(__name__)

class StripedHyenaPreTrainedModel(PreTrainedModel):
    config_class = StripedHyenaConfig
    base_model_prefix = "sh"
    supports_gradient_checkpointing = False
    _no_split_modules = ["AttentionBlock", "ParallelGatedConvBlock"]
    _skip_keys_device_placement = "past_key_values"
    _keys_to_ignore_on_load_missing = [r"freq"]
    _keys_to_ignore_on_load_unexpected = [r"fftconv", r"twiddle_factors"]
    _supports_flash_attn_2 = True

class StripedHyenaModelForCausalLM(StripedHyenaPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        model_config = dotdict(config.to_dict())
        self.backbone = StripedHyena(model_config)
        self.backbone.gradient_checkpointing = False
        self.config = config
        vocab_size = config.vocab_size
        if vocab_size % config.make_vocab_size_divisible_by != 0:
            vocab_size += config.make_vocab_size_divisible_by - (
                vocab_size % config.make_vocab_size_divisible_by
            )
        self.vocab_size = vocab_size
        self.post_init()
        self.force_dtype()

    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues() 
        
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.backbone.gradient_checkpointing = enable

    def get_input_embeddings(self):
        return self.backbone.embedding_layer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache:
            if self.backbone.gradient_checkpointing and self.backbone.training:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
            elif labels is not None:
                logger.warning_once(
                    "`use_cache=True` is incompatible with loss calculation. Setting `use_cache=False`..."
                )
                use_cache = False

        inputs = input_ids
        if use_cache:
            if past_key_values is None:
                past_key_values = self.backbone.initialize_inference_params()

                batch_size = input_ids.shape[0]
                past_key_values["mha"].max_batch_size = batch_size
                past_key_values["hyena"].max_batch_size = batch_size
            else:
                seqlen_offset = past_key_values["mha"].seqlen_offset
                if seqlen_offset == 0:
                    # second loop through generate will have prompt_len + 1 as seqlen
                    seqlen_offset = input_ids.shape[-1] - 1
                    past_key_values["hyena"].seqlen_offset = seqlen_offset
                    past_key_values["mha"].seqlen_offset = seqlen_offset
                else:
                    past_key_values["mha"].seqlen_offset += 1
                    past_key_values["hyena"].seqlen_offset += 1

                inputs = input_ids[
                    :,
                    -1:,
                ]

        logits, past_key_values = self.backbone(
            inputs,
            padding_mask=attention_mask,
            inference_params_dict=past_key_values if use_cache else None,
        )
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = F.cross_entropy(shift_logits, shift_labels)

            if (loss == 0) | (loss.isnan()):
                wandb.alert(
                    title="Invalid loss value",
                    text="Loss is zero or NaN",
                    level=AlertLevel.ERROR)
                wandb.finish(exit_code=600)

        if return_dict:
            return CausalLMOutputWithPast(
                logits=logits,
                hidden_states=None,
                past_key_values=past_key_values if use_cache else None,
                loss=loss,
            )
        else:
            return logits

    @classmethod
    def can_generate(cls) -> bool:
        return True

    def prepare_inputs_for_generation(
        self, input_ids, attention_mask=None, past_key_values=None, **kwargs
    ):
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def generate(
        self,
        input_ids: torch.LongTensor,
        num_return_sequences: int = 1,
        max_new_tokens: int = 20,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        use_cache: bool = True,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.LongTensor:
        # Minimal, self-contained generation to avoid relying on HF GenerationMixin
        device = input_ids.device
        self.eval()
        if eos_token_id is None and hasattr(self.config, "eos_token_id"):
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = eos_token_id

        # Duplicate prompts for num_return_sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat_interleave(num_return_sequences, dim=0)
            if attention_mask is not None:
                attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        sequences = input_ids
        batch_size = sequences.size(0)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0):
            top_k = max(top_k, 0)
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_values = values[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, torch.full_like(logits, -float("inf")), logits)
            if 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float("inf"))
            return logits

        past_key_values = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(
                    input_ids=sequences if past_key_values is None else sequences[:, -1:],
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                    return_dict=True,
                )
                logits = outputs.logits[:, -1, :]  # last token logits
                past_key_values = outputs.past_key_values if use_cache else None

                next_token_logits = logits.to(dtype=torch.float32)
                if temperature is not None and temperature > 0:
                    next_token_logits = next_token_logits / temperature

                if do_sample:
                    filtered = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    probs = torch.softmax(filtered, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)

                if eos_token_id is not None:
                    # If already finished, keep padding token
                    next_tokens = torch.where(
                        finished, torch.full_like(next_tokens, pad_token_id), next_tokens
                    )

                sequences = torch.cat([sequences, next_tokens.unsqueeze(-1)], dim=-1)

                if eos_token_id is not None:
                    finished = finished | (next_tokens == eos_token_id)
                    if torch.all(finished):
                        break

        return sequences

class StripedHyenaForEmbeddings(StripedHyena):
    def __init__(self, config, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.tts_token_id = tokenizer.convert_tokens_to_ids(['S'])[0]
        self.tes_token_id = tokenizer.convert_tokens_to_ids(['E'])[0]
        self.dna_token_ids = tokenizer.convert_tokens_to_ids(['W', 'X', 'Y', 'Z'])
        self.exon_token_ids = tokenizer.convert_tokens_to_ids(['a', 'c', 'g', 't'])
        self.intron_token_ids = tokenizer.convert_tokens_to_ids(['A', 'C', 'G', 'T'])

    def forward(self, x, inference_params_dict=None, padding_mask=None):
        B, L = x.shape  # B: batch size, L: sequence length
        self.e_token_indices = (x == self.tts_token_id).nonzero(as_tuple=True)
        
        x = self.embedding_layer.embed(x)  # Shape: (B, L, 128)

        e_token_embeddings = []

        if inference_params_dict is not None:
            x, block_e_embeddings = self.stateful_forward(x, inference_params_dict, padding_mask)
        else:    
            x, block_e_embeddings = self.stateless_forward(x, padding_mask)

        e_token_embeddings.extend(block_e_embeddings)

        x = self.norm(x) if self.norm else x
        x = self.unembed.unembed(x)

        e_token_embeddings = torch.stack(e_token_embeddings, dim=1)  # Shape: (B, 17, 128)
        
        return x, e_token_embeddings

    def stateful_forward(self, x, inference_params_dict, padding_mask=None):

        e_token_embeddings = []
        
        for block_idx, block in enumerate(self.blocks):
            block_name = "mha" if block_idx in self.config.attn_layer_idxs else "hyena"
            inference_params = inference_params_dict[block_name]

            x, _ = block(x, inference_params=inference_params)
                
            e_token_emb = x[self.e_token_indices]
            e_token_embeddings.append(e_token_emb)

        return x, e_token_embeddings

    def stateless_forward(self, x, padding_mask=None):
        e_token_embeddings = []

        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1)

        for block_idx, block in enumerate(self.blocks):
            x, _ = block(x, inference_params=None, padding_mask=padding_mask)
            e_token_emb = x[self.e_token_indices]
            e_token_embeddings.append(e_token_emb)

        return x, e_token_embeddings


@dataclass
class CausalLMEmbeddingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    e_token_embs: torch.FloatTensor = None

class StripedHyenaModelForExtractingEmbeddings(StripedHyenaPreTrainedModel):
    supports_gradient_checkpointing = True

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        model_config = dotdict(config.to_dict())
        self.backbone = StripedHyenaForEmbeddings(model_config, **kwargs)
        self.backbone.gradient_checkpointing = False
        self.config = config
        self.post_init()
        self.force_dtype()

    def force_dtype(self):
        self.backbone.to_bfloat16_except_poles_residues() 
        
    def _set_gradient_checkpointing(self, enable, gradient_checkpointing_func):
        self.backbone.gradient_checkpointing = enable

    def get_input_embeddings(self):
        return self.backbone.embedding_layer

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        past_key_values=None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if use_cache:
            if self.backbone.gradient_checkpointing and self.backbone.training:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
            elif labels is not None:
                logger.warning_once(
                    "`use_cache=True` is incompatible with loss calculation. Setting `use_cache=False`..."
                )
                use_cache = False

        inputs = input_ids
        if use_cache:
            if past_key_values is None:
                past_key_values = self.backbone.initialize_inference_params()

                batch_size = input_ids.shape[0]
                past_key_values["mha"].max_batch_size = batch_size
                past_key_values["hyena"].max_batch_size = batch_size
            else:
                seqlen_offset = past_key_values["mha"].seqlen_offset
                if seqlen_offset == 0:
                    # second loop through generate will have prompt_len + 1 as seqlen
                    seqlen_offset = input_ids.shape[-1] - 1
                    past_key_values["hyena"].seqlen_offset = seqlen_offset
                    past_key_values["mha"].seqlen_offset = seqlen_offset
                else:
                    past_key_values["mha"].seqlen_offset += 1
                    past_key_values["hyena"].seqlen_offset += 1

                inputs = input_ids[
                    :,
                    -1:,
                ]

        _, e_token_embeddings = self.backbone(
            inputs,
            padding_mask=attention_mask,
            inference_params_dict=past_key_values if use_cache else None,
        )

        loss = torch.tensor(0.0)

        return CausalLMEmbeddingOutput(
            loss=loss,
            e_token_embs=e_token_embeddings)