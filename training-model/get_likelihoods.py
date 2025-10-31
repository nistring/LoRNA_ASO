#!/usr/bin/env python

import os
import argparse
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from modeling_hyena import StripedHyenaModelForCausalLM

from datasets import load_from_disk, concatenate_datasets
from accelerate.utils import set_seed
from transformers import (
    set_seed as transformers_set_seed,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

SEED = 42
set_seed(SEED)
transformers_set_seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', default="model/mach-1-v00/checkpoint-16384", type=str, help='Path to model checkpoint directory')
parser.add_argument('-d', '--dataset', default="data/tokenization/databank_mouse_bambu_se_discovery.preprocessed.updated.csv", type=str, help='Path to tokenized datasets')
parser.add_argument('-t', '--dataset_type', type=str, help='Type of the dataset', default='test')
parser.add_argument('-n', '--num_shards', type=int, help='Number of shards', default=1)
parser.add_argument('-b', '--per_device_eval_batch_size', type=int, help='Per device evaluation batch size', default=32)
parser.add_argument('-e', '--eval_accumulation_steps', type=int, help='Evaluation accumulation steps', default=128)
parser.add_argument('-p', '--num_threads', type=int, help='Number of threads', default=48)
parser.add_argument('--emit_tokens_csv', action='store_true', default=True, help='Emit per-token logprobs CSV')
parser.add_argument('--emit_sequence_csv', action='store_true', default=True, help='Emit sequence-level likelihood CSV')

args = parser.parse_args()

model = StripedHyenaModelForCausalLM.from_pretrained(args.checkpoint)
project_id, model_name, checkpoint_id = args.checkpoint.split('/')[-3:]
dataset_id = os.path.basename(args.dataset)
dataset_prefix = f"{dataset_id}.{args.dataset_type}_dataset"

predictions_dir = f"{project_id}/{model_name}/{checkpoint_id}/{dataset_prefix}"
os.makedirs(predictions_dir, exist_ok=True)

tokenizer = PreTrainedTokenizerFast.from_pretrained(args.checkpoint)
vocab_size = tokenizer.vocab_size

tokenized_datasets = load_from_disk(args.dataset)

first_dataset = list(tokenized_datasets.keys())[0]
dataset_columns = tokenized_datasets.column_names[first_dataset]
columns_to_keep = ['input_ids', 'attention_mask', 'special_tokens_mask', 'seq_len', 'transcript_id']
columns_to_keep = list(set(dataset_columns) & set(columns_to_keep))
tokenized_datasets = tokenized_datasets.select_columns(columns_to_keep)

if args.dataset_type not in ['train', 'validation', 'test']:
    tokenized_datasets[args.dataset_type] = concatenate_datasets([v for _, v in tokenized_datasets.items()])

the_dataset = tokenized_datasets[args.dataset_type]

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False)

training_args = TrainingArguments(
    output_dir = predictions_dir,
    run_name = model_name,
    per_device_eval_batch_size = args.per_device_eval_batch_size,
    eval_accumulation_steps = args.eval_accumulation_steps,
    seed = SEED,
    # Important: keep sample order stable for mapping predictions -> dataset rows
    group_by_length = False,  # was True
    length_column_name = 'seq_len',
    auto_find_batch_size = True,
    dataloader_num_workers = args.num_threads,
    dataloader_persistent_workers = False)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator)

sharded_csvs = glob.glob(f"{predictions_dir}/{dataset_prefix}.likelihoods.shard_*.csv")

if len(sharded_csvs) == 0:
    last_saved_shard = -1
else:
    last_saved_shard = sorted([csv.split('.')[-2].removeprefix('shard_') for csv in sharded_csvs])[-1]
    last_saved_shard = int(last_saved_shard)

for i in range(last_saved_shard + 1, args.num_shards):
    print(f"Processing shard {i} of {args.num_shards}")

    sharded_dataset = the_dataset.shard(args.num_shards, i, contiguous=True)

    output = trainer.predict(sharded_dataset)

    logits, labels = torch.from_numpy(output.predictions), torch.from_numpy(output.label_ids)

    softmax_logprobs = torch.log_softmax(logits, dim=-1)
    softmax_logprobs = softmax_logprobs[:, :-1] 
    labels = labels[:, 1:]

    assert(softmax_logprobs.shape[1] == labels.shape[1])

    labels[labels == -100] = tokenizer.pad_token_id

    logprobs = torch.gather(
        softmax_logprobs,
        2,
        labels.unsqueeze(-1)
    ).squeeze(-1)

    softmax_logprobs = softmax_logprobs.to(torch.float64)
    logprobs = logprobs.to(torch.float64)

    # Sequence-level aggregation (unchanged)
    seq_likelihood = [torch.sum(logprobs[i, labels[i] != tokenizer.pad_token_id]) for i in range(len(labels))]
    seq_likelihood_float = [float(ll) for ll in seq_likelihood]
    
    tss_id, tts_id = tokenizer.convert_tokens_to_ids(['S', 'E'])

    tss_index = torch.where(labels == tss_id)
    tss_logprobs = logprobs[tss_index[0], tss_index[1]]
    tss_logprobs = tss_logprobs.tolist()

    tts_index = torch.where(labels == tts_id)
    tts_logprobs = logprobs[tts_index[0], tts_index[1]]
    tts_logprobs = tts_logprobs.tolist()

    tr_lens = np.array(tts_index[1]) - np.array(tss_index[1]) + 1

    seq_likelihood_float = [float(ll) for ll in seq_likelihood]
    tss_logprobs_float = [float(ll) for ll in tss_logprobs]
    tts_logprobs_float = [float(ll) for ll in tts_logprobs]

    # Write sequence-level CSV (same as before)
    if args.emit_sequence_csv:
        seq_lls = {
            'transcript_id': sharded_dataset['transcript_id'],
            'seq_len': sharded_dataset['seq_len'],
            'tr_len': tr_lens.tolist(),
            'likelihood': seq_likelihood_float,
            'tss_logprobs': tss_logprobs_float,
            'tts_logprobs': tts_logprobs_float
        }

        seq_lls = pd.DataFrame(seq_lls)

        include_header = True if i == 0 else False
        output_suffix = str(i).zfill(len(str(args.num_shards)))
        seq_lls_filename = f"{predictions_dir}/{dataset_prefix}.likelihoods.shard_{output_suffix}.csv"
        seq_lls.to_csv(seq_lls_filename, index=False, header=include_header)

    # New: write per-token logprobs (one row per non-pad position)
    if args.emit_tokens_csv:
        # labels/logprobs came from numpy -> CPU; keep on CPU
        mask = labels != tokenizer.pad_token_id
        b_idx, t_idx = torch.where(mask)

        # Map back to transcript ids and positions
        transcript_ids = [sharded_dataset['transcript_id'][int(b)] for b in b_idx.tolist()]
        positions = t_idx.tolist()

        token_ids = labels[mask].tolist()
        # Convert token ids to string tokens
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        token_logprobs = logprobs[mask].tolist()

        tok_df = pd.DataFrame({
            'transcript_id': transcript_ids,
            'position': positions,           # 0-based position after shift
            'token_id': token_ids,
            'token': tokens,
            'logprob': token_logprobs
        })

        output_suffix = str(i).zfill(len(str(args.num_shards)))
        tok_lls_filename = f"{predictions_dir}/{dataset_prefix}.token_logprobs.shard_{output_suffix}.csv"
        tok_df.to_csv(tok_lls_filename, index=False)
