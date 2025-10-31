#!/usr/bin/env python

import os
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Disable tokenizer parallelism to avoid fork warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Use spawn start method for multiprocessing to avoid CUDA fork issues
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

import torch

from modeling_hyena import StripedHyenaModelForCausalLM
from transformers import PreTrainedTokenizerFast

# Keep parser definition outside for when module is imported
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', default="model/mach-1-v00/checkpoint-16384", type=str, help='Path to model checkpoint directory')
parser.add_argument('-t', '--tokenizer', type=str, default="processing-seqs/mach_tokenizer.json", help='Path to tokenizer json file')
parser.add_argument('-m', '--max_seq_len', type=int, default=2**16, help='Maximum sequence length')
parser.add_argument('-i', '--input_files', nargs='+', required=True, help='List of input sequence files')
parser.add_argument('--aso_length', type=int, default=18, help='Length of aso n-mer')
parser.add_argument('--aso_step', type=int, default=1, help='Step size for sliding window')
parser.add_argument('--aso_start', type=int, default=21101, help='Start position for aso scanning (relative to sequence), zero-indexed')
parser.add_argument('--aso_end', type=int, default=27368, help='End position for aso scanning (None = end of sequence), end is exclusive')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--aso_embedding_mode', type=str, default='mean', choices=['mean', 'complementary'],
                    help='ASO embedding mode: "mean" uses mean of all bases, "complementary" uses mean of base and its complement')
parser.add_argument('--analyze_region', type=int, nargs=2, metavar=('START', 'END'), # default=(26870, 26924), exon 7
                    help='Analyze only a specific region of sequence (start and end positions). Zero-indexed, end is exclusive.')

def get_complement_base(base):
    """Get the complementary DNA base"""
    complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C',
                      'a': 't', 't': 'a', 'c': 'g', 'g': 'c'}
    return complement_map.get(base, base)

def get_nucleotide_ids(tokenizer, uppercase=False):
    """Get token IDs for nucleotides (lowercase or uppercase)"""
    nucleotides = ['A', 'C', 'G', 'T'] if uppercase else ['a', 'c', 'g', 't']
    return [tokenizer.convert_tokens_to_ids([nt])[0] for nt in nucleotides]

def get_nucleotide_pair_ids(tokenizer, base, uppercase=False):
    """Get token IDs for a base and its complement pair"""
    base = base.upper() if uppercase else base.lower()
    complement = get_complement_base(base)
    return [tokenizer.convert_tokens_to_ids([base])[0], 
            tokenizer.convert_tokens_to_ids([complement])[0]]

def create_aso_embeddings(model, tokenizer, embedding_mode='mean'):
    """Create aso token embeddings for both lowercase and uppercase nucleotides"""
    embeddings = model.get_input_embeddings()
    
    with torch.no_grad():
        if embedding_mode == 'mean':
            lowercase_ids = get_nucleotide_ids(tokenizer, uppercase=False)
            uppercase_ids = get_nucleotide_ids(tokenizer, uppercase=True)
            return {
                'lowercase': embeddings.weight[lowercase_ids].mean(dim=0),
                'uppercase': embeddings.weight[uppercase_ids].mean(dim=0)
            }
        elif embedding_mode == 'complementary':
            result = {}
            for base in ['a', 'c', 'g', 't']:
                pair_ids = get_nucleotide_pair_ids(tokenizer, base, uppercase=False)
                result[base] = embeddings.weight[pair_ids].mean(dim=0)
            for base in ['A', 'C', 'G', 'T']:
                pair_ids = get_nucleotide_pair_ids(tokenizer, base, uppercase=True)
                result[base] = embeddings.weight[pair_ids].mean(dim=0)
            return result
        else:
            raise ValueError(f"Unknown embedding mode: {embedding_mode}")

def tokenize_sequence(sequence, tokenizer, device):
    """Tokenize sequence and return input_ids on device"""
    tokenized = tokenizer(sequence, return_tensors='pt', truncation=True, max_length=tokenizer.model_max_length)
    return tokenized['input_ids'].to(device)

def compute_log_probs(logits, labels):
    """Compute log probabilities from logits and labels"""
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1).float()
    return token_log_probs.squeeze(0).cpu().numpy()

def apply_aso_embedding(embeds_batch, positions, sequence, aso_length, aso_embeddings, embedding_mode='mean'):
    """Apply ASO embeddings to batch of sequences at given positions."""
    aso_seqs = []
    for batch_idx, pos in enumerate(positions):
        actual_start = pos + 1
        actual_end = min(actual_start + aso_length, embeds_batch.shape[1])
        aso_seq = sequence[pos:pos+aso_length]
        for i, char in enumerate(aso_seq):
            if actual_start + i < actual_end:
                emb = aso_embeddings['uppercase' if char.isupper() else 'lowercase'] if embedding_mode == 'mean' else aso_embeddings[char]
                embeds_batch[batch_idx, actual_start + i] = emb
        aso_seqs.append(aso_seq)
    return aso_seqs, embeds_batch

def process_baseline(sequence, model, tokenizer, device, analyze_region=None):
    """Predict probabilities for a sequence"""
    input_ids = tokenize_sequence(sequence, tokenizer, device)
    
    with torch.no_grad():
        logits = model(input_ids).logits
    
    token_log_probs = compute_log_probs(logits, input_ids)
    region_tokens = tokenizer.convert_ids_to_tokens(input_ids[:, 1:].squeeze(0).cpu().tolist())
    if analyze_region:
        start, end = analyze_region
        # token_log_probs = token_log_probs[start:end]
        token_log_probs = np.concatenate((token_log_probs[start:start+1], token_log_probs[end:end+1]))
    return {
        'tokens': region_tokens,
        'log_probs': token_log_probs,
        'seq_log_prob': np.sum(token_log_probs)
    }

def forward_pass_batch(model, embeds_batch):
    """Forward pass through backbone for batch of embeddings."""
    x = embeds_batch
    for block in model.backbone.blocks:
        x, _ = block(x, inference_params=None, padding_mask=None)
    if model.backbone.norm:
        x = model.backbone.norm(x)
    return model.backbone.unembed.unembed(x)

def process_aso_comparison_batch(gpu_id, batch_data, checkpoint_path, tokenizer_path, max_seq_len, embedding_mode='mean', analyze_region=None):
    """Process aso candidates comparing desired vs undesired sequences on a specific GPU"""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    
    model = StripedHyenaModelForCausalLM.from_pretrained(checkpoint_path).to(device).eval()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_path, padding_side='right', truncation_side='right',
        cls_token='[CLS]', bos_token='[CLS]', sep_token='[SEP]', eos_token='[SEP]',
        unk_token='[UNK]', mask_token='[MASK]', pad_token='[PAD]', model_max_length=max_seq_len)
    
    aso_embeddings = create_aso_embeddings(model, tokenizer, embedding_mode=embedding_mode)
    
    d_seq, u_seq = batch_data['desired_sequence'], batch_data['undesired_sequence']
    d_input_ids = tokenize_sequence(d_seq, tokenizer, device)
    u_input_ids = tokenize_sequence(u_seq, tokenizer, device)
    d_baseline, u_baseline = batch_data['desired_baseline'], batch_data['undesired_baseline']
    aso_len = batch_data['aso_length']
    positions = batch_data['positions']
    batch_size = len(positions)
    
    # Create batch embeddings and apply ASO
    d_base_embeds = model.get_input_embeddings()(d_input_ids).repeat(batch_size, 1, 1).clone()
    u_base_embeds = model.get_input_embeddings()(u_input_ids).repeat(batch_size, 1, 1).clone()
    d_aso_seqs, d_embeds_batch = apply_aso_embedding(d_base_embeds, positions, d_seq, aso_len, aso_embeddings, embedding_mode)
    u_aso_seqs, u_embeds_batch = apply_aso_embedding(u_base_embeds, positions, u_seq, aso_len, aso_embeddings, embedding_mode)
    
    # Batched forward pass
    with torch.no_grad():
        d_logits = forward_pass_batch(model, d_embeds_batch)
        u_logits = forward_pass_batch(model, u_embeds_batch)
    
    # Compute token probs for batch (optionally restrict to region)
    start, end = analyze_region if analyze_region else (0, d_logits.shape[1] - 1 - 1)

    # This is to only get log probs for the ASO region if analyze_region is specified
    d_logits_shifted = d_logits[:, start:end, :]
    u_logits_shifted = u_logits[:, start:end, :]
    d_labels = d_input_ids[:, start+1:end+1].expand(batch_size, -1)
    u_labels = u_input_ids[:, start+1:end+1].expand(batch_size, -1)

    # Getting log probs for the "boundary tokens" of the ASO region
    # d_logits_shifted = torch.cat((d_logits[:, start:start+1, :], d_logits[:, end:end+1, :]), dim=1)
    # u_logits_shifted = torch.cat((u_logits[:, start:start+1, :], u_logits[:, end:end+1, :]), dim=1)
    # d_labels = torch.cat((d_input_ids[:, start:start+1], d_input_ids[:, end:end+1]), dim=1).expand(batch_size, -1)
    # u_labels = torch.cat((u_input_ids[:, start:start+1], u_input_ids[:, end:end+1]), dim=1).expand(batch_size, -1)

    d_log_probs, u_log_probs = torch.log_softmax(d_logits_shifted, dim=-1), torch.log_softmax(u_logits_shifted, dim=-1)
    d_total_probs = torch.gather(d_log_probs, 2, d_labels.unsqueeze(-1)).squeeze(-1).float().sum(dim=1).cpu().numpy()
    u_total_probs = torch.gather(u_log_probs, 2, u_labels.unsqueeze(-1)).squeeze(-1).float().sum(dim=1).cpu().numpy()
    
    # Compute results
    results = [{
        'position': pos,
        'aso_sequence': d_aso_seqs[i],
        'desired_baseline': d_baseline,
        'desired_aso_log_prob': d_total_probs[i],
        'desired_log_prob_delta': d_total_probs[i] - d_baseline,
        'undesired_baseline': u_baseline,
        'undesired_aso_log_prob': u_total_probs[i],
        'undesired_log_prob_delta': u_total_probs[i] - u_baseline,
        'combined_delta': (d_total_probs[i] - d_baseline) - (u_total_probs[i] - u_baseline)
    } for i, pos in enumerate(positions)]
    
    del model
    torch.cuda.empty_cache()
    return results

def _process_aso_comparison_batch_wrapper(batch_data):
    """Wrapper function for pickling - extracts params from batch_data dict"""
    return process_aso_comparison_batch(
        batch_data['gpu_id'],
        batch_data,
        batch_data['checkpoint_path'],
        batch_data['tokenizer_path'],
        batch_data['max_seq_len'],
        batch_data.get('embedding_mode', 'mean'),
        batch_data.get('analyze_region', None)
    )

def run_aso_experiment_comparison(desired_seq, undesired_seq, model, tokenizer, device, args, analyze_region=None):
    """Run aso virtual experiment comparing desired vs undesired sequences with GPU parallelization"""
    print("\n=== Running ASO Virtual Experiment ===")
    
    aso_start, aso_end = args.aso_start, args.aso_end or min(len(desired_seq), len(undesired_seq))
    aso_length, aso_step = args.aso_length, args.aso_step
    print(f"ASO length: {aso_length}, Scanning: {aso_start}-{aso_end}, Step: {aso_step}")
    
    print("Computing baseline probabilities...")
    d_baseline = process_baseline(desired_seq, model, tokenizer, device, analyze_region)['seq_log_prob']
    u_baseline = process_baseline(undesired_seq, model, tokenizer, device, analyze_region)['seq_log_prob']
    print(f"Desired: {d_baseline:.4f}, Undesired: {u_baseline:.4f}")
    
    positions = list(range(aso_start, aso_end - aso_length + 1, aso_step))
    print(f"Total ASO candidates: {len(positions)}")
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    batch_size = args.batch_size
    batch_data_list = [{
        'gpu_id': (batch_idx // batch_size) % num_gpus,
        'desired_sequence': desired_seq, 'undesired_sequence': undesired_seq,
        'positions': positions[batch_idx:batch_idx + batch_size], 'aso_length': aso_length,
        'desired_baseline': d_baseline, 'undesired_baseline': u_baseline,
        'checkpoint_path': args.checkpoint, 'tokenizer_path': args.tokenizer,
        'max_seq_len': args.max_seq_len, 'embedding_mode': args.aso_embedding_mode,
        'analyze_region': analyze_region,
    } for batch_idx in range(0, len(positions), batch_size)]
    
    print(f"Processing {len(batch_data_list)} batches on {num_gpus} GPU(s)...")
    print(f"Using ASO embedding mode: {args.aso_embedding_mode}")
    aso_results = []
    
    if num_gpus > 1:
        model.cpu()
        del model
        torch.cuda.empty_cache()
        with Pool(num_gpus) as pool:
            for batch_result in tqdm(pool.imap_unordered(_process_aso_comparison_batch_wrapper, batch_data_list), total=len(batch_data_list), desc="Processing ASO"):
                aso_results.extend(batch_result)
    else:
        for batch_data in tqdm(batch_data_list, desc="Processing ASO"):
            aso_results.extend(process_aso_comparison_batch(0, batch_data, batch_data['checkpoint_path'], batch_data['tokenizer_path'], batch_data['max_seq_len'], args.aso_embedding_mode, analyze_region))
    
    return pd.DataFrame(aso_results)

if __name__ == '__main__':
    args = parser.parse_args()
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPU(s) and {cpu_count()} CPU cores\n")
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer, padding_side='right', truncation_side='right',
        cls_token='[CLS]', bos_token='[CLS]', sep_token='[SEP]', eos_token='[SEP]',
        unk_token='[UNK]', mask_token='[MASK]', pad_token='[PAD]', model_max_length=args.max_seq_len)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = StripedHyenaModelForCausalLM.from_pretrained(args.checkpoint).to(device).eval()
    
    # Load sequences
    input_sequences = {}
    print(f"Loading {len(args.input_files)} input file(s)...\n")
    for input_file in args.input_files:
        with open(input_file, 'r') as f:
            seq = f.read().strip()
            if 'H' in seq:
                seq = seq[seq.index('H'):]
        file_name = os.path.basename(input_file).replace('.txt', '')
        input_sequences[file_name] = seq
    
    # ASO experiment for 2 files
    if len(input_sequences) == 2:
        file_names = list(input_sequences.keys())
        desired_seq, undesired_seq = input_sequences[file_names[0]], input_sequences[file_names[1]]
        print(f"\n[ASO Experiment] Desired: {file_names[0]}, Undesired: {file_names[1]}")
        analyze_region = tuple(args.analyze_region) if args.analyze_region else None
        start, end = analyze_region if analyze_region else (0, len(desired_seq))
        print(f"\nDesired sequence in analyze_region [{start}:{end}]:\n{desired_seq[start:end]}")
        print(f"\nUndesired sequence in analyze_region [{start}:{end}]:\n{undesired_seq[start:end]}")
        aso_df = run_aso_experiment_comparison(desired_seq, undesired_seq, model, tokenizer, device, args, analyze_region)
        aso_df = aso_df[aso_df['desired_log_prob_delta'] > 0] # Only leave desired_log_prob_delta is positive
        
        # Save sorted results
        for metric, ascending, name in [
            ('desired_log_prob_delta', False, f"desired_{file_names[0]}"),
            ('undesired_log_prob_delta', True, f"undesired_{file_names[1]}"),
            ('combined_delta', False, f"combined_{file_names[0]}_vs_{file_names[1]}"),
        ]:
            df_sorted = aso_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
            output_file = f"results/aso_{name}.csv"
            df_sorted.to_csv(output_file, index=False)
            print(f"\n  Saved: {output_file}")
            print(f"  Top 5 candidates:")
            print(df_sorted.head(5)[['position', 'aso_sequence', metric]].to_string(index=False))
    else:
        print(f"\nASO experiment requires exactly 2 input files. Got {len(input_sequences)} files.")
