import os
import argparse
from transformers import set_seed, PreTrainedTokenizerFast
from datasets import load_dataset, DatasetDict

SEED = 42
set_seed(SEED)

validation_chr = 'chr5'
test_chr = 'chr5'

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', type=str, default="data/preprocess/databank_mouse_bambu_se_discovery.preprocessed.updated.csv", help='Name of processed sequence data file')
parser.add_argument('-t', '--tokenizer', type=str, default="processing-seqs/mach_tokenizer.json", help='Path to tokenizer json file')
parser.add_argument('-m', '--max_seq_len', type=int, default = 2**16, help='Maximum sequence length')
parser.add_argument('-p', '--num_threads', type=int, default=48, help='Number of threads')

args = parser.parse_args()

tokenization_output_dir = 'data/tokenization'
os.makedirs(tokenization_output_dir, exist_ok=True)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file=args.tokenizer,
    padding_side='right',
    truncation_side='right',
    cls_token='[CLS]',
    bos_token='[CLS]',
    sep_token='[SEP]',
    eos_token='[SEP]',
    unk_token='[UNK]',
    mask_token='[MASK]',
    pad_token='[PAD]',
    model_max_length=args.max_seq_len)

raw_datasets = load_dataset('csv', data_files=args.data)

dataset_columns = raw_datasets.column_names['train']
raw_datasets = raw_datasets['train']

# validation_dataset = raw_datasets.filter(lambda x: x['chr'] == validation_chr, num_proc=args.num_threads)
# test_dataset = raw_datasets.filter(lambda x: x['chr'] == test_chr, num_proc=args.num_threads)
# train_dataset = raw_datasets.filter(lambda x: (x['chr'] != validation_chr) & (x['chr'] != test_chr), num_proc=args.num_threads)

# For debugging purposes, we tokenize the entire dataset as train, val, and test
raw_datasets = DatasetDict(
    {'train': raw_datasets,
    'validation': raw_datasets,
    'test': raw_datasets})

# raw_datasets = DatasetDict(
#     {'train': train_dataset,
#     'validation': validation_dataset,
#     'test': test_dataset})

def tokenize_seqs(examples):
    return tokenizer(
        examples['seq'],
        return_special_tokens_mask=True)

tokenized_datasets = raw_datasets.map(
    tokenize_seqs,
    batched=True,
    num_proc=args.num_threads,
    remove_columns=['seq'],
    desc='Tokenizing every isoform sequence')

data_prefix = os.path.basename(args.data).replace('.csv.gz', '')
tokenized_datasets.save_to_disk(f"{tokenization_output_dir}/{data_prefix}")
