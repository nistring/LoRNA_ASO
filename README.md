# Mach-1

Mach-1 is a long-context RNA foundation model for predicting transcriptome architecture. This repository houses the core model weights alongside the scripts needed to tokenize sequences, train and fine-tune the StripedHyena-based architecture, and run inference workflows.

This work has been modified for optimal rASO design. You can find the original repo at https://github.com/goodarzilab/mach-1

## Repository Structure

```
mach-1
├── data/SMN2                 # An input example of SMN2 with or without exon 7 skipping
├── processing-seqs/          # Tokenization configs and CLI tooling for data preparation
│   ├── mach_tokenizer.json    # Tokenizer configuration used across notebooks and scripts
│   ├── prepare_data.R         # RNA sequence preprocessing and formatting utilities
│   └── tokenize_data.py       # Batch tokenizer for genomic fastas/CSVs
├── training-model/           # Configuration, training, and inference scripts
│   ├── configuration_hyena.py # Default StripedHyena model definition
│   ├── generate_seqs.py       # Synthetic sequence generation entry point
│   ├── get_embeddings.py      # Embedding extraction for downstream analyses
│   ├── get_likelihoods.py     # Likelihood computation and scoring helpers
|   ├── get_likelihoods_aso.py # Likelihood computation for putative rASOs
│   ├── mach_dependencies.sh   # Environment bootstrap script
│   ├── modeling_hyena.py      # Core Hyena architecture implementation
│   └── train_model.py         # Training script for Mach-1 checkpoints
├── results/
|   ├── generations/          # Generated sequences by `generate_seqs.py`
|   ├── tokens/               # Token embedding visualization by `get_token_embeddings.py`
|   └── aso_ .csv             # List of rASOs sorted by their potency on restoring splicing
└── model/                    # Pretrained checkpoints and tokenizer artifacts
```

## Getting Started

1. Install dependencies listed in `training-model/mach_dependencies.sh` or adapt them to your compute environment.
2. Use the scripts in `processing-seqs/` to prepare and tokenize the RNA sequences of interest.
3. Train or fine-tune Mach-1 with `training-model/train_model.py`, or run inference with `get_likelihoods.py`, `get_embeddings.py`, and `generate_seqs.py`.
4. Transfer the resulting outputs (likelihoods, embeddings, variant scores, synthetic sequences) into the directory structure expected by `mach-1-manuscript` to reproduce the manuscript analyses.

## Companion Repository

The full set of data-processing pipelines, downstream analyses, and figure-generation workflows that accompany the Mach-1 study live in the companion repository [`mach-1-manuscript`](https://github.com/csglab/mach-1-manuscript.git).

## Update
- `get_likelihoods_aso.py` do mock experiments on the effect of rASO. It calculates the difference of likelihoods of desired (correct splicing) to undesired (incorrect splicing) after replacing a certain length of nucleotides with tokens that represent ASO sequences (n), which is an average of the token embeddings of the bases (A, C, G, T). When `analyze_region` provided as a tuple `(start, end)`:
  - Extracts only tokens/probabilities from that region
  - Computes log probability sum for only that region
  - Returns tokens and probabilities for that region only
- `get_token_embeddings.py` visualizes the token embeddings with 128 dimensions compressed by PCA.
- Added `generate` method in `modeling_hyena.py` for non-Hugging Face registered model support