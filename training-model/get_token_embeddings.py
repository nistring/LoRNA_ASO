import torch
from transformers import PreTrainedTokenizerFast
from modeling_hyena import StripedHyenaModelForExtractingEmbeddings
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Load tokenizer and model
checkpoint = "model/mach-1-v00/checkpoint-16384"  # Update with your checkpoint path
tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint)
model = StripedHyenaModelForExtractingEmbeddings.from_pretrained(checkpoint, tokenizer=tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Define the tokens you want embeddings for
tokens = ['A', 'C', 'G', 'T', 'a', 'c', 'g', 't', 'W', 'X', 'Y', 'Z']

# Convert tokens to token IDs
token_ids = tokenizer.convert_tokens_to_ids(tokens)
print(f"Token IDs: {dict(zip(tokens, token_ids))}")

# Create input tensor (batch_size=1, seq_len=len(tokens))
input_ids = torch.tensor([token_ids], device=device)

# Extract embeddings
with torch.no_grad():
    # Get token embeddings from the embedding layer
    token_embeddings = model.backbone.embedding_layer.embed(input_ids)
    # Shape: (1, num_tokens, embedding_dim)
    
# Convert to numpy for easier handling
token_embeddings_np = token_embeddings.float().cpu().numpy().squeeze(0)  # Shape: (num_tokens, embedding_dim)

# Create a dictionary mapping tokens to their embeddings
embedding_dict = {
    token: token_embeddings_np[i] 
    for i, token in enumerate(tokens)
}

print(f"Embedding shape for each token: {token_embeddings_np.shape[1]}")
print(f"Embeddings extracted for {len(embedding_dict)} tokens")

# Add artificial token 'n' as average of embeddings for 'a','c','g','t' and include it for PCA and similarity plots
# compute indices for lower-case bases and average their embeddings
try:
    lower_group = ['a', 'c', 'g', 't']
    lower_indices = [tokens.index(t) for t in lower_group]
    new_emb = token_embeddings_np[lower_indices, :].mean(axis=0)
    # append new token and its embedding (token_embeddings_np is used later to form `embeddings`)
    token_embeddings_np = np.vstack([token_embeddings_np, new_emb])
    tokens.append('n')
    embedding_dict['n'] = new_emb
    print("Added artificial token 'n' (mean of a,c,g,t) — new token appended to tokens and embeddings")
except ValueError:
    print("Could not create artificial token 'n': one of a,c,g,t not found in tokens")

# Perform PCA for n=2..6 and create n x n subplot grids of pairwise component plots
embeddings = np.asarray(token_embeddings_np)
if embeddings.ndim != 2 or embeddings.shape[0] == 0:
    print("Embeddings have unexpected shape, skipping PCA plots.")
else:
    # color groups
    lower_set = set(['a', 'c', 'g', 't'])
    upper_set = set(['A', 'C', 'G', 'T'])
    wx_set = set(['W', 'X', 'Y', 'Z'])

    # marker groups (same shape for (a,A,W), (c,C,X), (g,G,Y), (t,T,Z))
    marker_map = {
        'a': 'o', 'A': 'o', 'W': 'o',
        'c': 's', 'C': 's', 'X': 's',
        'g': '^', 'G': '^', 'Y': '^',
        't': 'D', 'T': 'D', 'Z': 'D',
    }

    # color map (defined for all tokens but will be used only for plotted tokens)
    color_map = {}
    for t in tokens:
        if t in lower_set:
            color_map[t] = 'tab:blue'
        elif t in upper_set:
            color_map[t] = 'tab:red'
        elif t in wx_set:
            color_map[t] = 'tab:green'
        else:
            color_map[t] = 'k'

    # Filter out W,X,Y,Z for PCA
    tokens_pca = [t for t in tokens if t not in wx_set]
    # corresponding embeddings (num_tokens_pca, embedding_dim)
    indices_pca = [tokens.index(t) for t in tokens_pca]
    embeddings_pca = embeddings[indices_pca, :]

    max_dim = embeddings.shape[1]
    for n_components in range(2, 7):  # 2..6
        if max_dim < n_components:
            print(f"Skipping n={n_components}: embedding dim ({max_dim}) < n_components.")
            continue

        pca = PCA(n_components=n_components)
        comps = pca.fit_transform(embeddings_pca)  # use filtered embeddings

        grid_size = n_components
        figsize = (3 * grid_size, 3 * grid_size)
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        axes = np.atleast_2d(axes)

        for r in range(grid_size):
            for c in range(grid_size):
                ax = axes[r, c]
                if r == c:
                    # diagonal: histogram of component r (for filtered tokens)
                    data = comps[:, r]
                    bins = min(10, max(3, len(data)))
                    counts, bin_edges = np.histogram(data, bins=bins)
                    ax.hist(data, bins=bins, color='lightgray', alpha=0.9)
                    hist_max = counts.max() if counts.size > 0 else 1.0
                    # overlay token markers near top of histogram, spaced slightly
                    for i, token in enumerate(tokens_pca):
                        x_val = data[i]
                        y_val = hist_max * (0.90 - (i * 0.04))  # small vertical offsets
                        ax.scatter(x_val, y_val, color=color_map[token], marker=marker_map.get(token, 'o'), s=60, edgecolors='k', linewidths=0.5)
                        ax.annotate(token, (x_val, y_val), textcoords="offset points", xytext=(4, 0), fontsize=7)
                    ax.set_title(f"PC{r+1} (hist)")
                else:
                    # off-diagonal: scatter PC{c+1} (x) vs PC{r+1} (y) plotting each filtered token
                    x = comps[:, c]  # PC_{c+1} on x-axis
                    y = comps[:, r]  # PC_{r+1} on y-axis
                    for i, token in enumerate(tokens_pca):
                        ax.scatter(x[i], y[i], color=color_map[token], marker=marker_map.get(token, 'o'), s=70, edgecolors='k', linewidths=0.5, alpha=0.9)
                        ax.annotate(token, (x[i], y[i]), textcoords="offset points", xytext=(3, 2), fontsize=7)
                    ax.set_title(f"PC{r+1} vs PC{c+1}")

                # Axis labels with optional explained variance
                xlabel = f"PC{c+1}"
                ylabel = f"PC{r+1}"
                try:
                    xlabel += f" ({pca.explained_variance_ratio_[c]:.2f})"
                    ylabel += f" ({pca.explained_variance_ratio_[r]:.2f})"
                except Exception:
                    pass
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

        plt.suptitle(f"PCA pairwise matrix (without W,X,Y,Z) — {n_components} components", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_path = f"results/tokens/token_embeddings_pca_{n_components}comps_no_WXYZ.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"PCA plots for n={n_components} (without WXYZ) saved to {out_path}")
        try:
            plt.show()
        except Exception:
            pass
        plt.close(fig)

# Compute cosine similarity heatmap of token embeddings
# embeddings is assumed to be the (num_tokens, embedding_dim) numpy array from above
if embeddings.ndim == 2 and embeddings.shape[0] > 0:
    # normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-12)
    cosine_sim = emb_norm @ emb_norm.T
    # numerical stability
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cosine_sim, cmap='viridis', vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)

    # annotate cells with numeric values
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            val = cosine_sim[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f"{val:.2f}", ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.title("Cosine similarity heatmap of token embeddings")
    plt.tight_layout()
    out_path = "tokens/token_embeddings_cosine_heatmap.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Cosine similarity heatmap saved to {out_path}")
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)
else:
    print("Skipping cosine similarity heatmap: embeddings shape unexpected.")