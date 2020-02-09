# %%
import matplotlib

import dill as pickle
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)
# %%
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar

def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

# %%
# enc-dec attention
with open("test/enc_dec_attn_list.pkl", 'rb') as f:
    attn = pickle.load(f)

attn_numpy = [[tensor.detach().cpu().numpy() for tensor in tensors] for tensors in attn]
attn_numpy_array = np.array(attn_numpy) # (test_size, num_layers, beam_size, trg_seq_len, src_seq_len)
mean_array = np.mean(np.mean(attn_numpy_array, axis=0), axis=1)  # (num_layers, trg_seq_len, src_seq_len)

# %%
rows = [1, 2, 3, 4, 5, "<EOS>"]
cols = ["<BOS>", 1, 2, 3, 4, 5, "<EOS>"]

for i in range(len(mean_array)):
    fig, ax = plt.subplots()
    im, cbar = heatmap(mean_array[i], rows, cols, ax=ax, cmap="YlGn", cbarlabel="weight")
    texts = annotate_heatmap(im, valfmt="{x:.1f}")
    fig.tight_layout()
    plt.show()

# %%
# dec attention
with open("test_3/dec_attn_list.pkl", 'rb') as f:
    dec_attn = pickle.load(f)

dec_attn_numpy = [[tensor.detach().cpu().numpy() for tensor in tensors] for tensors in dec_attn]
dec_attn_array = np.array(dec_attn_numpy) # (test_size, num_layers, beam_size, trg_seq_len, trg_seq_len)
dec_attn_mean_array = np.mean(np.mean(dec_attn_array, axis=0), axis=1)

# %%
rows = [1, 2, 3, 4, 5, 6]
cols = [1, 2, 3, 4, 5, 6]

for i in range(len(dec_attn_mean_array)):
    fig, ax = plt.subplots()
    im, cbar = heatmap(dec_attn_mean_array[i], rows, cols, ax=ax, cmap="YlGn", cbarlabel="weight")
    texts = annotate_heatmap(im, valfmt="{x:.1f}")
    fig.tight_layout()
    plt.show()

# %%
