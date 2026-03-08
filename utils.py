import numpy as np
import random
import unicodedata
from pathlib import Path
import json
import matplotlib.pyplot as plt

def create_kfold_splits(song_list, k=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    song_list_shuffled = song_list.copy()
    random.shuffle(song_list_shuffled)

    # Divide into k chunks for 80/10/10 split
    # Each fold: test=1 chunk (10%), val=1 chunk (10%), train=k-2 chunks (80%)
    num_chunks = k
    chunk_size = len(song_list_shuffled) // num_chunks
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else len(song_list_shuffled)
        chunks.append(song_list_shuffled[start:end])

    folds = []
    for i in range(k):
        # Test: 1 chunk (10%)
        test_idx = i
        test_songs = chunks[test_idx]

        # Val: 1 chunk (10%)
        val_idx = (i + 1) % k
        val_songs = chunks[val_idx]

        # Train: remaining k-2 chunks (80%)
        train_songs = []
        for j in range(num_chunks):
            if j != test_idx and j != val_idx:
                train_songs.extend(chunks[j])

        folds.append({
            'train': train_songs,
            'val': val_songs,
            'test': test_songs
        })

    for idx, fold in enumerate(folds):
        print(f"  Fold {idx + 1}: Train={len(fold['train'])}, Val={len(fold['val'])}, Test={len(fold['test'])}")

    return folds

def get_all_song_names(data_dir, label_json):
    data_dir = Path(data_dir)
    with open(label_json, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    names = []
    for item in raw:
        fu = unicodedata.normalize('NFC', item['file_upload'])
        midi_name = fu.rsplit('.', 1)[0] + '_vocal.mid'
        if (data_dir / midi_name).exists():
            names.append(midi_name)
    return names


def plot_posteriorgram(song_name, gt, pred_probs):
    """
    gt         : (T, 3) numpy array, one-hot ground truth
    pred_probs : (T, 3) numpy array, softmax probabilities
    Returns a matplotlib Figure.
    """
    CLASS_NAMES = ['no label', 'Ujo', 'GMjo', 'ANR']
    _CMAP = plt.cm.get_cmap('tab10')
    _CLASS_COLORS = [_CMAP(7), _CMAP(0), _CMAP(3), _CMAP(2)]  # grey, blue, red, green

    T = gt.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True,
                             constrained_layout=True)
    fig.suptitle(song_name, fontsize=10)

    # --- GT: categorical display (argmax → discrete color per class) ---
    gt_labels = np.argmax(gt, axis=1)  # (T,)
    import matplotlib.colors as mcolors
    gt_cmap = mcolors.ListedColormap([_CLASS_COLORS[i] for i in range(4)])
    axes[0].imshow(gt_labels[np.newaxis, :], aspect='auto', origin='lower',
                   cmap=gt_cmap, vmin=-0.5, vmax=3.5, interpolation='nearest',
                   extent=[0, T, -0.5, 0.5])
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(['class'])
    axes[0].set_title('Ground Truth')
    # legend patches
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=_CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(4)]
    axes[0].legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.7)

    # --- Pred: posteriorgram heatmap per class ---
    # Flip rows so top→bottom order is: no label, Ujo, GMjo, ANR
    im = axes[1].imshow(np.flipud(pred_probs.T), aspect='auto', origin='lower',
                        vmin=0, vmax=1, cmap='RdYlGn', interpolation='nearest',
                        extent=[0, T, -0.5, 3.5])
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(CLASS_NAMES[::-1])
    axes[1].set_title('Predicted Posteriorgram')

    axes[-1].set_xlabel('Frame')
    fig.colorbar(im, ax=axes[1], label='Probability', shrink=0.8)

    return fig


