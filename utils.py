import numpy as np
import random
import unicodedata
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import csv

def create_kfold_splits(song_list, k=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    song_list_shuffled = song_list.copy()
    random.shuffle(song_list_shuffled)

    num_chunks = k
    chunk_size = len(song_list_shuffled) // num_chunks
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else len(song_list_shuffled)
        chunks.append(song_list_shuffled[start:end])

    folds = []
    for i in range(k):

        test_idx = i
        test_songs = chunks[test_idx]


        val_idx = (i + 1) % k
        val_songs = chunks[val_idx]


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
    _CLASS_COLORS = [_CMAP(7), _CMAP(0), _CMAP(3), _CMAP(2)]

    T = gt.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(16, 5), sharex=True,
                             constrained_layout=True)
    fig.suptitle(song_name, fontsize=10)

    gt_labels = np.argmax(gt, axis=1)
    import matplotlib.colors as mcolors
    gt_cmap = mcolors.ListedColormap([_CLASS_COLORS[i] for i in range(4)])
    axes[0].imshow(gt_labels[np.newaxis, :], aspect='auto', origin='lower',
                   cmap=gt_cmap, vmin=-0.5, vmax=3.5, interpolation='nearest',
                   extent=[0, T, -0.5, 0.5])
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(['class'])
    axes[0].set_title('Ground Truth')

    legend_handles = [Patch(color=_CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(4)]
    axes[0].legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.7)

    im = axes[1].imshow(np.flipud(pred_probs.T), aspect='auto', origin='lower',
                        vmin=0, vmax=1, cmap='gray_r', interpolation='nearest',
                        extent=[0, T, -0.5, 3.5])
    axes[1].set_yticks([0, 1, 2, 3])
    axes[1].set_yticklabels(CLASS_NAMES[::-1])
    axes[1].set_title('Predicted Posteriorgram')

    axes[-1].set_xlabel('Frame')
    fig.colorbar(im, ax=axes[1], label='Probability', shrink=0.8)

    return fig

def plot_confusion_matrix(song_data, save_path):
    """Build and save a confusion matrix from all songs in song_data."""
    CLASS_NAMES = ['no label', 'Ujo', 'GMjo', 'ANR']
    all_gt, all_pred = [], []
    for data in song_data.values():
        all_gt.append(np.argmax(data['gt'], axis=1))
        all_pred.append(np.argmax(data['pred_probs'], axis=1))
    all_gt = np.concatenate(all_gt)
    all_pred = np.concatenate(all_pred)

    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(all_gt, all_pred):
        cm[t, p] += 1

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap='Blues')
    fig.colorbar(im, ax=ax, label='Recall')
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (normalised by true class)')
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                    ha='center', va='center', fontsize=8,
                    color='white' if cm_norm[i, j] > 0.6 else 'black')
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_test_csv(segment_results, csv_path):
    """Save per-segment test results to CSV."""
    if not segment_results:
        return
    fieldnames = ['song_name', 'time_range', 'start_sec', 'end_sec',
                  'loss', 'acc', 'f1_ujoh', 'f1_gyemyeon', 'f1_aniri', 'f1_macro']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(segment_results)


