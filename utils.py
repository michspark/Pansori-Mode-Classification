import numpy as np
import random
import unicodedata
from pathlib import Path
import json

def create_kfold_splits(song_list, k=5, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    song_list_shuffled = song_list.copy()
    random.shuffle(song_list_shuffled)

    # Divide into 10 chunks for 80/10/10 split with 5 folds
    num_chunks = k * 2  # 10 chunks for 5 folds
    chunk_size = len(song_list_shuffled) // num_chunks
    chunks = []

    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_chunks - 1 else len(song_list_shuffled)
        chunks.append(song_list_shuffled[start:end])

    folds = []
    for i in range(k):
        # Test: 1 chunk (10%)
        test_idx = i * 2
        test_songs = chunks[test_idx]

        # Val: 1 chunk (10%)
        val_idx = i * 2 + 1
        val_songs = chunks[val_idx]

        # Train: remaining 8 chunks (80%)
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
