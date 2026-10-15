"""Cut the sung (non-아니리) stretches out of recordings, using the per-frame
predictions that infer_ensemble.py / infer_<rep>.py write.

    python infer/export_segments.py \\
        --pred_dir  outputs/infer \\
        --audio_dir /path/to/original_wavs \\
        --out       /path/to/singing

For every `<stem>_ensemble.csv` (or `--pred_suffix`) in --pred_dir, the matching
`<stem>.wav` in --audio_dir is cut at the frame boundaries of each kept run and
written unchanged (same sample rate, channels and subtype) to --out as
`<stem>_<start>-<end>.wav`. A manifest CSV lists every segment with its time
span and mode composition.

Frame-level predictions flicker, so two clean-ups run before cutting:
  1. excluded runs (아니리 by default) shorter than --min_gap seconds are
     treated as singing, so a 0.2 s blip does not split a phrase;
  2. kept runs shorter than --min_len seconds are dropped.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def runs(mask):
    """Boolean array -> list of (value, start, end) with end exclusive."""
    out, i = [], 0
    while i < len(mask):
        j = i
        while j < len(mask) and mask[j] == mask[i]:
            j += 1
        out.append((bool(mask[i]), i, j))
        i = j
    return out


def keep_mask(labels, exclude, fps, min_gap, min_len):
    keep = ~np.isin(labels, list(exclude))
    for val, s, e in runs(keep):
        if not val and (e - s) / fps < min_gap:
            keep[s:e] = True
    for val, s, e in runs(keep):
        if val and (e - s) / fps < min_len:
            keep[s:e] = False
    return keep


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument('--pred_dir', required=True, type=Path)
    p.add_argument('--audio_dir', required=True, type=Path,
                   help='recordings to cut, named <stem>.wav')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--pred_suffix', default='_ensemble.csv')
    p.add_argument('--exclude', nargs='+', default=['아니리', 'Unknown'],
                   help='predicted labels that are not kept (default: 아니리 Unknown)')
    p.add_argument('--min_gap', type=float, default=1.0,
                   help='excluded runs shorter than this (s) count as kept')
    p.add_argument('--min_len', type=float, default=2.0,
                   help='kept runs shorter than this (s) are dropped')
    p.add_argument('--manifest', type=Path, default=None,
                   help='manifest CSV path (default: <out>/../<out name>_segments.csv)')
    args = p.parse_args()

    preds = sorted(args.pred_dir.glob(f'*{args.pred_suffix}'))
    if not preds:
        raise SystemExit(f"No *{args.pred_suffix} in {args.pred_dir}")
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or args.out.parent / f'{args.out.name}_segments.csv'

    rows, n_audio_sec = [], 0.0
    for pred in preds:
        stem = pred.name[:-len(args.pred_suffix)]
        audio = args.audio_dir / f'{stem}.wav'
        if not audio.exists():
            print(f"  skipping {pred.name}: no {audio.name} in {args.audio_dir}")
            continue
        df = pd.read_csv(pred)
        fps = 1.0 / float(df['time_sec'].iloc[1] - df['time_sec'].iloc[0])
        labels = df['pred_label'].to_numpy()
        keep = keep_mask(labels, args.exclude, fps, args.min_gap, args.min_len)

        info = sf.info(str(audio))
        n_audio_sec += info.duration
        wav, sr = sf.read(str(audio), always_2d=True, dtype='float64')
        for val, s, e in runs(keep):
            if not val:
                continue
            t0, t1 = s / fps, min(e / fps, info.duration)
            name = f'{stem.strip()}_{t0:06.1f}-{t1:06.1f}.wav'
            sf.write(str(args.out / name), wav[int(round(t0 * sr)):int(round(t1 * sr))],
                     sr, subtype=info.subtype)
            seg = pd.Series(labels[s:e]).value_counts(normalize=True)
            rows.append({
                'source': audio.name, 'segment': name,
                'start_sec': round(t0, 2), 'end_sec': round(t1, 2), 'dur_sec': round(t1 - t0, 2),
                'majority_mode': seg.index[0],
                **{f'pct_{c}': round(100 * seg.get(c, 0.0), 1) for c in ('우조', '계면조', '창조', '아니리')},
            })

    out = pd.DataFrame(rows)
    out.to_csv(manifest, index=False)
    kept = out['dur_sec'].sum() if len(out) else 0.0
    print(f"  {len(out)} segments from {out['source'].nunique() if len(out) else 0} recordings "
          f"-> {args.out}")
    print(f"  kept {kept / 60:.1f} of {n_audio_sec / 60:.1f} min; manifest -> {manifest}")


if __name__ == '__main__':
    main()
