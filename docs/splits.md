# Data splits

This repo trains and evaluates under two split protocols: the **version split**
(`selection: Version`) and the **song-stratified 10-fold** set
(`selection: SongStratified`). Both are driven by plain text index files that
live in the corpus root (`$PANSORI_DATA_ROOT`, default `../Pansori_Data/`), not
in this repository — the repo only holds the code that reads them
(`trainers.py:201` `init_selection`) and the labels under `data/Label/`.

Every index line starts with the 8-hex-char `hash_key` of a recording; loaders
key off `filename.split("-")[0]`. Keys not present in the current
`data.data_dir` are dropped silently, so per-modality counts can be slightly
lower than the raw line counts below (mel/pesto/midi cover different subsets).

```
$PANSORI_DATA_ROOT/
├── pansori_version_split/   train.txt (347) / val.txt (31) / test.txt (18)
├── song_stratified/         ch_{1,2}.txt hb_{1,2}.txt jb_{1,2}.txt
│                            sc_{1,2}.txt sg_{1,2}.txt  +  version_test.txt (18)
└── pansori_random_fold/     fold_01.txt … fold_10.txt   (selection: SharedFold)
```

---

## 1. Version split — `selection: Version`

**Question it answers:** does the model generalize to *another singer's rendition
of a 대목 it has already heard*? Test recordings are different performances of
pieces that also appear in training, sung by different 명창.

- Config: `train.version_split_dir: "${paths.data_root}/pansori_version_split"`
- Files: `train.txt` / `val.txt` / `test.txt`
- One fold only; `fold_names = ['version']`.

| Split | Recordings |
|---|---|
| train | 347 |
| val   | 31 |
| test  | **18** ("Version Test", VT) |
| total | 396 (no hash key appears in two splits) |

Full per-recording listing — all 396 recordings with singer, 대목 and duration,
grouped by split and 바탕, plus each VT recording paired with its counterpart
rendition on the train side: [`version_split.txt`](version_split.txt).

The 18 VT recordings, and the same 대목 sung by someone else on the train side:

| VT singer | 대목 | matching rendition in train/val |
|---|---|---|
| 김소희 | 춘향가 십장가 | 성우향 |
| 이일주 | 심청가 심봉사 개안 | 완창반 (`03_심청가 -- 심봉사_개안`) |
| 안숙선 | 적벽가 장승타령 | 김연수 |
| 오정숙 | 흥보가 온갖 비단이 나옴 | 이일주 |
| 박동진 | 적벽가 조조가 조자룡에게 쫓겨 도망가는데 | 김연수 |
| 이일주 | 심청가 곽씨 죽음 | 완창반 (`01/02_심청가`) |
| 오정숙 | 흥보가 둘째 박을 탐 | 이일주 |
| 오정숙 | 흥보가 흥보 살릴 중이 내려옴 | 이일주 |
| 이일주 | 심청가 장승상댁 | 오정숙 |
| 오정숙 | 흥보가 셋째 박에서 양귀비가 나옴 | 이일주 |
| 박록주 | 흥보가 도승이 흥보에게 집터를 잡아주는 데 | 성우향 |
| 이일주 | 심청가 방아타령 | 완창반 |
| 안숙선 | 적벽가 새타령 | 박동진 |
| 성창순 | 춘향가 적성가 | 김소희 |
| 김수연 | 심청가 범피중류 | 김소희 |
| 박초월 | 수궁가 토끼 수궁 들어가는 대목 | 박양덕 |
| 이일주 | 심청가 심봉사 목욕 | 완창반 |
| 이일주 | 심청가 심봉사 물에 빠짐 | 완창반 |

(All 18 have a ≥0.6 title-similarity counterpart on the train side; the "완창반"
rows are the numbered full-length 심청가 sets whose filenames carry no singer
field.)

**Note the leakage this protocol accepts on purpose:** the *piece* is shared
between train and test, only the *performance* is held out. That is the point of
the protocol — it isolates singer/recording variation — but it is not a
song-level generalization claim. That claim is what split 2 is for.

---

## 2. Song-stratified 10-fold — `selection: SongStratified`

**Question it answers:** does the model generalize to a 바탕 it has never seen?
The five pansori 바탕 are the stratification unit; a held-out 바탕 contributes
nothing to training in its folds.

- Config: `train.song_stratified_dir: "${paths.data_root}/song_stratified"`
- Files: `<genre>_1.txt`, `<genre>_2.txt` — each 바탕's recordings pre-split into
  two halves.

| code | 바탕 | half 1 | half 2 | total |
|---|---|---|---|---|
| `ch` | 춘향가 | 37 | 37 | 74 |
| `hb` | 흥보가 | 20 | 20 | 40 |
| `jb` | 적벽가 | 33 | 34 | 67 |
| `sc` | 심청가 | 79 | 79 | 158 |
| `sg` | 수궁가 | 13 | 14 | 27 |
| | | | | **366** |

Fold construction (`trainers.py:212`): for each held-out 바탕 *g*,
`train = every recording of the other four 바탕`, and the two halves of *g* swap
roles between val and test — giving 2 folds per 바탕, 10 total. Fold names are
`{genre}_{val}v{test}t`.

| # | fold | train | val | test |
|---|---|---|---|---|
| 1 | `ch_1v2t` | 292 | 37 | 37 |
| 2 | `ch_2v1t` | 292 | 37 | 37 |
| 3 | `hb_1v2t` | 326 | 20 | 20 |
| 4 | `hb_2v1t` | 326 | 20 | 20 |
| 5 | `jb_1v2t` | 299 | 33 | 34 |
| 6 | `jb_2v1t` | 299 | 34 | 33 |
| 7 | `sc_1v2t` | 208 | 79 | 79 |
| 8 | `sc_2v1t` | 208 | 79 | 79 |
| 9 | `sg_1v2t` | 339 | 13 | 14 |
| 10 | `sg_2v1t` | 339 | 14 | 13 |

Because the two halves swap, **every one of the 366 recordings is a test item in
exactly one fold** — that is what makes `pooled_eval_song_stratified.py` a valid
corpus-wide pooled evaluation.

Held-out 바탕 size drives the train size, so fold difficulty is uneven: the 심청가
folds train on 208 recordings, the 수궁가 folds on 339.

Full per-recording listing — each 바탕 half with the folds it serves in, and a
per-fold train/val/test breakdown with the test and val recordings spelled out:
[`song_stratified.txt`](song_stratified.txt).

### `version_test.txt` inside `song_stratified/`

Same 18 recordings as `pansori_version_split/test.txt` (verified identical hash
set), stored one per line as `hash_key 명창 — 대목` for readability. It is *not* a
fold; `trainers.py:626` `_load_version_test_hashes()` loads it only when
`selection == SongStratified`, and the trainer then pulls the VT recordings'
results out of whatever fold happens to be testing them, aggregating them into a
`Version Test Summary` run (`version_test_results.csv`, pooled posteriorgrams).
This lets one song-stratified sweep report a VT number comparable to the
version-split model's. All 18 fall inside the folds
(ch_1:1, ch_2:1, hb_1:4, hb_2:1, jb_1:1, jb_2:2, sc_1:2, sc_2:5, sg_1:1), so
each is tested exactly once per sweep.

---

## 3. Other selections (present in code, not used for headline results)

| Value | Source | Notes |
|---|---|---|
| `SharedFold` | `pansori_random_fold/fold_01..10.txt` | fixed random 10-fold shared across modalities; rolling window test=*i*, val=*i*+1 |
| `RandomSplit` | none | single 80/10/10 random split, seeded by `train.random_seed` |
| `KFold` | none | sklearn `KFold` over all loaded hashes; no val split |
| `Artist` | `data/Stratify/stratify.csv` | **not included in the repo** — path hardcoded at `trainers.py:242` |

---

## Which checkpoints came from which split

| Checkpoint dir | Split | Folds |
|---|---|---|
| `weights/frame/Mel_Original_Version`, `Mel_Sep_Version`, `Pesto_Version`, `weights/midi/MIDI_Version`, `weights/cmert/layer08_10k_version` | Version | 1 |
| `weights/frame/Mel_Original_Song_Stratified`, `Mel_Sep_Stratified`, `Pesto_Song_Stratified`, `weights/midi/MIDI_Song_Stratified`, `weights/cmert/layer08_song_stratified` | SongStratified | 10 |

## Reproducing these numbers

Both listings are generated from the index files and `data/Label/label.csv`:

```bash
python scripts/analysis/dump_splits.py          # -> docs/version_split.txt, docs/song_stratified.txt
```

```bash
# what each index file contains
wc -l $PANSORI_DATA_ROOT/pansori_version_split/*.txt
wc -l $PANSORI_DATA_ROOT/song_stratified/*.txt

# fold membership as the trainer sees it (after data_dir filtering)
python train.py --config configs/frame/mel_base.yaml   # prints per-fold counts at startup
```
