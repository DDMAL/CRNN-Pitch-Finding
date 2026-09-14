# pitch_finding_crnn

End-to-end pitch recognition for medieval chant manuscripts (square/neume
notation on a 4-line staff): given a scanned manuscript page and its
[MEI](https://music-encoding.org/) ground truth, train a CRNN+CTC model that
reads a line image and outputs the sequence of notes it contains — no
separate staff-line detection or symbol-classification stage.

**See [`DOCUMENTATION.md`](DOCUMENTATION.md)** for the full writeup:
motivation, a detailed pipeline walkthrough, both label schemes and why
`pos_seq` exists, complete results (including a low-data ablation and a
data-quality investigation that shaped the current filtering), known
limitations, and future directions. This README only covers enough to
install and run the code.

## Setup

```bash
pip install -r requirements.txt
```

You'll also need:
- Page scans of the target manuscript (one JPEG per page/folio).
- MEI ground truth for those pages (`<zone>` elements giving each note's
  pixel bounding box, `<sb>` marking system breaks, `<clef>` marking clef
  changes, `<nc>` neume components carrying `pname`/`oct`).

`yolo_weights/best.pt` is bundled (~39MB) — a `music-blocks` detector
trained zero-shot on other manuscripts (see `DOCUMENTATION.md` for details
and example training data). Pass `--yolo_weights` to use a different one.

## Usage

```bash
# 1. Build the line-crop dataset (--yolo_weights defaults to the bundled
#    yolo_weights/best.pt; pass your own to override)
python3 build_dataset.py \
  --mei_dir /path/to/mei --img_dir /path/to/scans \
  --mei_glob '*.mei' --folio_prefix '' \
  --out_dir data

# 2. Train (pick a label scheme: pname_seq = absolute pitch, pos_seq =
#    relative staff position — see DOCUMENTATION.md for what these mean)
python3 train.py --label_field pos_seq   --data_dir data --out_dir results_pos   --epochs 60
python3 train.py --label_field pname_seq --data_dir data --out_dir results_pname --epochs 60

# 3. Evaluate
python3 eval.py --data_dir data --results_dir results_pos --split test

# 4. (optional) render best/worst predictions with GT alignment
python3 visualize.py --data_dir data --results_dir results_pos --split test
```

`--folio_prefix` strips a common prefix from MEI filenames to recover the
folio id used to look up the matching page scan (e.g. `CH-E_611_087r.mei` →
`087r.jpg` needs `--folio_prefix CH-E_611_`).

## Results

Test accuracy (macro / micro) on the held-out test set:

| label scheme | test accuracy |
|---|---|
| `pname_seq` (absolute pitch) | 0.941 / 0.939 |
| `pos_seq` (relative staff position) | 0.953 / 0.952 |

`pos_seq` reaches a slightly higher accuracy and converges faster, and the
gap widens substantially with less training data. See
[`DOCUMENTATION.md`](DOCUMENTATION.md#results) for the full results
(including why these particular numbers need a caveat before citing them,
and what a fair like-for-like comparison shows instead).
