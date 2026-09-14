# pitch_finding_crnn

End-to-end pitch recognition for medieval chant manuscripts (square/neume
notation on a 4-line staff): given a scanned manuscript page and its
[MEI](https://music-encoding.org/) ground truth, train a CRNN+CTC model that
reads a line image and outputs the sequence of notes it contains — no
separate staff-line detection or symbol-classification stage.

Earlier approaches in this line of work tried the classical two-stage
pipeline (detect staff lines geometrically, then classify each note) and
consistently struggled on this notation style (4-line square notation isn't
what most staff-line detectors, tuned for modern 5-line notation, expect).
Skipping straight to an end-to-end sequence model on line-level crops turned
out to work far better in practice.

## How it works

1. **`build_dataset.py`** — for every page:
   1. Parse the MEI ground truth into per-system note sequences, tracking
      `<clef>` elements so each note gets both an absolute pitch name
      (`pname_seq`) and a clef-relative staff position (`pos_seq`).
   2. Run a YOLO detector over the page scan to find each system's visual
      bounding box (the `music-blocks` class).
   3. Match each detected box to its MEI system by top-to-bottom order —
      never by trusting a note's own MEI coordinates, since a chunk of those
      are wrong in this corpus (they point at blank parchment). A system
      with no confident match falls back to a crop derived from its own
      notes' coordinate percentiles instead.
   4. Flag (and later exclude) any line whose crop has too little ink, or
      whose notes have a suspiciously large horizontal gap — both signs the
      crop or the underlying annotation may be unreliable.
   5. **Once a fallback happens on a page, distrust the rest of that page
      too.** The system↔detection matching in step 3 walks top-to-bottom in
      one pass; a fallback means it lost sync with the detector for at least
      one system, so later systems' matches on the *same page* aren't
      trustworthy either even if they nominally found a detection. Flag the
      fallback line and every subsequent line on that page.

   Concrete effect of this filtering on the manuscript this was developed
   on (290 pages, 4199 raw line crops):

   | stage | lines remaining |
   |---|---|
   | all cropped lines | 4199 |
   | − low ink / large note gap (steps 1–4) | 3785 |
   | − fallback + rest of page (step 5) | **3513** (83.7% of raw) |

   giving a final train/val/test split of **2823 / 393 / 297** lines (down
   from 3021 / 393 / 371 without step 5). Step 5 alone removes 272 more lines
   on top of what step 4 had already dropped — split unevenly: train −6.6%,
   val unaffected (0%), test −20.0%. The uneven split-level impact is just
   where the 39 pages with at least one fallback happened to land in the
   train/val/test folio assignment, not a systematic bias in the filter
   itself.

2. **`train.py`** — a CNN → BiLSTM×2 → CTC model reads the padded line image
   and predicts the note sequence directly. Supports two label schemes,
   switchable via `--label_field`:
   - `pname_seq`: absolute pitch name (c/d/e/f/g/a/b) — the standard target
   - `pos_seq`: relative staff position (an integer — which line/space the
     notehead sits on). This is clef-aware but octave-independent and
     monotonic in the note's vertical pixel position, which made it
     noticeably faster to converge and modestly more accurate than
     `pname_seq` in controlled comparisons, especially with limited
     training data (see Results).
3. **`eval.py`** — evaluate a trained model on held-out lines (macro/micro
   accuracy = 1 − character error rate), with a per-folio breakdown.

## Setup

```bash
pip install -r requirements.txt
```

You'll also need:
- Page scans of the target manuscript (one JPEG per page/folio).
- MEI ground truth for those pages (`<zone>` elements giving each note's
  pixel bounding box, `<sb>` marking system breaks, `<clef>` marking clef
  changes, `<nc>` neume components carrying `pname`/`oct`).

`yolo_weights/best.pt` (bundled in this repo, ~39MB) is a YOLO detector for
a `music-blocks` class — the bounding box of each staff+text line region on
a page. It was trained (in a separate project, not included here) on 19
hand-labeled pages from roughly 10 other square-notation chant manuscripts
held at various libraries, and used *zero-shot* on the manuscript this
pipeline was developed on — it had never seen that manuscript's handwriting
during training. If you're pointing this pipeline
at a very different notation style or page layout, this detector may not
transfer well; swap in `--yolo_weights` any other checkpoint that localizes
staff systems on the page (any object detector framework works, as long as
its output exposes per-box class + xyxy coordinates the way `ultralytics`
does — see `build_dataset.py`'s YOLO-calling code if you need to adapt it).

Two examples of what that training data looked like (page images from other
manuscripts, kept anonymous here, with ground-truth boxes drawn on: blue =
`music-blocks`, the class `build_dataset.py` actually uses; orange =
`chant-text`, part of the original annotation scheme but unused by this
pipeline):

<p>
  <img src="docs/yolo_training_example_1.jpg" width="45%" alt="YOLO training example 1: page with music-blocks (blue) and chant-text (orange) boxes">
  <img src="docs/yolo_training_example_2.jpg" width="45%" alt="YOLO training example 2: page with music-blocks (blue) and chant-text (orange) boxes">
</p>

## Usage

```bash
# 1. Build the line-crop dataset (--yolo_weights defaults to the bundled
#    yolo_weights/best.pt; pass your own to override)
python3 build_dataset.py \
  --mei_dir /path/to/mei --img_dir /path/to/scans \
  --mei_glob '*.mei' --folio_prefix '' \
  --out_dir data

# 2. Train (pick a label scheme)
python3 train.py --label_field pos_seq   --data_dir data --out_dir results_pos   --epochs 60
python3 train.py --label_field pname_seq --data_dir data --out_dir results_pname --epochs 60

# 3. Evaluate
python3 eval.py --data_dir data --results_dir results_pos --split test
```

`--folio_prefix` strips a common prefix from MEI filenames to recover the
folio id used to look up the matching page scan (e.g. `CH-E_611_087r.mei` →
`087r.jpg` needs `--folio_prefix CH-E_611_`).

## Results

On the held-out test set produced by this pipeline (297 lines — the filtered
count from "How it works" above):

| label scheme | val accuracy (epoch reached) | test accuracy (macro / micro) |
|---|---|---|
| `pname_seq` (absolute pitch) | 0.888 (epoch 26) | 0.941 / 0.939 |
| `pos_seq` (relative staff position) | 0.901 (epoch 21) | 0.953 / 0.952 |

The relative-position scheme reaches a slightly higher accuracy and
converges faster. The gap widens substantially as training data shrinks —
in a low-data ablation (random subsets of the *unfiltered* training set),
with only 30 training lines `pos_seq` accuracy (0.57) was nearly double
`pname_seq`'s (0.33) — consistent with `pos_seq` being an easier, more
directly visually-grounded target to learn.

**Calibrate your expectations of the 0.94–0.95 test numbers above**: they
partly reflect that step 5 in "How it works" also curates the *test* set,
removing lines whose ground truth is least likely to be reliable. Measured
on an unfiltered 371-line test set (i.e. the same population, before step 5
trims it to 297), the same trained models score 0.801/0.828 and 0.812/0.840
respectively — a modest, more representative estimate of accuracy on a
line of unknown/unverified quality. Both numbers are legitimate; they just
answer different questions ("how good is the model on lines we're confident
are labeled correctly" vs. "how good is the model on an arbitrary line").

### Example predictions

Rendered with `visualize.py`, which aligns predicted vs. ground-truth tokens
per line and color-codes them (green = match, red = substitution, yellow =
insertion/deletion). 5 best + 5 worst test-set lines by accuracy, for each
label scheme:

**`pos_seq`** — [best](docs/predictions_pos/) (`best_*.jpg`) /
[worst](docs/predictions_pos/) (`worst_*.jpg`):

<p>
  <img src="docs/predictions_pos/best_05_269r_14.jpg" width="32%">
  <img src="docs/predictions_pos/best_06_269r_13.jpg" width="32%">
  <img src="docs/predictions_pos/best_07_269r_11.jpg" width="32%">
</p>
<p>
  <img src="docs/predictions_pos/worst_00_195r_08.jpg" width="32%">
  <img src="docs/predictions_pos/worst_02_195r_09.jpg" width="32%">
  <img src="docs/predictions_pos/worst_04_195r_14.jpg" width="32%">
</p>

**`pname_seq`** — [best](docs/predictions_pname/) (`best_*.jpg`) /
[worst](docs/predictions_pname/) (`worst_*.jpg`):

<p>
  <img src="docs/predictions_pname/best_05_269r_14.jpg" width="32%">
  <img src="docs/predictions_pname/best_06_269r_13.jpg" width="32%">
  <img src="docs/predictions_pname/best_07_269r_11.jpg" width="32%">
</p>
<p>
  <img src="docs/predictions_pname/worst_00_195r_08.jpg" width="32%">
  <img src="docs/predictions_pname/worst_02_195r_09.jpg" width="32%">
  <img src="docs/predictions_pname/worst_03_195r_14.jpg" width="32%">
</p>

The full set of 5+5 for each scheme is in `docs/predictions_pos/` and
`docs/predictions_pname/`. The worst lines are the same handful of pages for
both schemes (e.g. `195r`) — an unusually large decorated-initial/blank-staff
region eats up most of the crop width there, leaving little room for real
notes relative to what the label expects. A genuine, understandable failure
mode, not a broken crop; regenerate with:

```bash
python3 visualize.py --data_dir data --results_dir results_pos \
  --split test --n_best 5 --n_worst 5 --max_width 1600 --format jpg
```

## Known limitations

- The system↔detection matching (`match_detections_to_systems`) is greedy
  and order-preserving; it can occasionally lock onto the wrong neighboring
  system rather than its own even without ever triggering a fallback (~16%
  of lines by one audit, before dataset step 5's page-level cutoff). CRNN+CTC
  training tolerates this reasonably well — the majority of lines are still
  correctly matched — but it's a real, only partially mitigated source of
  label noise. A globally optimal (DP) matching algorithm was tried and
  *reduced* the mismatch rate substantially, but paradoxically also produced
  meaningfully worse trained models for a reason that was never fully
  root-caused; the simpler greedy version is kept here since it's what
  produced the reported results.
- A minority of MEI note coordinates in this corpus are themselves
  unreliable (don't correspond to real ink positions on the scan) — this
  affects any diagnostic that trusts individual note coordinates, not just
  the crop-matching step.
- The `pos_seq` vs `pname_seq` comparison above uses manuscript-specific
  data; the general finding (relative position is easier to learn,
  especially in low-data regimes) is plausibly transferable to other
  square-notation corpora, but hasn't been tested on one.
