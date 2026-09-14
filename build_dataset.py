"""
build_dataset.py — build the line-crop training set for the CRNN+CTC pitch
recognizer: detect each staff system's visual extent on the page (YOLO), crop
it out, and pair each crop with two label sequences derived from the MEI
ground truth:
  pname_seq : absolute pitch name per note (c/d/e/f/g/a/b)
  pos_seq   : relative staff position per note (integer; which line/space the
              notehead sits on — clef-aware, octave-independent)

Pipeline:
  1. Parse each MEI file in document order, splitting into systems on <sb>,
     tracking <clef> elements to convert each note's pitch into both an
     absolute pname and a clef-relative staff position.
  2. Run a YOLO "music-blocks" detector over the page scan to find the visual
     bounding box of each system (zero-shot: the detector was trained on
     other manuscripts, never this one).
  3. Match detected boxes to MEI systems by top-to-bottom order (greedy,
     never by trusting individual note coordinates — those are sometimes
     corrupted in this corpus, floating over blank parchment). A system with
     no confident match falls back to a crop derived from its own notes'
     coordinate percentiles.
  4. Flag (and exclude from train/val/test) any line whose crop has too
     little ink, or whose notes have a suspiciously large horizontal gap
     (both signs the crop or the annotation itself may be unreliable).
  5. Once a system on a page falls back to the coordinate-based crop (step 3),
     the sequential system<->detection matching has gone out of sync for at
     least one system on that page — flag (and exclude) that line *and every
     later system on the same page* too, not just the fallback line itself.

Known limitation: the system<->detection matching is greedy and can
occasionally lock onto the wrong neighboring system even without ever
triggering a fallback (~16% of lines by one audit, before step 5's filter) —
CRNN+CTC training tolerates this reasonably well since the majority of lines
are still correctly matched, but it's a real, unresolved source of label
noise that step 5 only partially mitigates. See the project README's "Known
limitations" section.

Requires a YOLO checkpoint that detects a `music-blocks` class (staff+text
line regions) on the target manuscript's page scans — see README for how
this project's weights were trained. Needs ultralytics (pulls in opencv).

Usage:
    python3 build_dataset.py \\
        --mei_dir /path/to/mei --img_dir /path/to/page/scans \\
        --yolo_weights /path/to/best.pt --out_dir data
"""
import argparse
import json
import random
import re
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

MEI_NS = 'http://www.music-encoding.org/ns/mei'
PNAMES = set('cdefgab')

HERE = Path(__file__).resolve().parent
DEFAULT_MEI_DIR = HERE / 'mei'
DEFAULT_IMG_DIR = HERE / 'scans'
DEFAULT_OUT_DIR = HERE / 'data'
DEFAULT_YOLO_WEIGHTS = HERE / 'yolo_weights' / 'best.pt'
MUSIC_BLOCKS_CLASS = 'music-blocks'

# Diatonic step relative to c0 (c0=0, d0=1, ..., b0=6, c1=7, ...).
_PNAME_STEP = {'c': 0, 'd': 1, 'e': 2, 'f': 3, 'g': 4, 'a': 5, 'b': 6}


def _diatonic_step(pname, oct_):
    return oct_ * 7 + _PNAME_STEP[pname]


def _staff_pos_from_pitch(pname, oct_, clef_shape, clef_line):
    """Relative staff position: 1=line1(bottom), 2=space1, 3=line2, ...,
    7=line4(top) on a 4-line staff; values outside [1,7] are ledger-line
    notes. Only defined for C and F clefs (the only ones this corpus uses).

    C-clef on line L: c4 is at staff_pos = 2*L - 1
    F-clef on line L: f3 is at staff_pos = 2*L - 1
    """
    cs = (clef_shape or 'C').upper()
    if cs == 'C':
        ref_step, ref_pos = _diatonic_step('c', 4), 2 * clef_line - 1
    elif cs == 'F':
        ref_step, ref_pos = _diatonic_step('f', 3), 2 * clef_line - 1
    else:
        return None
    return ref_pos + (_diatonic_step(pname, oct_) - ref_step)


def _percentile(values, pct):
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (pct / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    if lo == hi:
        return xs[lo]
    frac = k - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac


def parse_systems(mei_path):
    """Walk a MEI file in document order and split into systems on <sb>,
    tracking <clef> elements (default C-clef line 4 if none seen yet) to
    attach a relative staff position to each note alongside its absolute
    pitch. Returns list of systems, each a list of nc dicts (reading order):
        {'pname': str, 'oct': int, 'pos': int, 'bbox': (ulx, uly, lrx, lry)}
    Notes under a non-C/F clef (never observed in this corpus) are dropped."""
    tree = ET.parse(str(mei_path))
    root = tree.getroot()

    zone_dict = {}
    for z in root.iter(f'{{{MEI_NS}}}zone'):
        zid = None
        for k in z.attrib:
            if k.endswith('}id'):
                zid = z.attrib[k]
                break
        if zid is None:
            continue
        try:
            zone_dict[zid] = (int(z.get('ulx')), int(z.get('uly')),
                               int(z.get('lrx')), int(z.get('lry')))
        except (TypeError, ValueError):
            continue

    systems = []
    current = []
    current_clef = {'shape': 'C', 'line': 4}
    for elem in root.iter():
        tag = elem.tag.replace(f'{{{MEI_NS}}}', '')
        if tag == 'sb':
            if current:
                systems.append(current)
            current = []
        elif tag == 'clef':
            shape = elem.get('shape', 'C')
            try:
                line_int = int(elem.get('line', '4'))
            except ValueError:
                line_int = 4
            current_clef = {'shape': shape, 'line': line_int}
        elif tag == 'nc':
            facs = elem.get('facs', '')
            pname = (elem.get('pname') or '').lower()
            oct_str = elem.get('oct')
            if not facs.startswith('#') or pname not in PNAMES or oct_str is None:
                continue
            bbox = zone_dict.get(facs[1:])
            if bbox is None:
                continue
            try:
                oct_ = int(oct_str)
            except ValueError:
                continue
            pos = _staff_pos_from_pitch(pname, oct_, current_clef['shape'], current_clef['line'])
            if pos is None:
                continue
            current.append({'pname': pname, 'oct': oct_, 'pos': pos, 'bbox': bbox})
    if current:
        systems.append(current)
    return systems


def _system_y_range(system, pct_lo=5, pct_hi=95):
    ys = [(nc['bbox'][1] + nc['bbox'][3]) / 2.0 for nc in system]
    return _percentile(ys, pct_lo), _percentile(ys, pct_hi)


def fallback_crop_boxes(systems, img_w, img_h, pad_x, pad_y):
    """Crop derived purely from this system's own notes' coordinate
    percentiles — used when the YOLO detector can't be matched confidently."""
    y_ranges = [_system_y_range(s) for s in systems]
    n = len(systems)
    boxes = []
    for i, system in enumerate(systems):
        y_lo, y_hi = y_ranges[i]
        uly = y_lo - pad_y
        if i > 0:
            uly = max(uly, (y_ranges[i - 1][1] + y_lo) / 2.0)
        uly = max(0, uly)
        lry = y_hi + pad_y
        if i < n - 1:
            lry = min(lry, (y_hi + y_ranges[i + 1][0]) / 2.0)
        lry = min(img_h, lry)
        xs_ulx = [nc['bbox'][0] for nc in system]
        xs_lrx = [nc['bbox'][2] for nc in system]
        ulx = max(0, min(xs_ulx) - pad_x)
        lrx = min(img_w, max(xs_lrx) + pad_x)
        if lry <= uly or lrx <= ulx:
            boxes.append(None)
            continue
        boxes.append((int(ulx), int(uly), int(lrx), int(lry)))
    return boxes


def merge_split_detections(detected_boxes, y_merge_dist=100, min_width=800):
    """The detector sometimes splits one wide system into two side-by-side
    boxes (near-identical y-center) if its training data skewed narrower than
    the target manuscript's pages. Cluster boxes whose y-centers are within
    `y_merge_dist` px of each other and take the union bbox per cluster; drop
    merged boxes narrower than `min_width` (spurious small detections, e.g. a
    marginal decorated initial, not a real system)."""
    boxes = sorted(detected_boxes, key=lambda b: (b[1] + b[3]) / 2.0)
    clusters = []
    for b in boxes:
        yc = (b[1] + b[3]) / 2.0
        if clusters and abs(yc - clusters[-1]['yc']) <= y_merge_dist:
            c = clusters[-1]
            c['box'] = (min(c['box'][0], b[0]), min(c['box'][1], b[1]),
                        max(c['box'][2], b[2]), max(c['box'][3], b[3]))
            c['yc'] = (c['box'][1] + c['box'][3]) / 2.0
        else:
            clusters.append({'box': b, 'yc': yc})
    return [c['box'] for c in clusters if c['box'][2] - c['box'][0] >= min_width]


def match_detections_to_systems(systems, detected_boxes, img_h, max_dist_frac=0.4):
    """Greedy, order-preserving match between MEI systems (top-to-bottom
    document order) and detected boxes (sorted by y-center) — purely by
    position/order, never by trusting an individual note's own coordinates.

    Known limitation: this greedy strategy can occasionally lock onto the
    wrong neighboring system when detection counts don't line up exactly
    with system counts (see README's "Known limitations"). A globally
    optimal assignment (e.g. DP/edit-distance-style alignment) fixes most
    such cases but was found to *not* improve downstream training accuracy
    for reasons that were never fully root-caused — kept here for
    reproducibility of the reported results.

    Returns list aligned with `systems`: each entry is either a detected box
    (ulx, uly, lrx, lry) or None (no confident match — caller should fall back).
    """
    expected_y = [sum(_system_y_range(s)) / 2.0 for s in systems]
    order = sorted(range(len(systems)), key=lambda i: expected_y[i])

    dets = sorted(detected_boxes, key=lambda b: (b[1] + b[3]) / 2.0)
    max_dist = img_h * max_dist_frac

    result = [None] * len(systems)
    di = 0
    for i in order:
        y_exp = expected_y[i]
        while di < len(dets) and (dets[di][1] + dets[di][3]) / 2.0 < y_exp - max_dist:
            di += 1
        if di < len(dets) and abs((dets[di][1] + dets[di][3]) / 2.0 - y_exp) <= max_dist:
            result[i] = dets[di]
            di += 1
    return result


def ink_fraction(gray_crop, thresh=100):
    hist = gray_crop.histogram()
    dark = sum(hist[:thresh])
    total = gray_crop.width * gray_crop.height
    return dark / total if total else 0.0


def max_gap_info(system):
    xs = sorted((nc['bbox'][0] + nc['bbox'][2]) / 2.0 for nc in system)
    if len(xs) < 2:
        return None
    gaps = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    return max(gaps), statistics.median(gaps)


def find_image(img_dir, folio):
    candidates = [folio]
    m = re.match(r'^0*(\d+)(\D.*)$', folio)
    if m:
        candidates.append(m.group(1) + m.group(2))
    for cand in candidates:
        p = img_dir / f'{cand}.jpg'
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mei_dir', type=Path, default=DEFAULT_MEI_DIR)
    ap.add_argument('--img_dir', type=Path, default=DEFAULT_IMG_DIR)
    ap.add_argument('--out_dir', type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument('--mei_glob', default='*.mei', help='MEI 文件名匹配模式')
    ap.add_argument('--folio_prefix', default='', help='从 MEI 文件名提取 folio 时要去掉的前缀，例如 "CH-E_611_"')
    ap.add_argument('--yolo_weights', type=Path, default=DEFAULT_YOLO_WEIGHTS)
    ap.add_argument('--yolo_conf', type=float, default=0.25)
    ap.add_argument('--match_max_dist_frac', type=float, default=0.04,
                     help='匹配阈值：检测框与期望 y 位置的最大允许距离，占图像高度的比例')
    ap.add_argument('--det_pad_x', type=int, default=20)
    ap.add_argument('--det_pad_y', type=int, default=20)
    ap.add_argument('--y_merge_dist', type=int, default=100,
                     help='y-center 相差小于此值(px)的检测框合并为一个（修复系统被左右拆成两个框的问题）')
    ap.add_argument('--min_detection_width', type=int, default=800,
                     help='合并后宽度小于此值的检测框视为虚假检测（如页边装饰花体字），丢弃不参与匹配')
    ap.add_argument('--pad_x', type=int, default=60, help='fallback 裁剪的水平 padding (px)')
    ap.add_argument('--pad_y', type=int, default=100, help='fallback 裁剪的竖直 padding (px)')
    ap.add_argument('--min_notes', type=int, default=2)
    ap.add_argument('--ink_gray_thresh', type=int, default=100)
    ap.add_argument('--min_ink_frac', type=float, default=0.02)
    ap.add_argument('--max_gap_ratio', type=float, default=8.0)
    ap.add_argument('--max_gap_px', type=float, default=500.0)
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--test_frac', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--jpeg_quality', type=int, default=90)
    ap.add_argument('--max_folios', type=int, default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    yolo = YOLO(str(args.yolo_weights))
    class_names = yolo.names
    music_cls_id = next(i for i, n in class_names.items() if n == MUSIC_BLOCKS_CLASS)

    lines_dir = args.out_dir / 'lines'
    lines_dir.mkdir(parents=True, exist_ok=True)

    mei_files = sorted(args.mei_dir.glob(args.mei_glob))
    if args.max_folios:
        mei_files = mei_files[:args.max_folios]

    manifest = []
    n_folios_ok = 0
    n_missing_img = 0
    n_skipped_short = 0
    n_skipped_badbox = 0
    n_detected = 0
    n_fallback = 0

    for mei_path in mei_files:
        folio = mei_path.stem
        if args.folio_prefix and folio.startswith(args.folio_prefix):
            folio = folio[len(args.folio_prefix):]
        img_path = find_image(args.img_dir, folio)
        if img_path is None:
            print(f'[WARN] no image for folio {folio}, skipping')
            n_missing_img += 1
            continue

        try:
            systems = parse_systems(mei_path)
        except ET.ParseError as e:
            print(f'[WARN] failed to parse {mei_path.name}: {e}')
            continue

        n_before = len(systems)
        systems = [s for s in systems if len(s) >= args.min_notes]
        n_skipped_short += n_before - len(systems)
        if not systems:
            continue

        with Image.open(img_path) as img:
            img_w, img_h = img.size

            yolo_result = yolo.predict(str(img_path), conf=args.yolo_conf, verbose=False)[0]
            raw_boxes = []
            for b in yolo_result.boxes:
                if int(b.cls.item()) != music_cls_id:
                    continue
                raw_boxes.append(tuple(b.xyxy[0].tolist()))

            merged_boxes = merge_split_detections(
                raw_boxes, y_merge_dist=args.y_merge_dist, min_width=args.min_detection_width)
            detected_boxes = [(x1 - args.det_pad_x, y1 - args.det_pad_y,
                                x2 + args.det_pad_x, y2 + args.det_pad_y)
                               for x1, y1, x2, y2 in merged_boxes]

            matched = match_detections_to_systems(
                systems, detected_boxes, img_h, args.match_max_dist_frac)
            fallback_boxes = fallback_crop_boxes(systems, img_w, img_h, args.pad_x, args.pad_y)

            boxes = []
            sources = []
            for det, fb in zip(matched, fallback_boxes):
                if det is not None:
                    ulx = max(0, det[0]); uly = max(0, det[1])
                    lrx = min(img_w, det[2]); lry = min(img_h, det[3])
                    if lrx > ulx and lry > uly:
                        boxes.append((int(ulx), int(uly), int(lrx), int(lry)))
                        sources.append('detected')
                        continue
                boxes.append(fb)
                sources.append('fallback' if fb is not None else None)

            fallback_seen = False  # reset per page — see 'after_fallback' below
            for sys_idx, (system, box, source) in enumerate(zip(systems, boxes, sources)):
                if box is None:
                    n_skipped_badbox += 1
                    continue
                if source == 'detected':
                    n_detected += 1
                else:
                    n_fallback += 1

                pname_seq = ''.join(nc['pname'] for nc in system)
                oct_seq = ''.join(str(nc['oct']) for nc in system)
                pos_seq = [nc['pos'] for nc in system]

                out_name = f'{folio}_{sys_idx:02d}.jpg'
                out_path = lines_dir / out_name
                crop = img.crop(box)
                crop.save(out_path, quality=args.jpeg_quality)

                frac = ink_fraction(crop.convert('L'), args.ink_gray_thresh)
                reasons = []
                if frac < args.min_ink_frac:
                    reasons.append('low_ink')

                # Once a system on this page falls back to the coordinate-based
                # crop (no confident detection match), the sequential
                # system<->detection matching has gone out of sync for at
                # least one system — treat every later system on the same
                # page as untrustworthy too, not just this one line.
                if source == 'fallback' or fallback_seen:
                    reasons.append('after_fallback')
                fallback_seen = fallback_seen or source == 'fallback'

                gap_info = max_gap_info(system)
                max_gap = med_gap = None
                if gap_info is not None:
                    max_gap, med_gap = gap_info
                    if max_gap > args.max_gap_ratio * med_gap and max_gap > args.max_gap_px:
                        reasons.append('sparse_annotation')

                manifest.append({
                    'image': f'lines/{out_name}',
                    'folio': folio,
                    'system_idx': sys_idx,
                    'n_notes': len(system),
                    'pname_seq': pname_seq,
                    'oct_seq': oct_seq,
                    'pos_seq': pos_seq,
                    'bbox': list(box),
                    'crop_source': source,
                    'width': box[2] - box[0],
                    'height': box[3] - box[1],
                    'ink_frac': round(frac, 5),
                    'max_gap': max_gap,
                    'median_gap': med_gap,
                    'flag_reasons': reasons,
                    'flagged': bool(reasons),
                })

        n_folios_ok += 1
        if n_folios_ok % 25 == 0:
            print(f'  ...{n_folios_ok} folios processed, {len(manifest)} lines so far '
                  f'(detected={n_detected}, fallback={n_fallback})')

    if n_skipped_short:
        print(f'[note] {n_skipped_short} systems skipped for having < {args.min_notes} notes')
    if n_skipped_badbox:
        print(f'[note] {n_skipped_badbox} systems skipped: no detection match AND degenerate fallback box')
    if n_missing_img:
        print(f'[note] {n_missing_img} folios skipped: no matching image file')
    print(f'[note] crop source: {n_detected} detected (YOLO), {n_fallback} fallback (nc-percentile)')

    folios = sorted({m['folio'] for m in manifest})
    rng = random.Random(args.seed)
    rng.shuffle(folios)
    n_val = round(len(folios) * args.val_frac)
    n_test = round(len(folios) * args.test_frac)
    val_folios = set(folios[:n_val])
    test_folios = set(folios[n_val:n_val + n_test])
    train_folios = set(folios[n_val + n_test:])
    splits = {'train': train_folios, 'val': val_folios, 'test': test_folios}
    split_of = {}
    for name, fset in splits.items():
        for f in fset:
            split_of[f] = name
    for m in manifest:
        m['split'] = split_of[m['folio']]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_flagged = sum(1 for m in manifest if m['flagged'])
    n_low_ink = sum(1 for m in manifest if 'low_ink' in m['flag_reasons'])
    n_sparse = sum(1 for m in manifest if 'sparse_annotation' in m['flag_reasons'])
    n_after_fallback = sum(1 for m in manifest if 'after_fallback' in m['flag_reasons'])
    if n_after_fallback:
        print(f'[note] {n_after_fallback} lines dropped: fallback crop or after '
              f'the first fallback on that page')
    for name in ('train', 'val', 'test'):
        rows = [m for m in manifest if m['split'] == name and not m['flagged']]
        with open(args.out_dir / f'manifest_{name}.jsonl', 'w') as f:
            for m in rows:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')
    with open(args.out_dir / 'manifest_all.jsonl', 'w') as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    with open(args.out_dir / 'flagged.jsonl', 'w') as f:
        for m in manifest:
            if m['flagged']:
                f.write(json.dumps(m, ensure_ascii=False) + '\n')
    if n_flagged:
        print(f'[note] {n_flagged}/{len(manifest)} lines flagged and excluded from '
              f'train/val/test (low_ink={n_low_ink}, sparse_annotation={n_sparse}, '
              f'after_fallback={n_after_fallback})')

    kept = [m for m in manifest if not m['flagged']]
    notes_per_line = [m['n_notes'] for m in kept]
    vocab = sorted({c for m in kept for c in m['pname_seq']})
    pos_vocab = sorted({p for m in kept for p in m['pos_seq']})
    with open(args.out_dir / 'pos_vocab.json', 'w') as f:
        json.dump(pos_vocab, f)
    summary = {
        'n_folios_processed': n_folios_ok,
        'n_folios_missing_image': n_missing_img,
        'n_lines': len(manifest),
        'n_lines_kept': len(kept),
        'n_lines_flagged': n_flagged,
        'n_lines_flagged_low_ink': n_low_ink,
        'n_lines_flagged_sparse_annotation': n_sparse,
        'n_lines_flagged_after_fallback': n_after_fallback,
        'n_lines_skipped_short': n_skipped_short,
        'n_lines_skipped_badbox': n_skipped_badbox,
        'n_crop_source_detected': n_detected,
        'n_crop_source_fallback': n_fallback,
        'notes_per_line': {
            'min': min(notes_per_line) if notes_per_line else None,
            'max': max(notes_per_line) if notes_per_line else None,
            'mean': sum(notes_per_line) / len(notes_per_line) if notes_per_line else None,
        },
        'vocab': vocab,
        'pos_vocab': pos_vocab,
        'split_n_folios': {k: len(v) for k, v in splits.items()},
        'split_n_lines': {name: sum(1 for m in kept if m['split'] == name)
                           for name in ('train', 'val', 'test')},
        'yolo_weights': str(args.yolo_weights),
        'yolo_conf': args.yolo_conf,
        'match_max_dist_frac': args.match_max_dist_frac,
        'pad_x': args.pad_x, 'pad_y': args.pad_y,
        'min_notes': args.min_notes,
        'ink_gray_thresh': args.ink_gray_thresh,
        'min_ink_frac': args.min_ink_frac,
        'max_gap_ratio': args.max_gap_ratio,
        'max_gap_px': args.max_gap_px,
        'seed': args.seed,
    }
    with open(args.out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
