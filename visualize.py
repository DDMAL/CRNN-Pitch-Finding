"""
visualize.py — render a handful of eval.py's per-line predictions against the
line image, with the predicted/target token sequences aligned and
color-coded (match / substitution / insertion / deletion) underneath.

Reads eval.py's `results/per_line_{split}.csv` directly (no need to re-run
the model), picks the best/worst N lines by accuracy, and draws each line
image with the aligned target/prediction tokens underneath. Works for either
label scheme (`pname_seq` or `pos_seq`) since eval.py's CSV already stores
both as whitespace-separated tokens.

Usage:
    python3 visualize.py --data_dir data --results_dir results_pos --split test --n_worst 3 --n_best 3
    python3 visualize.py --data_dir data --results_dir results_pos --split test --folio 087r --system_idx 0
"""
import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

MATCH_COLOR = (40, 160, 40)
SUB_COLOR = (200, 40, 40)
GAP_COLOR = (180, 140, 0)
BG_COLOR = (255, 255, 255)
CHAR_W = 26
ROW_H = 30
PAD = 12
LABEL_W = 55


def align(pred, target):
    """Needleman-Wunsch-style alignment (unit cost) between two token
    sequences. Returns a list of (pred_token_or_None, target_token_or_None,
    op), op in {'match','sub','ins','del'} — 'ins' = extra token in pred,
    'del' = token missing from pred that target has."""
    n, m = len(pred), len(target)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred[i - 1] == target[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])

    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and pred[i - 1] == target[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
            ops.append((pred[i - 1], target[j - 1], 'match'))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append((pred[i - 1], target[j - 1], 'sub'))
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append((pred[i - 1], None, 'ins'))
            i -= 1
        else:
            ops.append((None, target[j - 1], 'del'))
            j -= 1
    ops.reverse()
    return ops


def _font():
    try:
        return ImageFont.truetype('DejaVuSansMono.ttf', 18)
    except OSError:
        return ImageFont.load_default()


def render_line(img_path, pred, target, out_path, title=None, max_width=None):
    ops = align(pred, target)
    font = _font()

    img = Image.open(img_path).convert('RGB')
    text_w = max(len(ops) * CHAR_W + LABEL_W + PAD, img.width)
    title_h = ROW_H if title else 0
    canvas = Image.new('RGB', (text_w, img.height + 2 * ROW_H + PAD + title_h), BG_COLOR)
    canvas.paste(img, (0, title_h))

    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((PAD, 4), title, fill=(0, 0, 0), font=font)

    pred_y = title_h + img.height + 4
    target_y = pred_y + ROW_H

    draw.text((PAD, pred_y), 'pred:', fill=(120, 120, 120), font=font)
    draw.text((PAD, target_y), 'GT:', fill=(120, 120, 120), font=font)

    for k, (p, t, op) in enumerate(ops):
        x = LABEL_W + k * CHAR_W
        color = {'match': MATCH_COLOR, 'sub': SUB_COLOR, 'ins': GAP_COLOR, 'del': GAP_COLOR}[op]
        draw.text((x, pred_y), p if p else '·', fill=color, font=font)
        draw.text((x, target_y), t if t else '·', fill=color, font=font)

    if max_width and canvas.width > max_width:
        scale = max_width / canvas.width
        canvas = canvas.resize((max_width, max(1, int(canvas.height * scale))), Image.LANCZOS)
    if out_path.suffix.lower() in ('.jpg', '.jpeg'):
        canvas.save(out_path, quality=88)
    else:
        canvas.save(out_path)
    return ops


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', type=Path, required=True)
    ap.add_argument('--results_dir', type=Path, required=True)
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    ap.add_argument('--n_worst', type=int, default=3)
    ap.add_argument('--n_best', type=int, default=3)
    ap.add_argument('--folio', default=None, help='只画指定 folio（配合 --system_idx）')
    ap.add_argument('--system_idx', type=int, default=None)
    ap.add_argument('--out_dir', type=Path, default=None)
    ap.add_argument('--max_width', type=int, default=None,
                     help='超过此宽度就等比缩放；配合 --format jpg 能显著减小文件体积')
    ap.add_argument('--format', choices=['png', 'jpg'], default='png')
    ap.add_argument('--title_prefix', default=None,
                     help='加在每张图标题最前面的标签（比如手稿名/标签方案），方便混在一起看的时候区分来源')
    args = ap.parse_args()

    out_dir = args.out_dir or (args.results_dir / 'viz')
    out_dir.mkdir(parents=True, exist_ok=True)

    per_line_path = args.results_dir / f'per_line_{args.split}.csv'
    with open(per_line_path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['acc'] = float(r['acc'])
        r['system_idx'] = int(r['system_idx'])

    if args.folio is not None:
        selected = [(r, 'pick') for r in rows
                    if r['folio'] == args.folio and
                    (args.system_idx is None or r['system_idx'] == args.system_idx)]
    else:
        rows_sorted = sorted(rows, key=lambda r: r['acc'])
        worst = [(r, 'worst') for r in rows_sorted[:args.n_worst]]
        best = [(r, 'best') for r in rows_sorted[::-1][:args.n_best]]
        selected = worst + best

    if not selected:
        print('no matching lines found')
        return

    for rank, (r, tag) in enumerate(selected):
        img_name = f"{r['folio']}_{r['system_idx']:02d}.jpg"
        img_path = args.data_dir / 'lines' / img_name
        if not img_path.exists():
            print(f'[WARN] missing image {img_path}, skipping')
            continue
        out_name = f"{tag}_{rank:02d}_{r['folio']}_{r['system_idx']:02d}.{args.format}"
        out_path = out_dir / out_name
        title = f"{r['folio']} #{r['system_idx']}  acc={r['acc']:.2f}  ({tag})"
        if args.title_prefix:
            title = f"{args.title_prefix} — {title}"
        render_line(img_path, r['pred'].split(), r['target'].split(), out_path, title=title,
                    max_width=args.max_width)
        print(f'  {out_name}')

    print(f'rendered {len(selected)} images -> {out_dir}')


if __name__ == '__main__':
    main()
