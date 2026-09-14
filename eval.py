"""
eval.py — evaluate a trained CRNN+CTC model on held-out lines, with a
per-folio breakdown. Reads label_field/architecture back from the training
run's config.json (written by train.py) so decoding always matches how that
specific model was trained — no separate --label_field flag to keep in sync.

acc here = 1 - CER (edit distance / len(GT)) per line, averaged.

Usage:
    python3 eval.py --results_dir results_pname --split test
    python3 eval.py --results_dir results_pos --split test
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import torch
from torch.utils.data import DataLoader, Subset

from dataset import LineDataset, collate_fn, vocab_for
from model import CRNN
from train import greedy_decode, edit_distance

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data_dir', type=Path, default=HERE / 'data')
    ap.add_argument('--results_dir', type=Path, default=HERE / 'results',
                     help='包含 config.json + best_model.pt 的训练输出目录')
    ap.add_argument('--out_dir', type=Path, default=None, help='默认与 results_dir 相同')
    ap.add_argument('--split', choices=['val', 'test'], default='test')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_width', type=int, default=1600)
    ap.add_argument('--max_lines', type=int, default=None, help='冒烟测试：只跑前 N 行')
    ap.add_argument('--num_threads', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    out_dir = args.out_dir or args.results_dir

    torch.set_num_threads(args.num_threads)
    device = torch.device(args.device)

    with open(args.results_dir / 'config.json') as f:
        config = json.load(f)
    label_field = config['label_field']
    token_to_idx, idx_to_token, num_classes = vocab_for(label_field, args.data_dir)
    assert num_classes == config['num_classes'], (
        'vocab_for() 得到的类别数和训练时 config.json 记录的不一致——data_dir 是否换过（比如 '
        'pos_vocab.json 被重新生成过、取值范围变了）？')

    manifest = args.data_dir / f'manifest_{args.split}.jsonl'
    ds = LineDataset(manifest, args.data_dir, token_to_idx, label_field=label_field,
                      target_height=config['img_height'], max_width=args.max_width)
    if args.max_lines:
        ds = Subset(ds, range(min(args.max_lines, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = CRNN(num_classes, img_height=config['img_height'],
                 lstm_hidden=config['lstm_hidden'], lstm_layers=config['lstm_layers']).to(device)
    model.load_state_dict(torch.load(args.results_dir / 'best_model.pt', map_location=device))
    model.eval()

    rows = []  # per-line results
    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            input_widths = batch['input_widths']
            log_probs = model(images)
            input_lengths = torch.clamp(CRNN.output_length(input_widths), min=1)
            preds = greedy_decode(log_probs.cpu(), input_lengths, idx_to_token)

            offset = 0
            for i, tl in enumerate(batch['target_lengths'].tolist()):
                tgt = tuple(idx_to_token[i2] for i2 in batch['targets'][offset:offset + tl].tolist())
                offset += tl
                pred = preds[i]
                d = edit_distance(pred, tgt)
                acc = 1.0 - d / len(tgt) if tgt else 0.0
                rows.append({
                    'folio': batch['folios'][i],
                    'system_idx': batch['system_idxs'][i],
                    'n_notes': len(tgt),
                    # space-joined so pos_seq's ints stay unambiguous in the CSV
                    # (pname_seq's chars read fine either way, e.g. "c d e").
                    'target': ' '.join(map(str, tgt)),
                    'pred': ' '.join(map(str, pred)),
                    'edit_distance': d,
                    'acc': acc,
                })

    out_dir.mkdir(parents=True, exist_ok=True)

    per_line_path = out_dir / f'per_line_{args.split}.csv'
    with open(per_line_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['folio', 'system_idx', 'n_notes', 'target',
                                                'pred', 'edit_distance', 'acc'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    by_folio = defaultdict(list)
    for r in rows:
        by_folio[r['folio']].append(r)

    per_folio_path = out_dir / f'per_folio_{args.split}.csv'
    with open(per_folio_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['folio', 'n_lines', 'n_notes',
                                                'mean_acc', 'total_edit_distance'])
        writer.writeheader()
        for folio in sorted(by_folio):
            lines = by_folio[folio]
            writer.writerow({
                'folio': folio,
                'n_lines': len(lines),
                'n_notes': sum(l['n_notes'] for l in lines),
                'mean_acc': sum(l['acc'] for l in lines) / len(lines),
                'total_edit_distance': sum(l['edit_distance'] for l in lines),
            })

    total_notes = sum(r['n_notes'] for r in rows)
    total_edits = sum(r['edit_distance'] for r in rows)
    macro_acc = sum(r['acc'] for r in rows) / len(rows) if rows else 0.0
    micro_acc = 1.0 - total_edits / total_notes if total_notes else 0.0

    folio_accs = {f: sum(l['acc'] for l in ls) / len(ls) for f, ls in by_folio.items()}
    worst_folios = sorted(folio_accs.items(), key=lambda kv: kv[1])[:5]
    best_folios = sorted(folio_accs.items(), key=lambda kv: kv[1], reverse=True)[:5]

    summary = {
        'split': args.split,
        'label_field': label_field,
        'n_lines': len(rows),
        'n_folios': len(by_folio),
        'total_notes': total_notes,
        'acc_macro': macro_acc,   # mean of per-line accuracy
        'acc_micro': micro_acc,   # 1 - (total edits / total notes), weights long lines more
        'worst_folios': worst_folios,
        'best_folios': best_folios,
        'model_path': str(args.results_dir / 'best_model.pt'),
    }
    with open(out_dir / f'summary_{args.split}.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'\nper-line results -> {per_line_path}')
    print(f'per-folio results -> {per_folio_path}')


if __name__ == '__main__':
    main()
