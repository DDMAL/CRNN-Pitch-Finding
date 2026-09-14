"""
train.py — train the CRNN+CTC line-level pitch recognizer.

Switches the prediction target between two label schemes via --label_field,
trained on the *same* crops/splits so the label scheme is the only variable:
  pname_seq : absolute pitch name (c/d/e/f/g/a/b)
  pos_seq   : relative staff position (integer; which line/space the notehead
              sits on, from build_dataset.py's clef-aware conversion)
See README.md for the motivation and dataset.py's vocab_for()/pos_vocab.json.

Usage:
    # smoke test: overfit a handful of lines to sanity-check the pipeline
    python3 train.py --label_field pos_seq --limit_train 20 --overfit_val_is_train --epochs 200 --log_every 20

    # full training run, absolute-pitch baseline (for a same-data comparison)
    python3 train.py --label_field pname_seq --out_dir results_pname --epochs 100

    # full training run, relative-position scheme
    python3 train.py --label_field pos_seq --out_dir results_pos --epochs 100

    # low-data-regime run: random 100-line training subset, real held-out val/test
    python3 train.py --label_field pos_seq --limit_train 100 --out_dir results_pos_n100 --epochs 200
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

# Must be set before torch (and its OpenMP/MKL backends) initializes. This is
# a shared HPC login node (64 cores) — without a cap, PyTorch's default
# "use all visible cores" thread pool badly oversubscribes small CPU ops
# (BiLSTM especially), making even a tiny smoke-test run pathologically slow
# and hogging the whole node. Full training runs should go through sbatch
# with --cpus-per-task matching this cap.
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import torch
from torch.utils.data import DataLoader, Subset

from dataset import LineDataset, collate_fn, vocab_for, BLANK_IDX
from model import CRNN

HERE = Path(__file__).resolve().parent


def greedy_decode(log_probs, input_lengths, idx_to_token):
    """log_probs: (T, B, C). Returns list of predicted token tuples (collapse
    repeats, drop blanks — standard CTC greedy decoding). Tuples rather than
    strings so this works identically for pname chars and pos ints."""
    preds = log_probs.argmax(dim=2).transpose(0, 1)  # (B, T)
    out = []
    for b, length in enumerate(input_lengths):
        seq = preds[b, :length].tolist()
        collapsed = []
        prev = None
        for s in seq:
            if s != prev and s != BLANK_IDX:
                collapsed.append(s)
            prev = s
        out.append(tuple(idx_to_token[i] for i in collapsed))
    return out


def edit_distance(a, b):
    """Levenshtein distance between two sequences (plain DP, sequences here
    are at most ~80 long so this is cheap). Works on any indexable sequence
    of hashable/comparable elements — strings or tuples of ints alike."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def seq_acc(preds, targets):
    """1 - CER (edit distance / len(target)), averaged per-line then
    macro-averaged. Scheme-agnostic (works for pname_seq or pos_seq)."""
    accs = []
    for p, t in zip(preds, targets):
        if len(t) == 0:
            continue
        d = edit_distance(p, t)
        accs.append(1.0 - d / len(t))
    return sum(accs) / len(accs) if accs else 0.0


def run_epoch(model, loader, ctc_loss, optimizer, device, train, idx_to_token):
    model.train(mode=train)
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []

    for batch in loader:
        images = batch['images'].to(device)
        input_widths = batch['input_widths']
        targets = batch['targets'].to(device)
        target_lengths = batch['target_lengths']

        log_probs = model(images)  # (T, B, C)
        input_lengths = torch.clamp(CRNN.output_length(input_widths), min=1)

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        with torch.no_grad():
            preds = greedy_decode(log_probs.detach().cpu(), input_lengths, idx_to_token)
        offset = 0
        for i, tl in enumerate(target_lengths.tolist()):
            tgt = tuple(idx_to_token[i2] for i2 in batch['targets'][offset:offset + tl].tolist())
            offset += tl
            all_preds.append(preds[i])
            all_targets.append(tgt)

    return total_loss / max(n_batches, 1), seq_acc(all_preds, all_targets)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_dir', type=Path, default=HERE / 'data')
    ap.add_argument('--out_dir', type=Path, default=HERE / 'results')
    ap.add_argument('--label_field', choices=['pname_seq', 'pos_seq'], default='pos_seq',
                     help='预测目标：pname_seq=绝对音名（原方案）, pos_seq=相对谱线位置（新方案）')
    ap.add_argument('--train_manifest', type=Path, default=None,
                     help='默认 <data_dir>/manifest_train.jsonl；可指定绝对路径覆盖（调试用）')
    ap.add_argument('--val_manifest', type=Path, default=None,
                     help='默认 <data_dir>/manifest_val.jsonl；可指定绝对路径覆盖（调试用）')
    ap.add_argument('--img_height', type=int, default=64)
    ap.add_argument('--lstm_hidden', type=int, default=256)
    ap.add_argument('--lstm_layers', type=int, default=2)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--max_width', type=int, default=1600, help='裁剪/缩放后行图像的最大宽度上限')
    ap.add_argument('--limit_train', type=int, default=None,
                     help='只用 N 条训练数据（随机子集，seed 复现）；默认仍用真实 val 集评估，'
                          '过拟合冒烟测试请加 --overfit_val_is_train')
    ap.add_argument('--overfit_val_is_train', action='store_true',
                     help='冒烟测试专用：让 val 集直接等于 limit_train 的子集本身（验证 pipeline 能过拟合）')
    ap.add_argument('--log_every', type=int, default=1)
    ap.add_argument('--val_every', type=int, default=1,
                     help='每隔几个 epoch 跑一次验证集；验证集大小固定，不随 --limit_train 变小，'
                          '小数据量+多 epoch 的实验里调大这个值能省很多时间')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--num_threads', type=int, default=4, help='CPU 训练时的线程数上限（登录节点上不要调大）')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    token_to_idx, idx_to_token, num_classes = vocab_for(args.label_field, args.data_dir)

    train_manifest = args.train_manifest or (args.data_dir / 'manifest_train.jsonl')
    val_manifest = args.val_manifest or (args.data_dir / 'manifest_val.jsonl')
    train_ds = LineDataset(train_manifest, args.data_dir, token_to_idx,
                            label_field=args.label_field,
                            target_height=args.img_height, max_width=args.max_width)
    val_ds = LineDataset(val_manifest, args.data_dir, token_to_idx,
                          label_field=args.label_field,
                          target_height=args.img_height, max_width=args.max_width)

    if args.limit_train:
        n = min(args.limit_train, len(train_ds))
        rng = random.Random(args.seed)
        idxs = rng.sample(range(len(train_ds)), n)  # random subset, not just the first N
        train_ds = Subset(train_ds, idxs)
        if args.overfit_val_is_train:
            val_ds = train_ds  # smoke test: check we can overfit the same lines we train on

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_fn)

    device = torch.device(args.device)
    model = CRNN(num_classes, img_height=args.img_height,
                 lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ctc_loss = torch.nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    print(f'label_field={args.label_field}  num_classes={num_classes}  '
          f'train lines: {len(train_ds)}, val lines: {len(val_ds)}, device: {device}')

    # eval.py reads this back so it always decodes with the same vocab the
    # model was trained on, instead of relying on a manually-passed flag.
    with open(args.out_dir / 'config.json', 'w') as f:
        json.dump({
            'label_field': args.label_field, 'num_classes': num_classes,
            'img_height': args.img_height, 'lstm_hidden': args.lstm_hidden,
            'lstm_layers': args.lstm_layers,
        }, f, indent=2)

    history = []
    best_val_acc = -1.0
    val_loss, val_acc = None, None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, ctc_loss, optimizer, device,
                                           train=True, idx_to_token=idx_to_token)

        do_val = (epoch % args.val_every == 0) or (epoch == args.epochs)
        if do_val:
            val_loss, val_acc = run_epoch(model, val_loader, ctc_loss, optimizer, device,
                                           train=False, idx_to_token=idx_to_token)
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                         'val_loss': val_loss, 'val_acc': val_acc})

        if epoch % args.log_every == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            val_str = f'val_loss={val_loss:.4f} val_acc={val_acc:.3f}' if do_val else '(val skipped)'
            print(f'epoch {epoch:4d}/{args.epochs}  train_loss={train_loss:.4f} '
                  f'train_acc={train_acc:.3f}  {val_str}  ({elapsed:.0f}s)')

        if do_val and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.out_dir / 'best_model.pt')

    with open(args.out_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f'best val_acc = {best_val_acc:.3f}  (label_field={args.label_field}, '
          f'results saved to {args.out_dir})')


if __name__ == '__main__':
    main()
