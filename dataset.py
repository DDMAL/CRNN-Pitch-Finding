"""
dataset.py — PyTorch Dataset for CRNN/CTC line-level pitch recognition.

The training target can be either scheme build_dataset.py writes into the manifest:
  pname_seq : absolute pitch name, string of 7 letters (c/d/e/f/g/a/b)
  pos_seq   : relative staff position, list of ints (clef-aware, see
              build_dataset.py's _staff_pos_from_pitch)
See README.md for the motivation.

Loads (line_image, target_sequence) pairs from build_dataset.py's manifest
files. Images are resized to a fixed height (aspect ratio preserved) so a
CNN can be applied uniformly; width stays variable and is padded per-batch.
"""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

BLANK_IDX = 0

PNAMES = sorted('abcdefg')
PNAME_TO_IDX = {p: i + 1 for i, p in enumerate(PNAMES)}  # 0 reserved for CTC blank
IDX_TO_PNAME = {v: k for k, v in PNAME_TO_IDX.items()}
PNAME_NUM_CLASSES = len(PNAMES) + 1  # 8: blank + 7 pitch letters


def load_pos_vocab(vocab_path):
    """pos_seq 的字表由 build_dataset.py 从实际保留数据里统计得出（不像 pname 固定是 7 个
    字母，staff_pos 的取值范围取决于语料里出现过多远的加线音符），写在 <data_dir>/pos_vocab.json。"""
    with open(vocab_path) as f:
        positions = json.load(f)
    pos_to_idx = {p: i + 1 for i, p in enumerate(positions)}  # 0 保留给 CTC blank
    idx_to_pos = {v: k for k, v in pos_to_idx.items()}
    return pos_to_idx, idx_to_pos


def vocab_for(label_field, data_dir):
    """返回 (token_to_idx, idx_to_token, num_classes)，按 label_field 选择字表。"""
    if label_field == 'pname_seq':
        return PNAME_TO_IDX, IDX_TO_PNAME, PNAME_NUM_CLASSES
    elif label_field == 'pos_seq':
        pos_to_idx, idx_to_pos = load_pos_vocab(Path(data_dir) / 'pos_vocab.json')
        return pos_to_idx, idx_to_pos, len(pos_to_idx) + 1
    raise ValueError(f'unknown label_field: {label_field}')


class LineDataset(Dataset):
    def __init__(self, manifest_path, data_dir, token_to_idx, label_field='pname_seq',
                 target_height=64, max_width=None):
        self.data_dir = Path(data_dir)
        self.target_height = target_height
        self.max_width = max_width
        self.token_to_idx = token_to_idx
        self.label_field = label_field
        with open(manifest_path) as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        img = Image.open(self.data_dir / row['image']).convert('L')
        w, h = img.size
        new_w = max(1, round(w * self.target_height / h))
        if self.max_width is not None:
            new_w = min(new_w, self.max_width)
        img = img.resize((new_w, self.target_height), Image.BILINEAR)

        tensor = TF.to_tensor(img)            # (1, H, W), values in [0, 1]
        tensor = (tensor - 0.5) / 0.5          # normalize to [-1, 1]

        # row[label_field] is a str for pname_seq (iterates as chars) or a
        # list of ints for pos_seq (iterates as ints) — token_to_idx handles
        # either uniformly.
        target = torch.tensor([self.token_to_idx[c] for c in row[self.label_field]], dtype=torch.long)
        return tensor, target, row['folio'], row['system_idx']


def collate_fn(batch):
    """Pad variable-width line images to the batch max width (padding value
    matches normalized background/white, not 0, so padding doesn't look like
    ink to the CNN). CTC wants a flat 1-D target tensor + per-sample lengths,
    not a padded 2-D one."""
    images, targets, folios, sys_idxs = zip(*batch)

    heights = {im.shape[1] for im in images}
    assert len(heights) == 1, 'all images in a batch must share the same height'
    height = heights.pop()

    max_w = max(im.shape[2] for im in images)
    batch_imgs = torch.full((len(images), 1, height, max_w), fill_value=1.0)
    input_widths = torch.zeros(len(images), dtype=torch.long)
    for i, im in enumerate(images):
        w = im.shape[2]
        batch_imgs[i, :, :, :w] = im
        input_widths[i] = w

    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat_targets = torch.cat(targets) if targets else torch.zeros(0, dtype=torch.long)

    return {
        'images': batch_imgs,
        'input_widths': input_widths,
        'targets': flat_targets,
        'target_lengths': target_lengths,
        'folios': folios,
        'system_idxs': sys_idxs,
    }
