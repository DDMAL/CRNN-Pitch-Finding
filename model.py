"""
model.py — CRNN architecture for line-level pitch recognition.
The architecture only depends on num_classes, which dataset.py's vocab_for()
derives per label scheme (pname_seq or pos_seq) — see README.md.

CNN feature extractor (downsamples height to 1, width by 4x, standard
CRNN/Shi-et-al-2015 layout) -> BiLSTM x2 -> linear classifier over the CTC
alphabet (blank + label-scheme vocabulary).
"""
import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes, img_height=64, cnn_out_channels=512,
                 lstm_hidden=256, lstm_layers=2):
        super().__init__()
        assert img_height % 16 == 0, 'img_height must be a multiple of 16'
        final_h = img_height // 16

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                          # H/2, W/2

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                                          # H/4, W/4

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),                                # H/8, W/4

            nn.Conv2d(256, cnn_out_channels, 3, padding=1),
            nn.BatchNorm2d(cnn_out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(cnn_out_channels, cnn_out_channels, 3, padding=1),
            nn.BatchNorm2d(cnn_out_channels), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),                                # H/16, W/4

            nn.Conv2d(cnn_out_channels, cnn_out_channels, (final_h, 1)),  # H/16 -> 1
            nn.ReLU(inplace=True),
        )

        self.rnn = nn.LSTM(cnn_out_channels, lstm_hidden, num_layers=lstm_layers,
                            bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, images):
        """images: (B, 1, H, W) -> log_probs: (T, B, num_classes), T = W // 4."""
        feats = self.cnn(images)              # (B, C, 1, W')
        feats = feats.squeeze(2)               # (B, C, W')
        feats = feats.permute(2, 0, 1)          # (W', B, C)  == (T, B, C)

        seq, _ = self.rnn(feats)               # (T, B, 2*hidden)
        logits = self.fc(seq)                  # (T, B, num_classes)
        return logits.log_softmax(dim=2)

    @staticmethod
    def output_length(input_width):
        """Downsampled sequence length for a given (unpadded) image width.
        Only the two (2,2) maxpools touch width; the (2,1) ones don't."""
        w = input_width // 2
        w = w // 2
        return w
