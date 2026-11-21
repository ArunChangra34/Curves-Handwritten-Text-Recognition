"""CRNN (CNN + BiLSTM + CTC) and the plain-CNN baseline."""
import torch, torch.nn as nn
from config import IMG_H, IMG_W, MAX_LEN

def block(i, o, pool):
    l = [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True)]
    return nn.Sequential(*l, nn.MaxPool2d(pool)) if pool else nn.Sequential(*l)

class Backbone(nn.Module):
    """(B,1,32,160) -> (B,128,2,40): height /16, width /4 -> 40 time steps."""
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(block(1, 16, (2, 2)), block(16, 32, (2, 2)), block(32, 64, None),
                               block(64, 64, (2, 1)), block(64, 128, None), block(128, 128, (2, 1)), nn.Dropout2d(0.1))
    def forward(self, x): return self.f(x)

class CRNN(nn.Module):
    def __init__(self, nc, hidden=128, rnn=True):
        super().__init__()
        self.cnn, self.use_rnn = Backbone(), rnn
        self.proj = nn.Linear(128 * 2, 256)
        self.rnn = nn.LSTM(256, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2) if rnn else None
        self.out = nn.Linear(2 * hidden if rnn else 256, nc)
    def forward(self, x):                        # returns (B,T,C) logits
        f = self.cnn(x); b, c, h, w = f.shape
        f = self.proj(f.permute(0, 3, 1, 2).reshape(b, w, c * h)).relu()
        if self.use_rnn: f, _ = self.rnn(f)
        return self.out(f)

class SlotCNN(nn.Module):
    """Plain CNN baseline: no sequence model, no alignment. One softmax per character slot."""
    def __init__(self, nc):
        super().__init__()
        self.cnn, self.nc = Backbone(), nc
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 2 * 40, 256), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(256, MAX_LEN * nc))
    def forward(self, x): return self.fc(self.cnn(x)).view(-1, MAX_LEN, self.nc)
