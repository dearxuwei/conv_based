# -*- coding: utf-8 -*-
"""
ConvFFTCA-SE
============
并行融合 ConvCA 的时域卷积前端和 FFTCA 的频域滤波前端，
通过 Squeeze-and-Excitation (SE) 通道注意力自适应加权。
包含示例训练 / 可视化函数，用于保存 loss-acc 曲线与混淆矩阵。
"""

import json, pickle, os, matplotlib.pyplot as plt
from typing import Dict, List
import torch, torch.nn as nn, torch.nn.functional as F

# ---------------- 1. 时域分支 ----------------
class ConvBranch(nn.Module):
    def __init__(self, C: int, T: int, dropout: float = 0.75):
        super().__init__()
        self.conv11 = nn.Conv2d(1, 8, (9,1), padding=(4,0))
        self.conv12 = nn.Conv2d(8, 8, (9,1), padding=(4,0))
        self.conv13 = nn.Conv2d(8, 1, 1)
        self.bn  = nn.BatchNorm2d(8)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                       # (B,1,T,C)
        x = self.act(self.bn(self.conv11(x)))
        x = self.act(self.bn(self.conv12(x)))
        x = self.conv13(x)                      # (B,1,T,C)
        x = x.mean(dim=3, keepdim=True)         # (B,1,T,1)
        return self.drop(x)

# ---------------- 2. 频域分支 ----------------
class FreqBranch(nn.Module):
    def __init__(self, C: int, T: int, D: int = 16, dropout: float = 0.75):
        super().__init__()
        self.T, self.C = T, C
        F = T//2 + 1
        self.wr = nn.Parameter(torch.randn(D, C, F)*0.01)
        self.wi = nn.Parameter(torch.randn(D, C, F)*0.01)
        self.combine = nn.Conv2d(D, 1, 1)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def _cmul(X, Wr, Wi):
        Xr, Xi = X.real, X.imag
        return torch.complex(Xr*Wr - Xi*Wi, Xr*Wi + Xi*Wr)

    def forward(self, x):                       # (B,1,T,C)
        B,_,T,C = x.shape
        X = x.squeeze(1).permute(0,2,1)         # (B,C,T)
        Xf = torch.fft.rfft(X, n=T, dim=-1)     # (B,C,F)
        Yf = self._cmul(Xf.unsqueeze(1), self.wr.unsqueeze(0), self.wi.unsqueeze(0))
        y  = torch.fft.irfft(Yf, n=T, dim=-1)   # (B,D,C,T)
        y  = y.permute(0,1,3,2)                 # (B,D,T,C)
        y  = self.combine(y).mean(dim=3, keepdim=True)  # (B,1,T,1)
        return self.drop(y)

# ---------------- 3. SE 融合 ----------------
class SEFusion(nn.Module):
    def __init__(self, reduction:int=2):
        super().__init__()
        self.fc1 = nn.Linear(2, 2//reduction, bias=False)
        self.fc2 = nn.Linear(2//reduction, 2, bias=False)

    def forward(self, xt, xf):                  # (B,1,T,1) x2
        x = torch.cat([xt, xf], dim=1)          # (B,2,T,1)
        s = x.mean(dim=[2,3])                   # (B,2)
        w = torch.sigmoid(self.fc2(F.gelu(self.fc1(s)))).view(-1,2,1,1)
        return (x * w).sum(dim=1, keepdim=True) # (B,1,T,1)

# ---------------- 4. 完整模型 ----------------
class ConvFFTCA_SE(nn.Module):
    def __init__(self, T=150, C=9, D=16):
        super().__init__()
        self.time = ConvBranch(C, T)
        self.freq = FreqBranch(C, T, D)
        self.fuse = SEFusion()
        self.flat = nn.Flatten(start_dim=2)

        self.conv21 = nn.Conv2d(C, 40, (9,1), padding='same')
        self.conv22 = nn.Conv2d(40, 1, (9,1), padding='same')
        self.drop2  = nn.Dropout(0.15)

        self.conv31 = nn.Conv2d(C+1, 40, (9,1), padding='same')
        self.conv32 = nn.Conv2d(40, 1, (9,1), padding='same')
        self.drop3  = nn.Dropout(0.15)

        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.fc    = nn.Linear(40, 40)

    @staticmethod
    def _corr(sig, tpl):
        xt = torch.bmm(sig, tpl)                # (B,1,40)
        xx = torch.bmm(sig, sig.transpose(1,2))
        tt = torch.sum(tpl*tpl, dim=1)
        return xt.squeeze(1) / torch.sqrt(xx.squeeze(1)) / torch.sqrt(tt)

    def forward(self, signal, temp, ref):
        x = self.fuse(self.time(signal), self.freq(signal))  # (B,1,T,1)
        x = self.flat(x)                                     # (B,1,T)

        tfeat = self.drop2(self.conv22(self.conv21(temp))).squeeze() # (B,T,40)
        rfeat = self.drop3(self.conv32(self.conv31(ref))).squeeze()  # (B,T,40)

        corr = self.alpha*self._corr(x,tfeat) + (1-self.alpha)*self._corr(x,rfeat)

        return self.fc(corr)                                 # (B,40)

# ---------------- 5. 可视化示例函数 ----------------
def plot_history(log_dict, png="curve.png"):
    import matplotlib.pyplot as plt
    ep = range(1, len(log_dict['loss'])+1)
    plt.figure(); plt.subplot(1,2,1); plt.plot(ep, log_dict['loss']); plt.title('Loss')
    plt.subplot(1,2,2); plt.plot(ep, log_dict['acc']); plt.title('Accuracy')
    plt.savefig(png, dpi=300); print("Curve saved:", png)

def plot_confmat(cm, classes, png="cm.png"):
    import numpy as np, matplotlib.pyplot as plt
    plt.figure(figsize=(5,5)); plt.imshow(cm, cmap='Blues')
    plt.xticks(np.arange(len(classes)), classes, rotation=45)
    plt.yticks(np.arange(len(classes)), classes)
    plt.colorbar(); plt.tight_layout(); plt.savefig(png, dpi=300)
    print("Confusion matrix saved:", png)

# ---------------- 6. 快速自检 ----------------
if __name__ == "__main__":
    B,T,C,K = 4,150,9,40
    net = ConvFFTCA_SE(T,C)
    sig = torch.randn(B,1,T,C)
    tpl = torch.randn(B,C,T,K)
    ref = torch.randn(B,C+1,T,K)
    out = net(sig,tpl,ref)
    print("Test forward OK, logits:", out.shape)
