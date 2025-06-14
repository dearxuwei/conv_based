# coding: utf-8
"""
FFTCA
=====
在 ConvCA 框架中，用可学习的 **频域滤波** 块替换前三个时域卷积层 (conv11/12/13)。
核心思路：对时间轴做 rFFT → 复数可学习滤波 → iFFT，还原到时域后继续
沿用 ConvCA 的模板分支与参考分支，最终输出 40-类 logits。

作者：ChatGPT  (Jun-2025)
"""

import torch
import torch.nn as nn


# ----------------------------------------------------------------------
# 1. 频域前端：F r e q C o n v B l o c k
# ----------------------------------------------------------------------
class FreqConvBlock(nn.Module):
    """
    在频域上进行复数卷积（乘性滤波），可学习 magnitude 与 phase。
    输入 : (B, 1, T, C)    —— 与原 ConvCA 保持一致
    输出 : (B, 1, T, 1)    —— 卷积后通道维度被压到 1，方便后续相关运算
    """

    def __init__(self,
                 C: int,        # 空间通道数（原 ConvCA 宽度）
                 T: int,        # 时间点数
                 internal_channels: int = 16,  # 复数滤波器个数
                 dropout: float = 0.75):       # 与 ConvCA 参数对应
        super().__init__()
        self.T, self.C = T, C

        F = T // 2 + 1          # rFFT 后正频率长度 (含 DC & Nyquist)

        # 复数滤波器的实部/虚部权重：shape = (滤波器个数, C, F)
        self.weight_real = nn.Parameter(torch.randn(internal_channels, C, F) * 0.01)
        self.weight_imag = nn.Parameter(torch.randn(internal_channels, C, F) * 0.01)

        # internal_channels → 1：用 1×1 卷积整合多路频域滤波结果
        self.combine = nn.Conv2d(internal_channels, 1, kernel_size=1)

        self.dropout = nn.Dropout(dropout)

    # -------- 复数乘法辅助函数 -----------------------------------------
    @staticmethod
    def _complex_mul(X: torch.Tensor, Wr: torch.Tensor, Wi: torch.Tensor):
        """
        (a+j b)(c+j d) = (ac-bd) + j(ad+bc)
        避免 torch.complex*torch.complex 过程中建立临时大张量。
        """
        Xr, Xi = X.real, X.imag
        Yr = Xr * Wr - Xi * Wi
        Yi = Xr * Wi + Xi * Wr
        return torch.complex(Yr, Yi)

    # -------- 前向传播 -------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        x 形状：(B, 1, T, C)
        1. squeeze/permute → (B, C, T)
        2. rFFT → 复数频谱 (B, C, F)
        3. 与复数权重相乘 (内部 16 路)
        4. iFFT → (B, internal, C, T)
        5. combine + 空间均值 → (B, 1, T, 1)
        """
        B, _, T, C = x.shape
        assert (T, C) == (self.T, self.C), f"输入尺寸与模型初始化不一致: {x.shape}"

        # --- 1. 时域 -> 频域 -------------------------------------------
        x_tc = x.squeeze(1).permute(0, 2, 1)      # (B,T,C)→(B,C,T)
        Xf = torch.fft.rfft(x_tc, n=T, dim=-1)    # (B,C,F)

        # --- 2. 复数滤波 ----------------------------------------------
        Wr = self.weight_real.unsqueeze(0)        # (1,internal,C,F)
        Wi = self.weight_imag.unsqueeze(0)        # (1,internal,C,F)
        Xf = Xf.unsqueeze(1)                      # (B,1,C,F)
        Yf = self._complex_mul(Xf, Wr, Wi)        # (B,internal,C,F)

        # --- 3. 回到时域 ----------------------------------------------
        yt = torch.fft.irfft(Yf, n=T, dim=-1)     # (B,internal,C,T)

        # --- 4. 整合空间/滤波器通道 ------------------------------------
        yt = yt.permute(0, 1, 3, 2)              # (B,internal,T,C)
        yt = self.combine(yt)                    # (B,1,T,C)
        yt = yt.mean(dim=3, keepdim=True)        # (B,1,T,1)
        yt = self.dropout(yt)
        return yt


# ----------------------------------------------------------------------
# 2. 整体网络：F F T C A
# ----------------------------------------------------------------------
class FFTCA(nn.Module):
    """
    ConvCA 的升级版：
    - 前端改为 FreqConvBlock
    - 模板分支 (conv21/22) 与参考分支 (conv31/32) 保持原样
    """

    def __init__(self,
                 params: dict,             # 兼容原工程的参数字典，可留空 {}
                 device: torch.device,
                 T: int = 150,             # 时间点数
                 C: int = 9,               # 通道数
                 internal_channels: int = 16,
                 use_temp: bool = True,    # 是否启用模板相关
                 use_ref: bool = True):    # 是否启用参考相关
        super().__init__()

        self.device = device
        self.use_temp = use_temp
        self.use_ref = use_ref

        # ---- ① 频域前端，取代 conv11/12/13 ----------------------------
        self.freq_front = FreqConvBlock(C, T, internal_channels, dropout=0.75)
        self.flatten = nn.Flatten(start_dim=2)   # (B,1,T,1) → (B,1,T)

        # ---- ② 模板分支 conv21 / conv22 ------------------------------
        self.conv21 = nn.Conv2d(C, 40, (9, 1), padding='same')
        self.conv22 = nn.Conv2d(40, 1, (9, 1), padding='same')
        self.drop2 = nn.Dropout(0.15)

        # ---- ③ 参考分支 conv31 / conv32 ------------------------------
        self.conv31 = nn.Conv2d(C + 1, 40, (9, 1), padding='same')
        self.conv32 = nn.Conv2d(40, 1, (9, 1), padding='same')
        self.drop3 = nn.Dropout(0.15)

        # ---- ④ 融合系数 α，及最终全连接 ------------------------------
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习权衡 temp/ref
        self.fc = nn.Linear(40, 40)

    # ------------------------------------------------------------------
    # 计算批量化归一化互相关，保持与 ConvCA 相同的数学操作
    # ------------------------------------------------------------------
    @staticmethod
    def _corr(sig: torch.Tensor, temp: torch.Tensor):
        """
        sig : (B,1,T)
        temp: (B,T,40)
        返回 : (B,40) 归一化互相关
        """
        xt = torch.bmm(sig, temp)            # (B,1,40)
        xx = torch.bmm(sig, sig.transpose(1, 2))  # (B,1,1)
        tt = torch.sum(temp * temp, dim=1)   # (B,40)
        return xt.squeeze(1) / torch.sqrt(tt) / torch.sqrt(xx.squeeze(1))

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------
    def forward(self,
                signal: torch.Tensor,   # (B,1,T,C)
                temp: torch.Tensor,     # (B,C,T,40)
                ref: torch.Tensor):     # (B,C+1,T,40)

        # -------- A. 频域前端处理 -------------------------------------
        sig = self.freq_front(signal)        # (B,1,T,1)
        sig = self.flatten(sig)              # (B,1,T)

        feats = []

        # -------- B. 模板互相关分支 -----------------------------------
        if self.use_temp:
            t = self.drop2(self.conv22(self.conv21(temp))).squeeze()  # (B,T,40)
            corr_t = self._corr(sig, t)
            feats.append(corr_t)

        # -------- C. 参考互相关分支 -----------------------------------
        if self.use_ref:
            r = self.drop3(self.conv32(self.conv31(ref))).squeeze()   # (B,T,40)
            corr_r = self._corr(sig, r)
            feats.append(corr_r)

        # -------- D. 融合两个分支 -------------------------------------
        if self.use_temp and self.use_ref:
            corr = self.alpha * feats[0] + (1.0 - self.alpha) * feats[1]
        elif self.use_temp:
            corr = feats[0]
        elif self.use_ref:
            corr = feats[0]
        else:
            raise ValueError("至少开启模板或参考分支之一！")

        # -------- E. 分类输出 -----------------------------------------
        logits = self.fc(corr)               # (B,40)
        return logits



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = FFTCA(params={}, device=device, T=250, C=9).to(device)

# 假设 data : (32,1,250,9)
#      temp : (32,9,250,40)
#      ref  : (32,10,250,40)
data = torch.randn(32, 1, 250, 9).to(device)
temp = torch.randn(32, 9, 250, 40).to(device)
ref  = torch.randn(32, 10, 250, 40).to(device)

logits = model(data, temp, ref)
print(logits.shape)  # >>> torch.Size([32, 40])
