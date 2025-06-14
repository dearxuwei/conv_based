import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.utils.data as data_utils

class ConvCA(nn.Module):
    def __init__(self, params, device, use_temp=True, use_ref=True):
        super(ConvCA, self).__init__()
        self.params = params
        self.device = device
        self.use_temp = use_temp
        self.use_ref = use_ref

        self.conv11 = nn.Conv2d(1, 16, 9, padding='same')
        self.conv12 = nn.Conv2d(16, 1, (1,9), padding='same')
        self.conv13 = nn.Conv2d(1, 1, (1,9), padding='valid')
        self.drop1 = nn.Dropout(p=0.75)
        self.flatten = nn.Flatten(start_dim=2)

        self.conv21 = nn.Conv2d(9, 40, (9,1), padding='same')
        self.conv22 = nn.Conv2d(40, 1, (9,1), padding='same')
        self.drop2 = nn.Dropout(p=0.15)

        self.conv31 = nn.Conv2d(10, 40, (9,1), padding='same')
        self.conv32 = nn.Conv2d(40, 1, (9,1), padding='same')
        self.drop3 = nn.Dropout(p=0.15)

        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始0.5

        input_dim = 40
        # if use_temp:
        #     input_dim += 40
        # if use_ref:
        #     input_dim += 40
        self.fc = nn.Linear(input_dim, 40)


    def Corr(self, signal, temp):
        corr_xt = torch.bmm(signal,temp)
        corr_xt = torch.squeeze(corr_xt, 1)
        corr_xx = torch.bmm(signal, torch.transpose(signal,1,2))
        corr_xx = torch.squeeze(corr_xx, 1)
        corr_tt = torch.sum(temp*temp, dim=1)
        corr = corr_xt/torch.sqrt(corr_tt)/torch.sqrt(corr_xx)
        return corr

    def forward(self, signal, temp, ref):
        # signal:32,1,250.9
        # temp:32,9,250,40

        if torch.is_tensor(signal) != True:
            signal = torch.from_numpy(signal).float().to(self.device)
        if torch.is_tensor(temp) != True:
            temp = torch.from_numpy(temp).float().to(self.device)
        if torch.is_tensor(ref) != True:
            ref = torch.from_numpy(ref).float().to(self.device)

        # signal:32,1,250,9
        # temp:32,10,250,40
        # ref:32,10,250,40
        signal = self.conv11(signal)
        signal = self.conv12(signal)
        signal = self.conv13(signal)
        signal = self.drop1(signal)
        signal = self.flatten(signal)

        features = []
        if self.use_temp:
            temp = self.conv21(temp)
            temp = self.conv22(temp)
            temp = self.drop2(temp)
            temp = torch.squeeze(temp)
            corr1 = self.Corr(signal, temp)
            features.append(corr1)

        if self.use_ref:
            ref = self.conv31(ref)
            ref = self.conv32(ref)
            ref = self.drop3(ref)
            ref = torch.squeeze(ref)
            corr2 = self.Corr(signal, ref)
            features.append(corr2)

        if self.use_temp and self.use_ref:
            corr = self.alpha * corr1 + (1 - self.alpha) * corr2
        elif self.use_temp:
            corr = corr1
        elif self.use_ref:
            corr = corr2
        else:
            raise ValueError("至少启用一个分支")
        
        # corr = torch.cat(features, dim=1)
        out = self.fc(corr)

        # signal:32,1,250
        # temp:32,250,40
        return out
