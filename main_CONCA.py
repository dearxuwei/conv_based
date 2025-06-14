import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import os
import pandas as pd
from tqdm import tqdm
import datetime
from scipy import signal
from CONVCA_tempsia import ConvCA
from scipy.io import loadmat
from convca_read import prepare_data_as, prepare_template, prepare_ref, normalize
import torch.utils.data as data_utils
import random
# random.seed(42)  # 固定种子，保证实验可重复
##############################################
# 配置参数
##############################################
class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_subjects = 35           # 被试数量
    num_trials = 6              # 每个被试的试次
    num_freqs = 40              # 类别数（目标频率数）
    num_channels = 9            # EEG 信号通道数
    # 实验条件配置（按需注释/取消注释）
    branches = ['sample', 'template', 'sincos']        # 条件1：三个分支
    # branches = ['sample', 'template']               # 条件2：样本+模板 
    # branches = ['sample', 'sincos']                 # 条件3：样本+sincos参考
    sample_length = 150         # 每个样本的时域长度（数据点数）
    batch_size = 32
    epochs = 200
    lr = 8e-4
    weight_decay = 1e-5
    data_file = r'dataset/bench.mat'
    result_dir = r'results/'
    train_runs = []
    test_runs = []

    num_of_TB = 5

    lambda_domain = 0.1

def clip_gradient(optimizer, grad_clip):
    """梯度裁剪，防止梯度爆炸"""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

class SSVEPDataset(Dataset):
    def __init__(self, data, labels, temp, ref, is_train=True):
        """
        Args:
            data: (N_samples, sample_length, num_channels)
            labels: (N_samples,)
            subj_train: (N_samples,)
        """
        self.cfg = Config()
        self.x = data
        self.labels = labels.astype(int)
        self.temp = temp
        self.ref = ref
        self.x = torch.tensor(np.array(self.x), dtype=torch.float32)    # (N, sample_length, num_channels)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.temp = torch.tensor(np.array(self.temp), dtype=torch.float32)    # (N, sample_length, num_channels)
        self.ref = torch.tensor(np.array(self.ref), dtype=torch.float32)    # (N, sample_length, num_channels)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx], self.temp[idx], self.ref[idx]

##############################################
# 训练与评估
##############################################
def train_model(model, train_loader, test_loader, num_epochs=10, lr=1e-3, lambda_domain=0.1):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=Config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, 1e-6)
    device = Config.device
    model.to(device)
    epoch_stats = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_cls_correct = 0
        total_samples = 0

        # 训练循环
        for data, temp, ref, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            data, labels, temp, ref = data.to(device), labels.to(device), temp.to(device), ref.to(device)

            optimizer.zero_grad()
            # signal:32,1,250.9
            # temp:32,9,250,40

            class_logits = model(data.unsqueeze(1), temp, ref)  # 返回 (B, num_freqs), (B, n_domains)
            loss = F.cross_entropy(class_logits, labels)
            loss.backward()
            clip_gradient(optimizer, 3.0)
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            total_cls_correct += (class_logits.argmax(dim=1) == labels).sum().item()
            total_samples += data.size(0)

        scheduler.step()
        avg_loss = total_loss / total_samples
        train_acc = 100.0 * total_cls_correct / total_samples
        # 测试集评估
        test_acc = evaluate_model(model, test_loader)
        epoch_stats.append({
            'epoch': epoch+1,
            'train_loss': avg_loss,
            'train_acc': train_acc,
            'test_acc': test_acc
        })
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")

    return model, epoch_stats

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    device = Config.device
    with torch.no_grad():
        for data, temp, ref, labels in loader:
            data, labels, temp, ref = data.to(device), labels.to(device), temp.to(device), ref.to(device)
            class_logits = model(data.unsqueeze(1), temp ,ref)
            preds = class_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += data.size(0)
    return 100.0 * correct / total

##############################################
# 主程序：留一被试
##############################################
if __name__ == "__main__":
    os.makedirs(Config.result_dir, exist_ok=True)
    overall_results = []

    combinations = [
    ('sample+template+ref', True, True),
    ('sample+template', True, False),
    ('sample+ref', False, True)
    ]
    
    all_runs = list(range(Config.num_trials))  # [0,1,2,3,4,5]
    train_runs = random.sample(all_runs, Config.num_of_TB)
    test_runs = [r for r in all_runs if r not in train_runs]
    Config.train_runs = train_runs
    Config.test_runs = test_runs
    print(f"Train runs: {train_runs} | Test runs: {test_runs}")
    for sub_id in range(1,36):
        print(f"\n========== 训练/测试被试 {sub_id} ========== ")

        train_data, train_labels, freq = prepare_data_as(sub_id, Config.train_runs, Config.sample_length)  ## [?,tw,ch]
        test_data, test_labels, __ = prepare_data_as(sub_id, Config.test_runs, Config.sample_length)  # [?,tw,ch]
        train_data = train_data.reshape((train_data.shape[0], Config.sample_length, Config.num_channels))
        test_data = test_data.reshape((test_data.shape[0], Config.sample_length, Config.num_channels))
        train_data = torch.tensor(train_data)
        train_labels = torch.tensor(train_labels).type(torch.LongTensor)
        test_data = torch.tensor(test_data)
        test_labels = torch.tensor(test_labels).type(torch.LongTensor)

        # x_train : 5200,1,250,9
        # template: 1040,30,250,9
        template = prepare_template(sub_id, Config.train_runs, Config.sample_length)  # [cl*sample,cl,tw,ch]
        template = np.transpose(template, axes=(0, 3, 2, 1))  # [cl*sample,ch,tw,cl]
        template = torch.tensor(template)

        ref = prepare_ref(sub_id, Config.train_runs, Config.sample_length)  # [cl*sample,cl,tw,ch]
        ref = np.transpose(ref, axes=(0, 3, 2, 1))  # [cl*sample,ch,tw,cl]
        ref = torch.tensor(ref)

        train_dataset = data_utils.TensorDataset(train_data, template.repeat(len(Config.train_runs), 1, 1, 1), ref.repeat(len(Config.train_runs), 1, 1, 1), train_labels)
        test_dataset = data_utils.TensorDataset(test_data, template.repeat(len(Config.test_runs), 1, 1, 1), ref.repeat(len(Config.test_runs), 1, 1, 1), test_labels)


        for mode_name, use_temp, use_ref in combinations:
            print(f"\n========== 模式：{mode_name} ==========")
            train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False)
            model = ConvCA(Config, Config.device, use_temp=use_temp, use_ref=use_ref)

            trained_model, epoch_stats = train_model(
                model, train_loader, test_loader,
                num_epochs=Config.epochs,
                lr=Config.lr,
                lambda_domain=Config.lambda_domain
            )

            final_acc = evaluate_model(trained_model, test_loader)
            overall_results.append({'subject': sub_id, 'mode': mode_name, 'Test Acc (%)': final_acc})
            dt_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            pd.DataFrame(epoch_stats).to_csv(
                os.path.join(Config.result_dir, f"subj_{sub_id}_{mode_name}_{Config.num_of_TB}_{dt_str}.csv"),
                index=False
            )
            pd.DataFrame(overall_results).to_csv(os.path.join(Config.result_dir, "cross_subject_results.csv"), index=False)
            del model, train_loader, test_loader
            torch.cuda.empty_cache()

    print("所有被试训练完成！")
    
