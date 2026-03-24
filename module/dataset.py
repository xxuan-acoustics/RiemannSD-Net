import collections
import os
import random

import numpy as np
import pandas as pd
import torch,librosa
from scipy import signal
from scipy.io import wavfile
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, Dataset
from .augment import WavAugment
import torchaudio.transforms as T
def pre_emphasis(sig, pre_emph_coeff=0.97):
    """
    perform preemphasis on the input signal.

    Args:
        sig   (array) : signal to filter.
        coeff (float) : preemphasis coefficient. 0 is no filter, default is 0.95.

    Returns:
        the filtered signal.
    """
    return np.append(sig[0], sig[1:] - pre_emph_coeff * sig[:-1])
def mean_power_normalization(transfer_function,
                             final_output,
                             lam_myu=0.999,
                             L=80,
                             k=1):
    myu = np.zeros(shape=(transfer_function.shape[0]))
    myu[0] = 0.0001
    normalized_power = np.zeros_like(transfer_function)
    for m in range(1, transfer_function.shape[0]):
        myu[m] = lam_myu * myu[m - 1] + \
            (1 - lam_myu) / L * \
            sum([transfer_function[m, s] for s in range(0, L - 1)])
    normalized_power = k * transfer_function / myu[:, None]
    # log_normalized_power = 10*np.log(normalized_power)
    # log_normalized_power = librosa.amplitude_to_db(transfer_function,ref=myu[:, None])
    return normalized_power
    # return log_normalized_power

def SNR(audio, snr):
    #在audio y中 添加噪声 噪声强度SNR为int
    # print("snr",snr)
    audio_power = audio ** 2
    audio_average_power = np.mean(audio_power)
    audio_average_db = 10 * np.log10(audio_average_power)
    noise_average_db = audio_average_db - snr
    noise_average_power = 10 ** (noise_average_db / 10)
    mean_noise = 0
    noise = np.random.normal(mean_noise, np.sqrt(noise_average_power), len(audio))
    return audio + noise
def load_audio(filename, second=3):
    # print("filename",filename)
    # print("second",second)
    sample_rate, waveform = wavfile.read(filename)
    audio_length = waveform.shape[0]

    if second <= 0:
        # waveform = waveform.astype(np.float64).copy()
        # # print("waveform", waveform.shape)
        # noise_waveform = SNR(waveform, snr=10)
        # # print("noise_waveform", noise_waveform.shape)
        # # return waveform#.copy()
        # return noise_waveform
        return waveform.astype(np.float64).copy()

    length = np.int64(sample_rate * second)

    if audio_length <= length:
        # print("0")
        shortage = length - audio_length
        waveform = np.pad(waveform, (0, shortage), 'wrap')
        waveform = waveform.astype(np.float64)
    else:
        # print("1")
        start = np.int64(random.random()*(audio_length-length))
        waveform =  waveform[start:start+length].astype(np.float64)
    waveform = np.stack([waveform], axis=0)
    return waveform#.copy()


# class Train_Dataset(Dataset):
#     def __init__(self, train_csv_path, second=3, pairs=True, aug=False, **kwargs):
#         self.second = second
#         self.pairs = pairs

#         df = pd.read_csv(train_csv_path)
#         self.labels = df["utt_spk_int_labels"].values
#         self.paths = df["utt_paths"].values
#         self.labels, self.paths = shuffle(self.labels, self.paths)
#         self.aug = aug
#         if aug:
#             self.wav_aug = WavAugment()

#         print("Train Dataset load {} speakers".format(len(set(self.labels))))
#         print("Train Dataset load {} utterance".format(len(self.labels)))

#     def __getitem__(self, index):

#         waveform_1 = load_audio(self.paths[index], self.second)
#         # if self.aug:
#         #     # print("1")
#         #     waveform_1 = self.wav_aug(waveform_1)
#         if self.pairs == False:
#             # print("2")
#             # print("waveform_1",waveform_1.shape)
#             waveform_1 = waveform_1.squeeze()
#             waveform_1 = pre_emphasis(sig=waveform_1, pre_emph_coeff=0.97)
#             S = librosa.feature.melspectrogram(y=waveform_1, sr=16000, power=1, n_fft=512, hop_length=160, n_mels=80)

#             U = mean_power_normalization(S,S ,L=100)
#             V = U ** (1 / 15)
#             # logmelspec = librosa.amplitude_to_db(S)
#             # print("torch.FloatTensor(U)",torch.FloatTensor(U).shape)
#             return torch.FloatTensor(V), self.labels[index]

#         else:
#             waveform_2 = load_audio(self.paths[index], self.second)
#             if self.aug == True:
#                 waveform_2 = self.wav_aug(waveform_2)
#             return torch.FloatTensor(waveform_1), torch.FloatTensor(waveform_2), self.labels[index]

#     def __len__(self):
#         return len(self.paths)

import torch
import torch.nn as nn
import torchaudio.functional as taf
from torchaudio.transforms import Spectrogram, AmplitudeToDB
from typing import Optional, Dict

class LogLinearFilterBank(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_filter: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        log_lf: bool = False, 
        top_db: float = 80.0,
        speckwargs: Optional[Dict] = None,
    ) -> None:
        super(LogLinearFilterBank, self).__init__()
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.n_filter = n_filter
        self.log_lf = log_lf
        
        # 初始化 AmplitudeToDB
        # 注意：如果你外面传入 power=2.0，这里 stype="power" 是对的
        self.amplitude_to_DB = AmplitudeToDB(stype="power", top_db=top_db) 
        
        speckwargs = speckwargs or {}
        # 默认值保护
        if 'n_fft' not in speckwargs: speckwargs['n_fft'] = 512
        if 'power' not in speckwargs: speckwargs['power'] = 2.0 # 建议默认设为 2.0 以匹配 DB 转换
        
        self.Spectrogram = Spectrogram(**speckwargs)
        
        n_freqs = (speckwargs['n_fft'] // 2) + 1
        
        # 获取线性滤波器矩阵
        # 形状: (n_freqs, n_filter)
        filter_mat = taf.linear_fbanks(
            n_freqs=n_freqs, 
            f_min=self.f_min, 
            f_max=self.f_max,
            n_filter=self.n_filter, 
            sample_rate=self.sample_rate,
        )
        
        # 注册为 buffer (不参与梯度更新，但随模型移动设备)
        self.register_buffer("filter_mat", filter_mat)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # 1. 提取声谱图 (Batch, Freq, Time)
        specgram = self.Spectrogram(waveform) 
        
        # 2. 应用线性滤波器
        # 矩阵乘法需要维度对齐：(Time, Freq) * (Freq, Filter) -> (Time, Filter)
        specgram = torch.matmul(specgram.transpose(-1, -2), self.filter_mat)
        
        # 转回 (Batch, Filter, Time)
        specgram = specgram.transpose(-1, -2) 
        
        # 3. 取对数 (dB)
        if self.log_lf:
            # 自然对数 log_e
            log_lfb = torch.log(specgram + 1e-6)
        else:
            # 分贝刻度 10*log_10
            log_lfb = self.amplitude_to_DB(specgram)
            
        return log_lfb

class Train_Dataset(Dataset):
    def __init__(self, train_csv_path, second=3, pairs=True, aug=False, **kwargs):
        self.second = second
        self.pairs = pairs

        df = pd.read_csv(train_csv_path)
        self.labels = df["utt_spk_int_labels"].values
        self.paths = df["utt_paths"].values
        self.labels, self.paths = shuffle(self.labels, self.paths)
        self.aug = aug
        if aug:
            self.wav_aug = WavAugment()

        self.lfcc_transform = LogLinearFilterBank(
                                sample_rate=16000, 
                                n_filter=80, 
                                f_min=0.0, 
                                f_max=8000.0, 
                                log_lf=False,  # False 表示使用 AmplitudeToDB (10*log10)，这通常是声谱图的标准做法
                                speckwargs={
                                "n_fft": 512,       # 必须和原配置一致
                                "hop_length": 160,  # 必须和原配置一致
                                "win_length": 512,  
                                "center": True,
                                "power": 2.0        # 建议设为 2.0 (能量谱)，配合 AmplitudeToDB
                            }
                            )

        print("Train Dataset load {} speakers".format(len(set(self.labels))))
        print("Train Dataset load {} utterance".format(len(self.labels)))

    def __getitem__(self, index):
        # 1. 设定目标长度 (Sample Rate = 16000)
        # 例如 3秒 = 48000 采样点
        target_len = int(self.second * 16000)

        # 2. 加载第一个波形
        waveform_1 = load_audio(self.paths[index], self.second)
        waveform_1 = np.squeeze(waveform_1) # 去除多余维度

        # [修复] 处理 waveform_1 的长度 (Pad 或 Cut)
        if waveform_1.shape[0] < target_len:
            pad_size = target_len - waveform_1.shape[0]
            # mode='wrap' 表示循环填充，避免补0产生静音段影响特征
            waveform_1 = np.pad(waveform_1, (0, pad_size), mode='wrap')
        elif waveform_1.shape[0] > target_len:
            waveform_1 = waveform_1[:target_len]

        # 分支 A: 单样本模式 (pairs=False)
        if self.pairs == False:
            # if self.aug:
            #     waveform_1 = self.wav_aug(waveform_1)

            waveform_for_ecapa = torch.FloatTensor(waveform_1)

            waveform_1 = pre_emphasis(sig=waveform_1, pre_emph_coeff=0.97)
            
            # 生成 Mel 谱
            # S = librosa.feature.melspectrogram(y=waveform_1, sr=16000, power=1, n_fft=512, hop_length=160, n_mels=80)
            # logmelspec = librosa.amplitude_to_db(S)

            # # # 归一化 (因为 waveform_1 长度已固定，S 的帧数足够，不会再报 index out of bounds)
            # # # U = mean_power_normalization(S, S, L=100)
            # # # V = U ** (1 / 15)
            
            # return torch.FloatTensor(logmelspec), waveform_for_ecapa, self.labels[index]


            # 生成 LFCC 谱
            waveform_tensor = torch.from_numpy(waveform_1).float()
            lfcc_feat = self.lfcc_transform(waveform_tensor)
            return lfcc_feat, waveform_for_ecapa, self.labels[index]

            # 生成 CQCC 谱 
            # cqt_complex = librosa.cqt(y=waveform_1, sr=16000, hop_length=160, n_bins=80)
            # cqt_mag = np.abs(cqt_complex)
            # cqt_spec = librosa.amplitude_to_db(cqt_mag, ref=np.max)
            # return torch.FloatTensor(cqt_spec), waveform_for_ecapa, self.labels[index]

            

            

        # 分支 B: 成对模式 (pairs=True)
        else:
            waveform_2 = load_audio(self.paths[index], self.second)
            waveform_2 = np.squeeze(waveform_2)

            # [修复] 同样处理 waveform_2 的长度
            if waveform_2.shape[0] < target_len:
                pad_size = target_len - waveform_2.shape[0]
                waveform_2 = np.pad(waveform_2, (0, pad_size), mode='wrap')
            elif waveform_2.shape[0] > target_len:
                waveform_2 = waveform_2[:target_len]

            if self.aug == True:
                waveform_2 = self.wav_aug(waveform_2)
                
            return torch.FloatTensor(waveform_1), torch.FloatTensor(waveform_2), self.labels[index]

    def __len__(self):
        return len(self.paths)


class Semi_Dataset(Dataset):
    def __init__(self, label_csv_path, unlabel_csv_path, second=2, pairs=True, aug=False, **kwargs):
        self.second = second
        self.pairs = pairs

        df = pd.read_csv(label_csv_path)
        self.labels = df["utt_spk_int_labels"].values
        self.paths = df["utt_paths"].values

        self.aug = aug
        if aug:
            self.wav_aug = WavAugment()

        df = pd.read_csv(unlabel_csv_path)
        self.u_paths = df["utt_paths"].values
        self.u_paths_length = len(self.u_paths)

        if label_csv_path != unlabel_csv_path:
            self.labels, self.paths = shuffle(self.labels, self.paths)
            self.u_paths = shuffle(self.u_paths)

        # self.labels = self.labels[:self.u_paths_length]
        # self.paths = self.paths[:self.u_paths_length]
        print("Semi Dataset load {} speakers".format(len(set(self.labels))))
        print("Semi Dataset load {} utterance".format(len(self.labels)))

    def __getitem__(self, index):
        waveform_l = load_audio(self.paths[index], self.second)

        idx = np.random.randint(0, self.u_paths_length)
        waveform_u_1 = load_audio(self.u_paths[idx], self.second)
        if self.aug == True:
            waveform_u_1 = self.wav_aug(waveform_u_1)

        if self.pairs == False:
            return torch.FloatTensor(waveform_l), self.labels[index], torch.FloatTensor(waveform_u_1)

        else:
            waveform_u_2 = load_audio(self.u_paths[idx], self.second)
            if self.aug == True:
                waveform_u_2 = self.wav_aug(waveform_u_2)
            return torch.FloatTensor(waveform_l), self.labels[index], torch.FloatTensor(waveform_u_1), torch.FloatTensor(waveform_u_2)

    def __len__(self):
        return len(self.paths)


class Evaluation_Dataset(Dataset):
    def __init__(self, paths, second=-1, **kwargs):
        self.paths = paths
        self.second = second
        self.lfcc_transform = LogLinearFilterBank(
                                sample_rate=16000, 
                                n_filter=80, 
                                f_min=0.0, 
                                f_max=8000.0, 
                                log_lf=False,  # False 表示使用 AmplitudeToDB (10*log10)，这通常是声谱图的标准做法
                                speckwargs={
                                "n_fft": 512,       # 必须和原配置一致
                                "hop_length": 160,  # 必须和原配置一致
                                "win_length": 512,  
                                "center": True,
                                "power": 2.0        # 建议设为 2.0 (能量谱)，配合 AmplitudeToDB
                            }
                            )
        print("load {} utterance".format(len(self.paths)))

    def __getitem__(self, index):
        # if index>=0 and index<=4707:

        waveform = load_audio(self.paths[index], self.second)
        waveform = waveform.squeeze()


        pre_waveform = pre_emphasis(sig=waveform, pre_emph_coeff=0.97)


        # 生成 Mel 谱
        # S = librosa.feature.melspectrogram(y=pre_waveform, sr=16000, power=1, n_fft=512, hop_length=160, n_mels=80)

        # logmelspec = librosa.amplitude_to_db(S)
        # # U = mean_power_normalization(S, S, L=100)
        # # V = U ** (1 / 15)
        # # print("torch.FloatTensor(V)",torch.FloatTensor(V).shape)
        # return torch.FloatTensor(logmelspec), self.paths[index]
    
        # 生成 LFCC 谱
        waveform_tensor = torch.from_numpy(pre_waveform).float()
        lfcc_feat = self.lfcc_transform(waveform_tensor)

        return lfcc_feat, self.paths[index]

        # 生成 CQCC 谱 
        # cqt_complex = librosa.cqt(y=pre_waveform, sr=16000, hop_length=160, n_bins=80)
        # cqt_mag = np.abs(cqt_complex)
        # cqt_spec = librosa.amplitude_to_db(cqt_mag, ref=np.max)
        # return torch.FloatTensor(cqt_spec), self.paths[index]

    def __len__(self):
        return len(self.paths)

if __name__ == "__main__":
    dataset = Train_Dataset(train_csv_path="data/train.csv", second=3)
    loader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=False
    )
    for x, label in loader:
        pass

