import os
from typing import Any, Callable, Optional

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from pl_bolts.datasets import UnlabeledImagenet
from pl_bolts.utils.warnings import warn_missing_pkg

from .dataset import Evaluation_Dataset, Train_Dataset, Semi_Dataset


class SPK_datamodule(LightningDataModule):
    def __init__(
        self,
        train_csv_path,
        trial_path,

        unlabel_csv_path = None,
        second: int = 2,
        num_workers: int = 16,
        batch_size: int = 32,
        shuffle: bool = True,
        pin_memory: bool = True,
        drop_last: bool = True,
        pairs: bool = True,
        aug: bool = False,
        semi: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.train_csv_path = train_csv_path
        self.unlabel_csv_path = unlabel_csv_path
        self.second = second
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.trial_path = trial_path
        self.pairs = pairs
        self.aug = aug
        print("second is {:.2f}".format(second))

    def train_dataloader(self) -> DataLoader:
        if self.unlabel_csv_path is None:
            train_dataset = Train_Dataset(self.train_csv_path, self.second, self.pairs, self.aug)
        else:
            train_dataset = Semi_Dataset(self.train_csv_path, self.unlabel_csv_path, self.second, self.pairs, self.aug)
        loader = torch.utils.data.DataLoader(
                train_dataset,
                shuffle=True,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                pin_memory=True,
                drop_last=True,
                )
        return loader

    # def val_dataloader(self) -> DataLoader:
    #     trials = np.loadtxt(self.trial_path, str)
    #     self.trials = trials
    #     # if trials.T[0]==0 or trials.T[0]==1:
    #     # print("trials.T[0]",trials.T[0])
    #     # print("trials.T[1]", trials.T[1])
    #     # eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
    #     # print("number of enroll: {}".format(len(set(trials.T[1]))))
    #     # print("number of test: {}".format(len(set(trials.T[2]))))
    #     # print("number of evaluation: {}".format(len(eval_path)))
    #     eval_path = np.unique(np.concatenate((trials.T[1], trials.T[2])))
    #     print("number of enroll: {}".format(len(set(trials.T[1]))))
    #     print("number of test: {}".format(len(set(trials.T[2]))))
    #     print("number of evaluation: {}".format(len(eval_path)))
    #     eval_dataset = Evaluation_Dataset(eval_path, second=-1)
    #     loader = torch.utils.data.DataLoader(eval_dataset,
    #                                          num_workers=10,
    #                                          shuffle=False, 
    #                                          batch_size=1)
    #     return loader


    def val_dataloader(self) -> DataLoader:
        # 定义4个文件的路径 (请确保路径正确)
        path1 = "/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/pairs_protocal/large_version/Source-Speaker_evaluation_protocal/Final/seen_seen_same_speaker.txt"
        path2 = "/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/pairs_protocal/large_version/Source-Speaker_evaluation_protocal/Final/seen_seen_diff_speaker.txt"
        path3 = "/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/pairs_protocal/large_version/Source-Speaker_evaluation_protocal/Final/unseen_unseen_same_speaker.txt"
        path4 = "/xxuan-acoustics/dataset/MLAAD_v8/ptotocal/pairs_protocal/large_version/Source-Speaker_evaluation_protocal/Final/unseen_unseen_diff_speaker.txt"

        # 1. 读取所有文件
        t1 = np.loadtxt(path1, str)
        t2 = np.loadtxt(path2, str)
        t3 = np.loadtxt(path3, str)
        t4 = np.loadtxt(path4, str)

        # 2. 【核心技巧】把它们合并成一个大的列表
        # 这样做的目的是为了提取出所有需要用到的音频路径 (Union)
        combined_trials = np.vstack([t1, t2, t3, t4])

        # 3. 提取所有唯一的音频路径
        # 这样无论哪个协议需要的音频，都会被包含在 eval_path 里
        eval_path = np.unique(np.concatenate((combined_trials.T[1], combined_trials.T[2])))
        
        print(f"Total validation audio files: {len(eval_path)}")

        # 4. 创建唯一的 Dataset 和 Loader
        eval_dataset = Evaluation_Dataset(eval_path, second=10)
        loader = torch.utils.data.DataLoader(
            eval_dataset,
            num_workers=self.num_workers, 
            shuffle=False, 
            batch_size=1
        )
        return loader

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()


