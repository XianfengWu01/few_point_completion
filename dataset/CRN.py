import sys
sys.path.append('.')

import os
import random

import torch
import torch.utils.data as data
import open3d as o3d
import os.path
import torch
import numpy as np
import h5py


def random_sample(self, pc, n):
    idx = np.random.permutation(pc.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
    return pc[idx[:n]]


class CRNShapeNet(data.Dataset):
    """
    Dataset with GT and partial shapes provided by CRN
    Used for shape completion and pre-training tree-GAN
    """

    def __init__(self, dataset_path, split):
        self.dataset_path = dataset_path
        self.split = split

        pathname = os.path.join(self.dataset_path, f'{self.split}_data.h5')

        data = h5py.File(pathname, 'r')
        self.gt = data['complete_pcds'][()]
        self.partial = data['incomplete_pcds'][()]
        self.labels = data['labels'][()]

        self.index_list = np.array([i for (i, j) in enumerate(self.labels)])

    def __getitem__(self, index):
        full_idx = self.index_list[index]
        gt = torch.from_numpy(self.gt[full_idx]) # fast alr
        label = self.labels[index]
        partial = torch.from_numpy(self.partial[full_idx])
        return gt, partial #, full_idx

    def __len__(self):
        return len(self.index_list)
