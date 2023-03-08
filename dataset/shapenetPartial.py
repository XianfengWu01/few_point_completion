from re import T
import sys
sys.path.append('.')

import os
import random

import torch
import torch.utils.data as data
import numpy as np
import open3d as o3d


class ShapenetPartial(data.Dataset):
    """
    ShapeNet dataset in "PCN: Point Completion Network". It contains 28974 training
    samples while each complete samples corresponds to 8 viewpoint partial scans, 800
    validation samples and 1200 testing samples.
    """
    
    def __init__(self, dataroot, split, category, path_list = None):
        assert split in ['train', 'valid', 'test', 'test_novel'], "split error value!"

        self.cat2id = {
            # seen categories
            "airplane"  : "02691156",  # plane
            "cabinet"   : "02933112",  # dresser
            "car"       : "02958343",
            "chair"     : "03001627",
            "lamp"      : "03636649",
            "sofa"      : "04256520",
            "table"     : "04379243",
            "vessel"    : "04530566",  # boat
            
            # alis for some seen categories
            "boat"      : "04530566",  # vessel
            "couch"     : "04256520",  # sofa
            "dresser"   : "02933112",  # cabinet
            "airplane"  : "02691156",  # airplane
            "watercraft": "04530566",  # boat

            # unseen categories
            "bus"       : "02924116",
            "bed"       : "02818832",
            "bookshelf" : "02871439",
            "bench"     : "02828884",
            "guitar"    : "03467517",
            "motorbike" : "03790512",
            "skateboard": "04225987",
            "pistol"    : "03948459",
        }

        # self.id2cat = {cat_id: cat for cat, cat_id in self.cat2id.items()}

        self.dataroot = dataroot
        self.split = split
        self.category = category
   
        self.partial_paths = self._load_data()
   
    def __getitem__(self, index):

        # if self.split == 'train':
        #     partial_path = self.partial_paths[index]#.format(random.randint(0, 7))          
        # else:
        #     partial_path = self.partial_paths[index]
        partial_path = self.partial_paths[index]
        
        tmp = partial_path.split("/")[6: 9]
        tmp = tmp[0] + '/' + tmp[1] +'/'+ tmp[2]        

        partial_name = tmp[:-4]

        partial_pc = self.random_sample(self.read_point_cloud(partial_path), 2048)

        return torch.from_numpy(partial_pc), partial_name

    def __len__(self):
        return len(self.partial_paths)

    def _load_data(self):
        with open(os.path.join(self.dataroot, '{}.list').format(self.split), 'r') as f:
            lines = f.read().splitlines()

        if self.category != 'all':
            lines = list(filter(lambda x: x.startswith(self.cat2id[self.category]), lines))
        
        partial_paths = list()
    
        for line in lines:
            category, model_id = line.split('/')
            if self.split == 'train':
                for i in range(8):
                    partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '_{}.ply'.format(i)))
            else:
                partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '.ply'))
            
        
        
        return partial_paths
    
    def read_point_cloud(self, path):
        pc = o3d.io.read_point_cloud(path)
        return np.array(pc.points, np.float32)
    
    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]
