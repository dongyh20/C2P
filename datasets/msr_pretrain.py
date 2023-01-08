import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import math
# import cv2
# import open3d as o3d
import random
import time
import h5py

class MSRAction3D(Dataset):
    def __init__(self, root = None):
        super(MSRAction3D, self).__init__()

        self.clip = []
        self.clip_complete = []
        for i in range(1,9):
            with h5py.File('/datasets_all/msr_data/msr_data_pre'+str(i)+'.h5','r') as f:
                self.clip.append(np.array(f['clip']))
                self.clip_complete.append(np.array(f['clip_complete']))
        self.clip = np.concatenate(self.clip, axis=0)
        self.clip_complete = np.concatenate(self.clip_complete, axis=0)

        
    def __len__(self):
        return len(self.clip)

    def __getitem__(self, idx):
        clip = self.clip[idx]
        clip_complete = self.clip_complete[idx]

        return clip.astype(np.float32), clip_complete.astype(np.float32)


if __name__ == '__main__':
    dataset = MSRAction3D(root='/processed_data')
