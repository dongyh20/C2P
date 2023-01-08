import os
import sys
import numpy as np
import random
import open3d as o3d
from pyquaternion import Quaternion
from torch.utils.data import Dataset
import h5py

class SegDataset(Dataset):
    def __init__(self, root='/share/datasets/Seg_pre_last_h5', root_complete='/share/datasets/Seg_data_base_4096_h5', meta='/HOI4D_splits/release.txt',frames_per_clip=10, num_points = 4096, train=True):
        super(SegDataset, self).__init__()

        self.frames_per_clip = frames_per_clip
        self.train = train
        self.num_points = num_points

        self.pcd = []
        self.center = []
        self.pcd_complete = []

        lines = []
        with open(meta,encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                lines.append(line)
        print(len(lines))

        for i, fil in enumerate(lines):
            print(i, fil)
            with h5py.File(root+'/'+fil+'/h5.h5','r') as f:
                self.pcd.append(np.array(f['pcd']))
                self.center.append(np.array(f['center']))        
        for filename in ['train1.h5', 'train2.h5', 'train3.h5', 'train4.h5']:
        # for filename in ['train1.h5']:
            print(filename)
            with h5py.File(root_complete+'/'+filename,'r') as f:
                self.pcd_complete.append(np.array(f['pcd']))

    def __len__(self):
        leng = 2971 * 94
        # leng=750
        return leng

    def augment(self, pc, center):
        flip = np.random.uniform(0, 1) > 0.5
        if flip:
            pc = pc - center
            jittered_data = np.clip(0.01 * np.random.randn(self.frames_per_clip,self.num_points,3), -1*0.05, 0.05)
            jittered_data += pc
            pc = pc + center

        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center

        rot_axis = np.array([0, 1, 0])
        rot_angle = np.random.uniform(np.pi * -0.05, np.pi * 0.05)
        q = Quaternion(axis=rot_axis, angle=rot_angle)
        R = q.rotation_matrix

        pc = np.dot(pc - center, R) + center
        return pc

    def __getitem__(self, index):
        s = int(index / 94)
        d = index % 94

        pc = self.pcd[s][d]
        center = self.center[s][d][0]

        s_ = int(s / 750)
        d_ = s % 750
        pc_complete = self.pcd_complete[s_][d_]

        cho = np.array(range(0,20,2))
        pc_complete = pc_complete[int(d*3):int(d*3+self.frames_per_clip*2)][cho]

        pc = self.augment(pc, center)

        return pc.astype(np.float32), pc_complete.astype(np.float32)


if __name__ == '__main__':
    datasets = SegDataset(root='/share/datasets/Seg_pre_last_h5', root_complete='/share/datasets/Seg_data_base_4096_h5', meta='/HOI4D_splits/release.txt',frames_per_clip=10, num_points = 4096, train=True)