import os
import sys
import numpy as np
from pyquaternion import Quaternion
from torch.utils.data import Dataset
import h5py

class SegDataset(Dataset):
    def __init__(self, root= None, num_points= None, frames_per_clip=12, train= True):
        super(SegDataset, self).__init__()
        self.train = train
        self.num_points = num_points
        self.frames_per_clip = frames_per_clip

        self.pcs_pre = []
        self.rgbs_pre = []
        self.pcs = []
        self.rgbs = []
        self.centers = []

        if self.train:
            for filename in ['FOG.h5', 'FALL.h5', 'DAWN.h5', 'NIGHT.h5', 'RAINNIGHT.h5', 'SOFTRAIN.h5', 'SUMMER.h5', 'WINTER.h5', 'WINTERNIGHT.h5']:
            # for filename in ['FOG.h5']:
            # for filename in ['HI.h5']:
                print(filename)
                with h5py.File(root+'/'+filename,'r') as f:
                    self.pcs_pre.append(np.array(f['pcs_pre']))
                    self.rgbs_pre.append(np.array(f['rgbs_pre']))
                    self.pcs.append(np.array(f['pcs']))
                    self.rgbs.append(np.array(f['rgbs']))
                    self.centers.append(np.array(f['centers']))

        self.pcs_pre = np.concatenate(self.pcs_pre, axis=0)
        self.rgbs_pre = np.concatenate(self.rgbs_pre, axis=0)
        self.pcs = np.concatenate(self.pcs, axis=0)
        self.rgbs = np.concatenate(self.rgbs, axis=0)
        self.centers = np.concatenate(self.centers, axis=0)

    def __len__(self):
        return len(self.pcs)

    def augment(self, pc, center):
        flip = np.random.uniform(0, 1) > 0.5
        if flip:
            pc = (pc - center)
            pc[:, 0] *= -1
            pc += center

        scale = np.random.uniform(0.8, 1.2)
        pc = (pc - center) * scale + center

        rot_axis = np.array([0, 1, 0])
        rot_angle = np.random.uniform(np.pi * 2)
        q = Quaternion(axis=rot_axis, angle=rot_angle)
        R = q.rotation_matrix

        pc = np.dot(pc - center, R) + center
        return pc

    def __getitem__(self, index):
        pc_pre = self.pcs_pre[index]
        rgb_pre = self.rgbs_pre[index]
        pc = self.pcs[index]
        rgb = self.rgbs[index]

        if pc_pre.shape[1] > self.num_points:
            r = np.random.choice(pc_pre.shape[1], self.num_points, replace=False)
            pc_pre = pc_pre[:,r]
            rgb_pre = rgb_pre[:,r]
            pc = pc[:,r]
            rgb = rgb[:,r]

        center = self.centers[index][0]

        if self.train:
            pc_pre = self.augment(pc_pre, center)

        rgb_pre = np.swapaxes(rgb_pre, 1, 2)
        rgb = np.swapaxes(rgb, 1, 2)

        return pc_pre.astype(np.float32), rgb_pre.astype(np.float32), pc.astype(np.float32), rgb.astype(np.float32)

if __name__ == '__main__':
    dataset = SegDataset(
            root='/data/Synthia_pre',
            num_points=8192,
            frames_per_clip=12,
            train=True
    )
