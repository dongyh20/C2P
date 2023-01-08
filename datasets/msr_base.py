import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

class MSRAction3D(Dataset):
    def __init__(self, root, frames_per_clip=24, step_between_clips=1, num_points=2048, train=True):
        super(MSRAction3D, self).__init__()

        self.videos = []
        self.primitives = []
        self.labels = []
        self.index_map = []
        index = 0
        for video_name in os.listdir(root):
            if video_name == 'a12_s03_e02_sdepth.npz' or video_name == 'a18_s05_e03_sdepth.npz' or video_name == 'a12_s08_e03_sdepth.npz':
                continue
            if train and (int(video_name.split('_')[1].split('s')[1]) <= 5):
                # print(os.path.join(root, video_name))
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                primitive = np.load(os.path.join('/msrfitting/afterFitting/', video_name), allow_pickle=True)['primitives']
                # print('Done')
                self.videos.append(video)
                self.primitives.append(primitive)
                label = int(video_name.split('_')[0][1:])-1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
                    self.index_map.append((index, t))
                index += 1

            if not train and (int(video_name.split('_')[1].split('s')[1]) > 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                primitive = np.load(os.path.join('/msrfitting/afterFitting/', video_name), allow_pickle=True)['primitives']
                self.videos.append(video)
                self.primitives.append(primitive)
                label = int(video_name.split('_')[0][1:])-1
                self.labels.append(label)

                nframes = video.shape[0]
                for t in range(0, nframes-step_between_clips*(frames_per_clip-1), step_between_clips):
                    self.index_map.append((index, t))
                index += 1

        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.num_points = num_points
        self.train = train
        self.num_classes = max(self.labels) + 1


    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        index, t = self.index_map[idx]

        video = self.videos[index]
        primitive = self.primitives[index]
        label = self.labels[index]

        L = video.shape[0]

        for clip_id in range(L):
            tmp = np.argsort(np.squeeze(primitive[clip_id]))
            video[clip_id] = video[clip_id][tmp]

        # L = video.shape[0]
        
        # for clip_id in range(L):
        #     np.random.shuffle(video[clip_id])

        clip = [video[t+i*self.step_between_clips] for i in range(self.frames_per_clip)]
        for i, p in enumerate(clip):
            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            clip[i] = p[r, :]
        clip = np.array(clip)

        if self.train:
            # scale the points
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

        clip = clip / 300


        return clip.astype(np.float32), label, index


if __name__ == '__main__':
    primitive = np.load('/msrfitting/afterFitting/a12_s03_e02_sdepth.npz', allow_pickle=True)['primitives']