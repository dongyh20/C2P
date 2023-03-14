import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import math
import cv2
# import open3d as o3d
import random
import time
import h5py
import torch.multiprocessing as mlp


def ball(angle1, angle2):
    ''' calculate the normalized view direction given horizontal angle1(\theta) and vertical angle2(\phi), 
        will be used in function oper'''
    x = math.cos(angle1/180 * math.pi)
    z = math.sin(angle1/180 * math.pi)
    y = math.sin(angle2/180 * math.pi)
    direct = np.array([x,y,z])
    direct = -1 * direct / np.linalg.norm(direct)
    return direct

def oper(points, angle1, angle2, f = None, center = None):
    ''' partial view generation for one frame(one point cloud) given horizontal angle1(\theta) and vertical angle2'''
    if f is None: 
        ''' f is the focal length of the camera, you can adjust f according to your demand. 
            Large f will hide too many points while samll f can not hide invisible points'''
        f = 150

    if center is None:
        ''' center is the center of the point cloud'''
        center = np.mean(points, axis=0)

    new_camera = np.linalg.norm(center) * ball(angle1, angle2) + center
    ''' calculate the position of the new camera which generates the partial view'''

    z_vec  = center - new_camera
    z_vec = z_vec / np.linalg.norm(z_vec)
    aux_vec = np.array([0,1,0])
    x_vec = np.cross(aux_vec, z_vec)
    x_vec = x_vec / np.linalg.norm(x_vec)
    y_vec = np.cross(z_vec, x_vec)
    '''determine the pose of the new camera, the z-axis of the new camera directs to the center of the point cloud'''
    
    points_in_camera_coordiante = points - new_camera

    xyz = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    xyz_view = np.concatenate([x_vec.reshape(3,1), y_vec.reshape(3,1), z_vec.reshape(3,1)],axis=1)
    trans = xyz @ np.linalg.inv(xyz_view)
    points_in_camera_coordiante = (trans @ points_in_camera_coordiante.T).T
    '''transform points from the original camera coordiante to the new camera coordinate'''

    screenZBuff_x = 120
    screenZBuff_y = 180
    '''size of the camera screen, you can adjust them but please make sure they are large enough 
        to hold all the projected points to avoid loss of points'''
    screenZBuff = np.full([screenZBuff_x, screenZBuff_y], -1.0)
    pixelIds = [[-1, -1] for i in range(points_in_camera_coordiante.shape[0])]
    zVals = [10.0 for i in range(points_in_camera_coordiante.shape[0])]

    for i in range(points_in_camera_coordiante.shape[0]):
            z = points_in_camera_coordiante[i][2]
            p = points_in_camera_coordiante[i] * f / z
            if p[0] < 1e6 and p[1] < 1e6:                               
                '''check whether p is inf'''
                px = int(p[0] + screenZBuff_x*0.5)
                py = int(p[1] + screenZBuff_y*0.5)
                zVals[i] = z
                pixelIds[i] = [px, py]
                if px < screenZBuff_x and  px >= 0 and py < screenZBuff_y and py >= 0:
                    if screenZBuff[px,py] > z or screenZBuff[px,py] < 0.0:
                        screenZBuff[px,py] = z
            else:
                continue
    '''calculate the projection and record them in screenZBuff'''

    outpoints = []
    for i in range(points_in_camera_coordiante.shape[0]):
        if pixelIds[i][0] >= 0 and pixelIds[i][0] < screenZBuff_x and pixelIds[i][1] >= 0 and pixelIds[i][1] < screenZBuff_y:###
            if (zVals[i] - screenZBuff[pixelIds[i][0]][pixelIds[i][1]]) < 0.001:
                outpoints.append(points_in_camera_coordiante[i])
    out = np.array(outpoints)

    # screenZBuff = screenZBuff.T
    # cv2.imwrite('look.png', screenZBuff*100)
    '''the projection image'''
    
    return out, new_camera, trans  
    '''output points is now in the new camera coordiante, 
        if you want to transform back to the original camera coordiante, just do:
            out = (np.linalg.inv(trans) @ out.T).T
            out = out + new_camera'''

class MSRAction3D_partial(Dataset):
    def __init__(self, root, frames_per_clip=16, step_between_clips=1, num_points=2048, train=True):
        super(MSRAction3D_partial, self).__init__()

        self.videos = []
        self.primitives = []
        self.labels = []
        self.index_map = []
        index = 0
        for video_name in os.listdir(root):
            if video_name == 'a12_s03_e02_sdepth.npz' or video_name == 'a18_s05_e03_sdepth.npz' or video_name == 'a12_s08_e03_sdepth.npz':
                continue
            if train and (int(video_name.split('_')[1].split('s')[1]) <= 5):
                video = np.load(os.path.join(root, video_name), allow_pickle=True)['point_clouds']
                primitive = np.load(os.path.join('/home/yuhao/msrfitting/afterFitting/', video_name), allow_pickle=True)['primitives']
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
                primitive = np.load(os.path.join('/home/yuhao/msrfitting/afterFitting/', video_name), allow_pickle=True)['primitives']
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

        '''below is the generation part'''
        hor_angle_0 = np.arange(15, 166, 150/(self.frames_per_clip-1))
        hor_angle_1 = np.arange(15, 166, 150/(self.frames_per_clip-1))[::-1]
        '''horizontal angle change patterns'''

        ver_angle_0 = np.zeros(self.frames_per_clip)

        E = np.arange(0,4.99999,5/(self.frames_per_clip/2))
        F = np.arange(5,0.00001,-5/(self.frames_per_clip/2))
        ver_angle_1 = np.concatenate([E,F])

        G = np.arange(0,-4.99999,-5/(self.frames_per_clip/2))
        H = np.arange(-5,-0.00001,5/(self.frames_per_clip/2))
        ver_angle_2 = np.concatenate([G,H])
        '''vertical angle change patterns'''

        sel_hor = random.randint(0, 1)
        if sel_hor == 0:
            hor_angle_select = hor_angle_0
        else:
            hor_angle_select = hor_angle_1

        sel_ver = random.randint(0, 2)
        if sel_ver == 0:
            ver_angle_select = ver_angle_0
        elif sel_ver == 1:
            ver_angle_select = ver_angle_1
        else:
            ver_angle_select = ver_angle_2

        clip = [video[t+i*self.step_between_clips] for i in range(self.frames_per_clip)]
        partial = []
        complete = []

        for i, p in enumerate(clip):
            out, camera, trans = oper(clip[i], hor_angle_select[i], ver_angle_select[i])

            if out.shape[0] > self.num_points:
                r = np.random.choice(out.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // out.shape[0], self.num_points % out.shape[0]
                r = np.random.choice(out.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(out.shape[0]) for _ in range(repeat)] + [r], axis=0)
            partial.append((np.linalg.inv(trans) @ out[r, :].T).T + camera)
            '''transform back to original camera coordinate '''

            if p.shape[0] > self.num_points:
                r = np.random.choice(p.shape[0], size=self.num_points, replace=False)
            else:
                repeat, residue = self.num_points // p.shape[0], self.num_points % p.shape[0]
                r = np.random.choice(p.shape[0], size=residue, replace=False)
                r = np.concatenate([np.arange(p.shape[0]) for _ in range(repeat)] + [r], axis=0)
            complete.append(p[r, :])

        clip = np.array(partial)
        clip_complete = np.array(complete)
        
        if self.train:
            scales = np.random.uniform(0.9, 1.1, size=3)
            clip = clip * scales

        clip = clip / 300
        clip_complete = clip_complete / 300

        return clip.astype(np.float32), clip_complete.astype(np.float32), label, index

if __name__ == '__main__':
    dataset = MSRAction3D_partial(root='processed_data', frames_per_clip=24)
    print(len(dataset))

    clip_list = []
    clip_complete_list = []
    for i in range(len(dataset)):
        print(i)
        clip, clip_complete, label, video_idx = dataset[i]
        clip_list.append(clip)
        clip_complete_list.append(clip_complete)
    clip_array = np.stack(clip_list, axis=0)
    clip_complete_array = np.stack(clip_complete_list, axis=0)
    with h5py.File('msr_data_pre.h5','w') as f:
        f.create_dataset('clip',data=np.array(clip_array))
        f.create_dataset('clip_complete',data=np.array(clip_complete_array))
    