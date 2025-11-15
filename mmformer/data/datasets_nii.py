import os
import torch
from torch.utils.data import Dataset

from .rand import Uniform
from .transforms import Rot90, Flip, Identity, Compose
from .transforms import GaussianBlur, Noise, Normalize, RandSelect
from .transforms import RandCrop, CenterCrop, Pad,RandCrop3D,RandomRotion,RandomFlip,RandomIntensityChange
from .transforms import NumpyType
from .data_utils import pkload

import numpy as np
import nibabel as nib
import glob
join = os.path.join

HGG = []
LGG = []
for i in range(0, 260):
    HGG.append(str(i).zfill(3))
for i in range(336, 370):
    HGG.append(str(i).zfill(3))
for i in range(260, 336):
    LGG.append(str(i).zfill(3))

mask_array = np.array([[True, False, False, False, True], [False, True, False, False, True], [False, False, True, False, True], [False, False, False, True, True],
                      [True, True, False, False, True], [True, False, True, False, True], [True, False, False, True, True], [False, True, True, False, True], [False, True, False, True, True], [False, False, True, True, True], [True, True, True, False, True], [True, True, False, True, True], [True, False, True, True, True], [False, True, True, True, True],
                      [True, True, True, True, True]])

class Brats_loadall_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
        # data_file_path = os.path.join(root, train_file)
        # with open(data_file_path, 'r') as f:
        #     datalist = [i.strip() for i in f.readlines()]
        # datalist.sort()

        # volpaths = []
        # for dataname in datalist:
        #     volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        '''Yao'''
        patients_dir = glob.glob(join(root, 'vol_addcanny', '*_vol.npy'))
        patients_dir.sort(key=lambda x: x.split('/')[-1][:-8])
        print('###############', len(patients_dir))
        n_patients = len(patients_dir)
        pid_idx = np.arange(n_patients)
        np.random.seed(0)
        np.random.shuffle(pid_idx)
        n_fold_list = np.split(pid_idx, 3)

        volpaths = []
        for i, fold in enumerate(n_fold_list):
            if i != 0:
                for idx in fold:
                    volpaths.append(patients_dir[idx])
        datalist = [x.split('/')[-1].split('_vol')[0] for x in volpaths]

        # hfn加入edge
        # edge_root = 'D:/Data_seg/BRATS2020_Training_canny_npy'
        # datanumber = [x[-3:] for x in datalist]
        #
        # edge_path_list = [join(edge_root, 'seg_crop', 'BraTS20_Training_'+x+'_canny.npy') for x in datanumber]


        '''Yao'''

        self.volpaths = volpaths
        # self.edgepaths = edge_path_list
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3,4])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        # edgepath = self.edgepaths[index]

        # edge = np.load(edgepath)
        x = np.load(volpath)
        segpath = volpath.replace('vol_addcanny', 'seg').replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]
        # normalize x to [N, H, W, D, C] before transforms
        if x.ndim == 4:
            x = x[..., None]
        elif x.ndim == 5 and (x.shape[1] in (4, 5)) and (x.shape[-1] not in (4, 5)):
            x = np.transpose(x, (0, 2, 3, 4, 1))

        x,y = self.transforms([x, y])

        # edge = self.transforms([edge])
        # normalize x shape to [N, H, W, D, C] regardless of saved layout
        # cases seen: [N, H, W, D, C], [N, C, H, W, D], or [N, H, W, D]
        if x.ndim == 4:
            # missing channel dim, assume single channel
            x = x[..., None]
        elif x.ndim != 5:
            raise ValueError(f"Unexpected x ndim: {x.ndim}, expected 4 or 5")
        # if channels at axis 1 (N, C, H, W, D), move to last
        if x.shape[1] in (4, 5) and x.shape[-1] not in (4, 5):
            x = np.transpose(x, (0, 2, 3, 4, 1))
        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        # edge = np.reshape(edge, (-1))

        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))
        #
        # edgeo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        # edgeo = np.ascontiguousarray(edgeo.transpose(0, 4, 1, 2, 3))

        # ensure required channels exist (e.g., 5 for 'all') by duplicating last channel if needed
        required_c = int(np.max(self.modal_ind)) + 1 if isinstance(self.modal_ind, np.ndarray) else x.shape[1]
        c = x.shape[1]
        if c < required_c:
            pad = np.repeat(x[:, -1:, ...], required_c - c, axis=1)
            x = np.concatenate([x, pad], axis=1)
        # select requested channels
        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        # edgeo = torch.squeeze(torch.from_numpy(edgeo), dim=0)

        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', test_file='test.txt'):
        # data_file_path = os.path.join(root, test_file)
        # with open(data_file_path, 'r') as f:
        #     datalist = [i.strip() for i in f.readlines()]
        # datalist.sort()
        # volpaths = []
        # for dataname in datalist:
        #     volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        
        '''Yao'''
        patients_dir = glob.glob(join(root, 'vol_addcanny', '*_vol.npy'))
        patients_dir.sort(key=lambda x: x.split('/')[-1][:-8])
        n_patients = len(patients_dir)
        pid_idx = np.arange(n_patients)
        np.random.seed(0)
        np.random.shuffle(pid_idx)
        n_fold_list = np.split(pid_idx, 3)

        volpaths = []
        for i, fold in enumerate(n_fold_list):
            if i == 0:
                for idx in fold:
                    volpaths.append(patients_dir[idx])
        datalist = [x.split('/')[-1].split('_vol')[0] for x in volpaths]
        '''Yao'''

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3, 4])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol_addcanny', 'seg').replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        # normalize x to [N, H, W, D, C] before transforms
        if x.ndim == 4:
            x = x[..., None]
        elif x.ndim == 5 and (x.shape[1] in (4, 5)) and (x.shape[-1] not in (4, 5)):
            x = np.transpose(x, (0, 2, 3, 4, 1))
        x,y = self.transforms([x, y])
        # normalize x shape to [N, H, W, D, C]
        if x.ndim == 4:
            x = x[..., None]
        elif x.ndim != 5:
            raise ValueError(f"Unexpected x ndim: {x.ndim}, expected 4 or 5")
        if x.shape[1] in (4, 5) and x.shape[-1] not in (4, 5):
            x = np.transpose(x, (0, 2, 3, 4, 1))
        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        # ensure y spatial dims match x (crop or pad as needed)
        xH, xW, xZ = x.shape[2:5]
        yH, yW, yZ = y.shape[1:4]
        # crop if larger
        y = y[:, :xH, :xW, :xZ]
        # pad if smaller
        padH = max(0, xH - y.shape[1])
        padW = max(0, xW - y.shape[2])
        padZ = max(0, xZ - y.shape[3])
        if padH or padW or padZ:
            y = np.pad(y, ((0,0),(0,padH),(0,padW),(0,padZ)), mode='constant')
        # ensure required channels exist by duplicating last channel if needed
        required_c = int(np.max(self.modal_ind)) + 1 if isinstance(self.modal_ind, np.ndarray) else x.shape[1]
        c = x.shape[1]
        if c < required_c:
            pad = np.repeat(x[:, -1:, ...], required_c - c, axis=1)
            x = np.concatenate([x, pad], axis=1)
        # select requested channels
        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, name

    def __len__(self):
        return len(self.volpaths)

class Brats_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, settype='train', modal='all'):
        data_file_path = os.path.join(root, 'val.txt')
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])
        # normalize x shape to [N, H, W, D, C]
        if x.ndim == 4:
            x = x[..., None]
        elif x.ndim != 5:
            raise ValueError(f"Unexpected x ndim: {x.ndim}, expected 4 or 5")
        if x.shape[1] in (4, 5) and x.shape[-1] not in (4, 5):
            x = np.transpose(x, (0, 2, 3, 4, 1))
        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)
        # ensure required channels exist by duplicating last channel if needed
        required_c = int(np.max(self.modal_ind)) + 1 if isinstance(self.modal_ind, np.ndarray) else x.shape[1]
        c = x.shape[1]
        if c < required_c:
            pad = np.repeat(x[:, -1:, ...], required_c - c, axis=1)
            x = np.concatenate([x, pad], axis=1)
        # select requested channels
        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        mask = mask_array[index%15]
        mask = torch.squeeze(torch.from_numpy(mask), dim=0)
        return x, y, mask, name

    def __len__(self):
        return len(self.volpaths)
