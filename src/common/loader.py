import torch
from torch.utils.data import DataLoader

import cv2
import numpy as np
import pandas as pd
from .util import alignment, face_ToTensor

_lfw_root = '../datasets/lfw/'
_lfw_landmarks = '../data/LFW.csv'
_lfw_pairs = '../data/lfw_pairs.txt'


class LFWDataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super(LFWDataset, self).__init__()
        self.args = args
        df = pd.read_csv(_lfw_landmarks, delimiter=",", header=None)
        numpyMatrix = df.values
        self.landmarks = numpyMatrix[:, 1:]
        self.df = df
        with open(_lfw_pairs) as f:
            pairs_lines = f.readlines()[1:]
        self.pairs_lines = pairs_lines

    def __getitem__(self, index):
        p = self.pairs_lines[index].replace('\n', '').split('\t')
        if 3 == len(p):
            sameflag = np.int32(1).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = np.int32(0).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        img1 = alignment(cv2.imread(_lfw_root + name1),
                         self.landmarks[self.df.loc[self.df[0] == name1].index.values[0]])
        img2 = alignment(cv2.imread(_lfw_root + name2),
                         self.landmarks[self.df.loc[self.df[0] == name2].index.values[0]])
        ## Resize second image
        if self.args.size != -1:
            ## Use args.size
            img2 = cv2.resize(img2, (self.args.size, self.args.size), interpolation=cv2.INTER_CUBIC)
        else:
            ## Use args.down_factor
            img2 = cv2.resize(img2, None, fx=1 / self.args.down_factor, fy=1 / self.args.down_factor,
                              interpolation=cv2.INTER_CUBIC)

        ## Resize the to the required size of FNet
        img1 = cv2.resize(img1, (self.args.w, self.args.h), cv2.INTER_CUBIC)
        if not self.args.isSR:
            img2 = cv2.resize(img2, (self.args.w, self.args.h), cv2.INTER_CUBIC)

        ## Obtain the mirror faces
        img1_flip = cv2.flip(img1, 1)
        img2_flip = cv2.flip(img2, 1)

        return face_ToTensor(img1), face_ToTensor(img2), \
               face_ToTensor(img1_flip), face_ToTensor(img2_flip), \
               torch.LongTensor(sameflag)

    def __len__(self):
        return len(self.pairs_lines)


def get_loader(args, name, num_workers=4):
    if name == 'lfw':
        dataset = LFWDataset(args)
        dataloader = DataLoader(dataset=dataset,
                                num_workers=num_workers,
                                batch_size=args.lfw_bs,
                                shuffle=False,
                                drop_last=False)
    return dataloader
