import os
import os.path as osp
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import cv2

class GapsDataset(Dataset):
    """GAPs dataset."""

    def __init__(self, root_dir=None, split='train', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): Split of dataset [ train | valid | valid-test]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if root_dir is not None:
            self.root_dir = root_dir
        else:
            pwd = os.getcwd()
            self.root_dir = osp.join(pwd, '../../data')
        self.split = split
        self.transform = transform
        basedir = os.path.join(self.root_dir, 'v2', 'segmentation')
        self.images_dir = os.path.join(basedir, 'images')
        self.label_dir = os.path.join(basedir, split)
        list_label = os.listdir(self.label_dir)
        self.data_df = pd.DataFrame(np.array([list_label, list_label]).T, columns=['id', 'lbl_file'])
        self.data_df = self.data_df[self.data_df['id'].str.endswith('.png')]
        self.data_df['id'] = self.data_df['lbl_file'].map(lambda x: x[:-4])
        self.data_df['img_file'] = self.data_df['id'].map(lambda x: "%s.jpg"%x)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_file_path = os.path.join(self.images_dir,  self.data_df['img_file'][idx])
        lbl_file_path = os.path.join(self.label_dir,  self.data_df['lbl_file'][idx])
        image = cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(lbl_file_path, cv2.IMREAD_GRAYSCALE)
        image, label = image[..., np.newaxis], label[..., np.newaxis]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

if __name__ == '__main__':
    dataset = GapsDataset()
    sample = dataset[0]
    img, lbl = sample['image'], sample['label']
    img, lbl = img[:,:,0], lbl[:,:,0]
    print(img.shape)
    print(lbl.shape)
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(lbl)
    plt.show()
