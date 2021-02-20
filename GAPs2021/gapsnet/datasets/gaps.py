import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import cv2

class GapsDataset(Dataset):
    """GAPs dataset."""
    class_names = np.array([
        'VOID',
        'intact road',
        'applied patch',
        'pothole',
        'inlaid patch',
        'open joint',
        'crack',
        'street inventory'
    ])
    def __init__(self, root_dir=None, split='train', transform=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): Split of dataset [ train | valid | test]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if root_dir is not None:
            self.root_dir = root_dir
        else:
            pwd = osp.dirname(osp.abspath(__file__))
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
        image = cv2.imread(img_file_path)
        label = cv2.imread(lbl_file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))
        label = cv2.resize(label, (256, 256))

        # image = image[..., np.newaxis]
        if self.transform:
            image, label = self.transforms(image, label)
        return image, label

    def transforms(self, img, lbl):
        img = img.astype(np.float64)
        img /= 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransforms(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img *= 255.0
        img = img.astype(np.uint8)
        lbl = lbl.numpy()
        return img, lbl


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = GapsDataset()
    imgs, lbls = dataset[0]
    print(imgs.shape)
    print(lbls.shape)
    img, lbl = dataset.untransforms(imgs, lbls)
    plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(lbl)
    plt.show()
