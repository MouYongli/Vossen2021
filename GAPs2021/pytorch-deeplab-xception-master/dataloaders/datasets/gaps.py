import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

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
    def __init__(self, args, split='train', root_dir=None, transform=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            split (string): Split of dataset [ train | valid | test]
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.args = args
        if root_dir is not None:
            self.root_dir = root_dir
        else:
            pwd = osp.dirname(osp.abspath(__file__))
            self.root_dir = osp.join(pwd, '../../../data')
        self.split = split
        self.transform = transform
        basedir = os.path.join(self.root_dir, 'v2', 'segmentation')
        self.images_dir = os.path.join(basedir, 'images')
        self.label_dir = os.path.join(basedir, split)
        if self.args.focus_on_minority:
            self.data_df = pd.read_csv(osp.join(self.root_dir, 'minority_data.csv'))
        else:
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
        height, width = image.shape[0], image.shape[1]

        if self.split == 'train':
            if self.args.crop_strategy == 'resize':
                image = cv2.resize(image, (512, 512))
                label = cv2.resize(label, (512, 512))
            elif self.args.crop_strategy == 'rand':
                x_cord =np.random.randint(256,824)
                y_cord =np.random.randint(256,1664)
                image = image[(x_cord - 256):(x_cord + 256), (y_cord - 256):(y_cord + 256)]
                label = label[(x_cord - 256):(x_cord + 256), (y_cord - 256):(y_cord + 256)]
            elif self.args.crop_strategy == 'global-prob':
                prob = np.array([9.053839e+04, 1.833789e+06, 7.457733e+03, 1.792296e+03, 4.969282e+03, 2.587876e+03,
                                 5.355967e+04, 7.890600e+04])  # lbl_i表示的概率设置为prob_i
                prob = 1. / np.log(prob)
                prob[1] = 0.0
                ps = np.exp(prob[label[256:(height - 256), 256:(width - 256)]].reshape(-1))
                ps /= np.sum(ps)
                point = np.random.choice((width - 512) * (height - 512), 1, p=ps)
                y_cord = (point[0] % (width - 512)) + 256
                x_cord = (point[0] // (width - 512)) + 256
                image = image[(x_cord - 256):(x_cord + 256), (y_cord - 256):(y_cord + 256)]
                label = label[(x_cord - 256):(x_cord + 256), (y_cord - 256):(y_cord + 256)]
            elif self.args.crop_strategy == 'local-prob':
                res = dict(sum(map(Counter, label[256:(height - 256), 256:(width - 256)]), Counter()))
                for i in range(8):
                    if i not in res.keys():
                        res[i] = 0
                prob = np.array([res[i] for i in range(8)]) / ((width - 512) * (height - 512))
                prob = 1. / np.exp(prob)
                prob[1] = 0.0
                ps = np.exp(prob[label[256:(height - 256), 256:(width - 256)]].reshape(-1))
                ps /= np.sum(ps)
                point = np.random.choice((width - 512) * (height - 512), 1, p=ps)
                y_cord = (point[0] % (width - 512)) + 256
                x_cord = (point[0] // (width - 512)) + 256
                image = image[(x_cord - 256):(x_cord + 256), (y_cord - 256):(y_cord + 256)]
                label = label[(x_cord - 256):(x_cord + 256), (y_cord - 256):(y_cord + 256)]
            else:
                raise NotImplemented('Dataset crop strategy {} is not implemented'.format(self.args.crop_strategy))

        if self.transform:
            image, label = self.transforms(image, label)
        return {'image': image, 'label': label}

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
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--crop-strategy', type=str, default='local-prob',
                        choices=['resize', 'rand', 'global-prob', 'local-prob'],
                        help='crop strategy (default: rand)')
    parser.add_argument('--focus-on-minority', type=bool, default=True,
                        help='training data focus on minority classes')
    args = parser.parse_args()
    dataset = GapsDataset(args, split='train', root_dir=None, transform=True)
    print(len(dataset))
    sample = dataset[0]
    img, lbl = sample['image'], sample['label']
    img, lbl = dataset.untransforms(img, lbl)
    print(img.shape)
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(lbl)
    plt.show()

