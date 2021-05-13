import os
import os.path as osp
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
import cv2

class Crack500Dataset(Dataset):
    """
    Crack500 dataset
    """
    class_names = np.array([
        'road',
        'crack'
    ])

    def __init__(self, args, base_dir=None, cropped_dataset=False, split='train', transform=True):
        """
        :param base_dir: path to crack500 dataset directory
        :param split: train/test
        :param cropped_dataset: apply crop image dataset
        :param transform: transform to apply
        """
        super().__init__()
        if base_dir is not None:
            self.base_dir = base_dir
        else:
            pwd = osp.dirname(osp.abspath(__file__))
            self.base_dir = osp.join(pwd, '../../../data/crack500')
        self.split = split
        self.cropped_dataset = cropped_dataset
        self.transform = transform
        self.image_dir = osp.join(self.base_dir, self.split+'crop' if cropped_dataset else self.split+'data')
        self.label_dir = osp.join(self.base_dir, self.split+'crop' if cropped_dataset else self.split+'data')
        self.args = args

        files = os.listdir(self.image_dir)
        img_data_df = pd.DataFrame(np.array([files, files]).T, columns=['id', 'img_file'])
        img_data_df = img_data_df[img_data_df['id'].str.endswith('.jpg')]
        img_data_df['id'] = img_data_df['img_file'].map(lambda x: x[:-4])
        lbl_data_df = pd.DataFrame(np.array([files, files]).T, columns=['id', 'lbl_file'])
        if self.cropped_dataset:
            lbl_data_df = lbl_data_df[lbl_data_df['id'].str.endswith('.png')]
            lbl_data_df['id'] = lbl_data_df['lbl_file'].map(lambda x: x[:-4])
        else:
            lbl_data_df = lbl_data_df[lbl_data_df['id'].str.endswith('_mask.png')]
            lbl_data_df['id'] = lbl_data_df['lbl_file'].map(lambda x: x[:-9])
        self.data_df = pd.merge(img_data_df, lbl_data_df, on='id', how='inner')

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        id = self.data_df['id'][idx]
        img_file_path = os.path.join(self.image_dir, self.data_df['img_file'][idx])
        lbl_file_path = os.path.join(self.label_dir, self.data_df['lbl_file'][idx])
        image = cv2.imread(img_file_path)
        label = cv2.imread(lbl_file_path, cv2.IMREAD_GRAYSCALE)
        label = label/255
        height, width = image.shape[0], image.shape[1]
        crop_size = self.args.crop_size
        if self.cropped_dataset:
            raise NotImplementedError
        else:
            if self.split == 'train':
                if self.args.train_crop_strategy == 'origin':
                    pass
                elif self.args.train_crop_strategy == 'resize':
                    image = cv2.resize(image, (crop_size, crop_size))
                    label = cv2.resize(label, (crop_size, crop_size))
                elif self.args.train_crop_strategy == 'rand':
                    x_cord = np.random.randint(crop_size // 2, height - crop_size // 2 - 1)
                    y_cord = np.random.randint(crop_size // 2, width - crop_size // 2 - 1)
                    image = image[(x_cord - crop_size // 2):(x_cord + crop_size // 2),
                            (y_cord - crop_size // 2):(y_cord + crop_size // 2)]
                    label = label[(x_cord - crop_size // 2):(x_cord + crop_size // 2),
                            (y_cord - crop_size // 2):(y_cord + crop_size // 2)]
                elif self.args.train_crop_strategy == 'prob':
                    ps = label[crop_size // 2:(height - crop_size // 2), crop_size // 2:(width - crop_size // 2)].reshape(
                        -1)
                    ps = ps / np.sum(ps)
                    point = np.random.choice((width - crop_size) * (height - crop_size), 1, p=ps)
                    y_cord = (point[0] % (width - crop_size)) + crop_size // 2
                    x_cord = (point[0] // (width - crop_size)) + crop_size // 2
                    image = image[(x_cord - crop_size // 2):(x_cord + crop_size // 2),
                            (y_cord - crop_size // 2):(y_cord + crop_size // 2)]
                    label = label[(x_cord - crop_size // 2):(x_cord + crop_size // 2),
                            (y_cord - crop_size // 2):(y_cord + crop_size // 2)]
                else:
                    raise NotImplementedError
            elif self.split == 'val':
                if self.args.val_crop_strategy == 'origin':
                    pass
                elif self.args.val_crop_strategy == 'resize':
                    image = cv2.resize(image, (crop_size, crop_size))
                    label = cv2.resize(label, (crop_size, crop_size))
                elif self.args.val_crop_strategy == 'rand':
                    x_cord = np.random.randint(crop_size // 2, height - crop_size // 2 - 1)
                    y_cord = np.random.randint(crop_size // 2, width - crop_size // 2 - 1)
                    image = image[(x_cord - crop_size // 2):(x_cord + crop_size // 2),
                            (y_cord - crop_size // 2):(y_cord + crop_size // 2)]
                    label = label[(x_cord - crop_size // 2):(x_cord + crop_size // 2),
                            (y_cord - crop_size // 2):(y_cord + crop_size // 2)]
                elif self.args.val_crop_strategy == 'prob':
                    ps = label[crop_size // 2:(height - crop_size // 2), crop_size // 2:(width - crop_size // 2)].reshape(
                        -1)
                    ps = ps / np.sum(ps)
                    point = np.random.choice((width - crop_size) * (height - crop_size), 1, p=ps)
                    y_cord = (point[0] % (width - crop_size)) + crop_size // 2
                    x_cord = (point[0] // (width - crop_size)) + crop_size // 2
                    image = image[(x_cord - crop_size // 2):(x_cord + crop_size // 2),
                            (y_cord - crop_size // 2):(y_cord + crop_size // 2)]
                    label = label[(x_cord - crop_size // 2):(x_cord + crop_size // 2),
                            (y_cord - crop_size // 2):(y_cord + crop_size // 2)]
                else:
                    raise NotImplementedError
            elif self.split == 'test':
                if self.args.test_crop_strategy == 'origin':
                    pass
                elif self.args.test_crop_strategy == 'resize':
                    image = cv2.resize(image, (crop_size, crop_size))
                    label = cv2.resize(label, (crop_size, crop_size))
                elif self.args.test_crop_strategy == 'rand':
                    x_cord = np.random.randint(crop_size // 2, height - crop_size // 2 - 1)
                    y_cord = np.random.randint(crop_size // 2, width - crop_size // 2 - 1)
                    image = image[(x_cord - crop_size // 2):(x_cord + crop_size // 2),
                            (y_cord - crop_size // 2):(y_cord + crop_size // 2)]
                    label = label[(x_cord - crop_size // 2):(x_cord + crop_size // 2),
                            (y_cord - crop_size // 2):(y_cord + crop_size // 2)]
                elif self.args.test_crop_strategy == 'prob':
                    ps = label[crop_size // 2:(height - crop_size // 2), crop_size // 2:(width - crop_size // 2)].reshape(
                        -1)
                    ps = ps / np.sum(ps)
                    point = np.random.choice((width - crop_size) * (height - crop_size), 1, p=ps)
                    y_cord = (point[0] % (width - crop_size)) + crop_size // 2
                    x_cord = (point[0] // (width - crop_size)) + crop_size // 2
                    image = image[(x_cord - crop_size // 2):(x_cord + crop_size // 2),
                            (y_cord - crop_size // 2):(y_cord + crop_size // 2)]
                    label = label[(x_cord - crop_size // 2):(x_cord + crop_size // 2),
                            (y_cord - crop_size // 2):(y_cord + crop_size // 2)]
                else:
                    raise NotImplementedError
            else:
                NotImplementedError

        if self.transform:
            image, label = self.transforms(image, label)
        sample = {'id': id, 'image': image, 'label': label}
        return sample

    def transforms(self, img, lbl):
        img = img.astype(np.float64)
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransforms(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img = img * 255.0
        img = img.astype(np.uint8)
        lbl = lbl.numpy()
        return img, lbl

    def __str__(self):
        return 'Crack500(split=' + str(self.split) + ')'

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import skimage.io
    import argparse
    import numpy as np

    import sys
    sys.path.append('../')
    import visualization

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.crop_strategy = 'rand'
    args.crop_size = 512

    train_ds = Crack500Dataset(args, split='test', transform=False)

    # if not osp.exists('./vis'):
    #     os.mkdir('./vis')
    # for idx in range(len(train_ds)):
    #     sample = train_ds[idx]
    #     id, img, lbl = sample['id'], sample['image'], sample['label']
    #     skimage.io.imsave("./vis/{}.jpg".format(id), img)
    #     skimage.io.imsave("./vis/{}_mask.png".format(id), lbl)
    #     print('{}/{} {}'.format(idx, len(train_ds), id))


    idx = 32
    sample = train_ds[idx]
    img, lbl = sample['image'], sample['label']
    print(img.shape)
    print(lbl.shape)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(lbl)
    plt.show()
    # viz = visualization.visualize_segmentation(lbl_true=lbl, img=img, n_class=len(train_ds.class_names), label_names=train_ds.class_names)
    # plt.imshow(viz)
    # plt.show()


    # # wronglist = [..., ...]
    # for idx in range(len(train_ds)):
    #     sample = train_ds[idx]
    #     id, img, lbl = sample['id'], sample['image'], sample['label']
    #
        # skimage.io.imsave("./test/{}.jpg".format(id), img)
        # skimage.io.imsave("./test/{}_mask.png".format(id), lbl)
        #
        # lbl = lbl/255
        # viz = visualization.visualize_segmentation(lbl_true=lbl, img=img, n_class=len(train_ds.class_names), label_names=train_ds.class_names)
        # skimage.io.imsave("./vis/test/{}.png".format(idx), viz)
        # print('{}/{} {}'.format(idx, len(train_ds), id))

        # skimage.io.imsave("./test/{}.jpg".format(id), img)
        # skimage.io.imsave("./test/{}_mask.png".format(id), lbl)
        # print('{}/{} {}'.format(idx, len(train_ds), id))

        # if idx in wronglist:
        #     lbl = cv2.flip(lbl, 0)
        #     lbl = cv2.flip(lbl, 1)
        #     skimage.io.imsave("./test/{}.jpg".format(id), img)
        #     skimage.io.imsave("./test/{}_mask.png".format(id), lbl)
        #     print('{}/{} {}'.format(idx, len(train_ds), id))

    # train_dl = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)


