from fastai.vision.all import *
import pandas as pd
import numpy as np

input_path = './datasets/data/v2/segmentation/images'
images = Path(input_path).ls()
mask_path = './datasets/data/v2/segmentation/train'

masks = pd.read_csv(Path('../input/airbus-ship-detection/train_ship_segmentations_v2.csv'), index_col='ImageId').EncodedPixels
masks[:3]

sample = [image for image in images if isinstance(masks[image.name], str)]
len(sample)

def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def label_func(fn):
    rle = masks[fn.name]
    if not isinstance(rle, str):
        return np.zeros((768, 768), dtype=np.uint8)
    return rle_decode(rle)

show_image(label_func(sample[1]))
dls = SegmentationDataLoaders.from_label_func(
    input_path,
    bs=8,
    fnames=sample,
    label_func=label_func,
    item_tfms=RandomResizedCrop(224, min_scale=0.3),
    batch_tfms=aug_transforms()
)

dls.show_batch()


learn = unet_learner(dls, resnet34, n_out=2, metrics=Dice())


learn.fine_tune(8)


learn.show_results(dl=dls.valid)


import matplotlib.pyplot as plt
import cv2


def predict(path):
    img = cv2.imread(path)
    predicted_masks = learn.predict(img)
    mask = cv2.resize(predicted_masks[1].numpy().astype('uint8'), (768, 768))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    plt.axis('off')
    ax1.imshow(img)
    ax1.imshow(mask, cmap='jet', alpha=0.3)
    ax2.imshow(img)
    plt.show()


predict(str(sample[2]))


predict(str(sample[3]))


predict(str(sample[4]))


