import argparse
import multiprocessing as mp
import os

import cv2
import numpy as np
import pandas as pd
import skimage.io
import tifffile
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--csv', type=str, required=True)
parser.add_argument('--dest', type=str, required=True)
parser.add_argument('--level', type=int, choices=[0,1,2], default=1)

args = parser.parse_args()


def crop_white(image: np.ndarray, value: int = 255) -> np.ndarray:
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) < value).nonzero()
    xs, = (image.min(0).min(1) < value).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def crop_white_with_mask(image: np.ndarray, mask: np.ndarray, value: int = 255) -> (np.ndarray, np.ndarray):
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    ys, = (image.min((1, 2)) < value).nonzero()
    xs, = (image.min(0).min(1) < value).nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return image, mask
    return image[ys.min():ys.max() + 1, xs.min():xs.max() + 1], mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def to_png(image_id: str):
    img_path = os.path.join(args.data_dir, 'train_images', '{}.tiff'.format(image_id))
    img_save_path = os.path.join(args.dest, 'images', '{}.png'.format(image_id))
    image = skimage.io.MultiImage(img_path)[args.level]

    mask_path = os.path.join(args.data_dir, 'train_label_masks', '{}_mask.tiff'.format(image_id))
    if os.path.exists(mask_path):
        mask = skimage.io.MultiImage(mask_path)[args.level]
        mask_save_path = os.path.join(args.dest, 'masks', '{}_mask.png'.format(image_id))
        image, mask = crop_white_with_mask(image, mask)
        cv2.imwrite(img_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(mask_save_path, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
        return 1
    else:
        image = crop_white(image)
        cv2.imwrite(img_save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        return 0


if __name__ == '__main__':
    """
    Trim whitespace from images and corresponding masks to reduce memory footprint and improve tiling
    process later on.
    """
    csv_path = os.path.join(args.data_dir, args.csv)
    train = pd.read_csv(csv_path)

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)
        os.mkdir(os.path.join(args.dest, 'images'))
        os.mkdir(os.path.join(args.dest, 'masks'))

    with mp.Pool() as pool:
        for _ in tqdm(pool.imap(to_png, train.image_id), total=len(train), leave=False):
            pass

