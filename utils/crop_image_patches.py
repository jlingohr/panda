import argparse
import multiprocessing as mp
import os
import zipfile

import cv2
import numpy as np
import pandas as pd
import skimage.io
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--outfile', type=str, required=True)
parser.add_argument('--train-csv', type=str, required=True)
parser.add_argument('--patch-csv', type=str, required=True)
parser.add_argument('--window-size', type=int, default=256)

args = parser.parse_args()


if __name__ == '__main__':
    """
    Crop image patches from full image and filter out all white patches.
    All-white patches are assumed to be patches where the background mask value accounts for
    all the pixels in the patch.
    """
    train = pd.read_csv(args.train_csv)
    patches = pd.read_csv(args.patch_csv)

    df = pd.merge(train, patches)

    # Filter out all-white patches
    df = df[df.iloc[:, -5:].sum(axis=1) > 0]
    df.to_csv(args.outfile, index=False)

    filepaths = df.image_id.unique()
    window_size = args.window_size

    # Crop patches and save
    with zipfile.ZipFile('{}.zip'.format(args.outfile.replace('.csv', '')), 'w') as img_out:
        for image_id in tqdm(filepaths, total=len(filepaths)):
            crop_locations = df[df.image_id == image_id]
            locations = zip(crop_locations.x_loc, crop_locations.y_loc)

            path = os.path.join(args.data_dir, 'images', '{}.png'.format(image_id))
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            patches = [img[y:y + window_size, x:x + window_size] for (x, y) in locations]
            patches = [cv2.imencode('.png', patch)[1] for patch in patches]

            for idx, patch in enumerate(patches):
                img_out.writestr('{}_{}.png'.format(image_id, idx), patch)

