import argparse
import multiprocessing as mp
import os

import cv2
import numpy as np
import pandas as pd
import skimage.io
from tqdm import tqdm

from image_processor import ImageProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--outfile', type=str, required=True)
parser.add_argument('--csv', type=str, required=True)
parser.add_argument('--overlap', type=float, default=0.125)
parser.add_argument('--window-size', type=int, default=256)


args = parser.parse_args()
CLASSES = [0, 1, 2, 3, 4, 5]


def process_images(params):
    img_path = params
    processor = ImageProcessor(os.path.join(args.data_dir, 'images'),
                               os.path.join(args.data_dir, 'masks'),
                               CLASSES,
                               overlap=args.overlap,
                               window_size=args.window_size)
    return processor.process_image(img_path)


if __name__ == '__main__':
    """
    Compute number of pixels per class using weakly labeled masks and also find
    RGB values for each patch
    """
    csv_path = os.path.join(args.data_dir, args.csv)
    train = pd.read_csv(csv_path)

    with mp.Pool(mp.cpu_count()) as pool:
        dfs = list(tqdm(pool.imap(process_images, train.image_id.to_list()),
                        total=len(train)))
    dfs = pd.concat(dfs)

    dfs.to_csv(args.outfile, index=False)

