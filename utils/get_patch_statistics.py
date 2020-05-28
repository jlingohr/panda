import argparse
import multiprocessing as mp
import os

import cv2
import pandas as pd
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--outfile', type=str, required=True)
parser.add_argument('--csv', type=str, required=True)

args = parser.parse_args()


def get_statistics(params):
    image_id, patch_id = params
    img_path = os.path.join(args.data_dir, '{}_{}.png'.format(image_id, patch_id))
    img = cv2.imread(img_path)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hmu = hsv_img[:, :, 0].mean()
    smu = hsv_img[:, :, 1].mean()
    vmu = hsv_img[:, :, 2].mean()
    df = pd.DataFrame({'image_id': [image_id],
                       'patch_id': [patch_id],
                       'hue': [hmu],
                       'saturation': [smu],
                       'value': [vmu]})
    return df


if __name__ == '__main__':
    """
    Find HSV statistics for each image patch
    """
    patches_df = pd.read_csv(args.csv)

    with mp.Pool(mp.cpu_count()) as pool:
        rows = list(tqdm(
            pool.imap(get_statistics, zip(patches_df.image_id, patches_df.patch_id)),
            total=len(patches_df)))
    rows = pd.concat(rows)
    rows.to_csv(args.outfile, index=False)

