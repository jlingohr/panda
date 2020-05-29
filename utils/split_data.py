import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, required=True)
parser.add_argument('--splits', type=int, default=5)
args = parser.parse_args()

if __name__ == '__main__':
    train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    # Treat negative and 0+0 as the same
    train.loc[train.gleason_score == 'negative', 'gleason_score'] = '0+0'

    # Remove images without masks
    to_remove = pd.read_csv(os.path.join(args.data_dir, 'suspicious_test_cases.csv'))
    cleaned = train[~train.image_id.isin(to_remove.image_id)]

    # Create Dev/Eval sets
    dev, eval = train_test_split(cleaned, test_size=0.2, random_state=42, shuffle=True, stratify=cleaned.isup_grade)
    dev.to_csv(os.path.join(args.data_dir, 'dev.csv'), index=False)
    eval.to_csv(os.path.join(args.data_dir, 'eval.csv'), index=False)

    # Create CV on dev
    kf = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=42)
    for idx, (train_index, test_index) in enumerate(kf.split(dev, dev.isup_grade)):
        X_train, X_test = train.iloc[train_index, :], train.iloc[test_index, :]

        train_idx = X_train.index.to_numpy()
        val_idx = X_test.index.to_numpy()

        np.save(os.path.join(args.data_dir, 'train_{}.npy'.format(idx)), train_idx)
        np.save(os.path.join(args.data_dir, 'val_{}.npy'.format(idx)), val_idx)


