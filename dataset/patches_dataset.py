import cv2
import numpy as np

import torch
from torch.utils.data import Dataset


class PatchesDataset(Dataset):
    def __init__(self, data_root, df, crop_df, transforms, window_size=256, num_samples=12, max_patches=12):
        assert num_samples <= max_patches, "Number of samples should be less than max patches"
        self.data_root = data_root
        self.df = df

        self.num_samples = num_samples

        # Retain only images patches with highest saturation
        crop_df = crop_df.groupby(['image_id']).apply(
            lambda x: x.sort_values(by=['saturation'],
                                    ascending=False)[:max_patches]).reset_index(drop=True)

        self.crops = crop_df
        has_enough_crops = self.crops.groupby(['image_id']).apply(lambda x: len(x) >= self.num_samples).reset_index() #TODO
        has_enough_crops = has_enough_crops[has_enough_crops.iloc[:, 1] == True]
        self.df = self.df[self.df.image_id.isin(has_enough_crops.image_id)]

        self.transforms = transforms
        self.window_size = window_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image_id']
        crops = self.crops[self.crops.image_id == image_id] #TODO properly randomly sample or something?
        patch_ids = crops.patch_id.tolist()
        patch_ids = np.random.choice(patch_ids, size=self.num_samples, replace=False)

        patches = [cv2.imread('{}/images/{}_{}.png'.format(self.data_root, image_id, idx)) for idx in patch_ids]
        patches = [self.transforms(image=patch) for patch in patches]
        patches = [patch['image'] for patch in patches]
        patches = torch.stack(patches)

        label = row['isup_grade']

        sample = {
            'filename': image_id,
            'image': patches,
            'label': label
        }

        return sample