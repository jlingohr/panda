import cv2

import torch
from torch.utils.data import Dataset


class PatchesDataset(Dataset):
    def __init__(self, data_root, df, crop_df, transforms, window_size=256):
        self.data_root = data_root
        self.df = df
        self.crops = crop_df

        has_enough_crops = self.crops.groupby(['image_id']).apply(lambda x: len(x) == 16).reset_index() #TODO
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

        patches = [cv2.imread('{}/images/{}_{}.png'.format(self.data_root, image_id, idx)) for idx in range(len(crops))]
        patches = [self.transforms(patch) for patch in patches] #TODO are these proper channel order w/ cv2?
        patches = torch.stack(patches)

        label = row['isup_grade']

        sample = {
            'filename': image_id,
            'image': patches,
            'label': label
        }

        return sample