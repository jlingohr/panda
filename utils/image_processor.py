import pandas as pd
import skimage.io
import cv2
from collections import Counter
from math import ceil


class ImageProcessor:
    def __init__(self, image_root, mask_root, classes, overlap=0.125, window_size=256):
        self.image_root = image_root
        self.mask_root = mask_root
        self.classes = classes
        self.overlap = overlap
        self.window_size = window_size
        self.step_size = int((self.window_size - (self.window_size * self.overlap)))

    def process_image(self, img_path):
        img_paths, patch_ids, x_loc, y_loc, hue, saturation, value, class_counts = self._get_locations(img_path)
        df = pd.DataFrame({'image_id': img_paths,
                           'patch_id': patch_ids,
                           'x_loc': x_loc,
                           'y_loc': y_loc,
                           'hue': hue,
                           'saturation': saturation,
                           'value': value,
                           **class_counts})
        return df

    def _get_locations(self, image_id):
        img_path = '{}/{}.png'.format(self.image_root, image_id)
        mask_path = '{}/{}_mask.png'.format(self.mask_root, image_id)

        try:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path)[:, :, -1]
            y, x = mask.shape

            x_steps = int(ceil((x - self.window_size) / self.step_size)) + 1
            y_steps = int(ceil((y - self.window_size) / self.step_size)) + 1

            x_idx = []
            y_idx = []
            hue = []
            saturation = []
            value = []
            img_paths = []
            class_counts = {k: [] for k in self.classes}
            patch_ids = []
            for ix in range(x_steps * y_steps):
                x_ix = int(min(x - self.window_size, (ix % x_steps) * self.step_size))
                y_ix = int(min(y - self.window_size, (ix // x_steps) * self.step_size))

                # Find mask statistics
                mask_data = mask[y_ix:y_ix+self.window_size, x_ix:x_ix+self.window_size]
                counts = dict(Counter(mask_data.flatten()))
                for k in self.classes:
                    class_counts[k].append(counts.get(k, 0))

                # Find image statistics
                img_data = img[y_ix:y_ix+self.window_size, x_ix:x_ix+self.window_size]
                hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
                hmu = hsv_img[:, :, 0].mean()
                smu = hsv_img[:, :, 1].mean()
                vmu = hsv_img[:, :, 2].mean()

                x_idx.append(x_ix)
                y_idx.append(y_ix)
                hue.append(hmu)
                saturation.append(smu)
                value.append(vmu)
                img_paths.append(image_id)
                patch_ids.append(ix)
        except:
            x_idx, y_idx, hue, saturation, value, patch_ids = [-1], [-1], [-1], [-1], [-1], [-1]
            class_counts = {k: [-1] for k in self.classes}
            img_paths = [image_id]
        return img_paths, patch_ids, x_idx, y_idx, hue, saturation, value, class_counts