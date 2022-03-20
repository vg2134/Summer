import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from scipy.interpolate import griddata

from utils import string_utils, safe_load, augmentation

import random

INTERPOLATION = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC
}


def warp_image(img, random_state=None, **kwargs):
    if random_state is None:
        random_state = np.random.RandomState()

    w_mesh_interval = kwargs.get('w_mesh_interval', 12)
    w_mesh_std = kwargs.get('w_mesh_std', 1.5)

    h_mesh_interval = kwargs.get('h_mesh_interval', 12)
    h_mesh_std = kwargs.get('h_mesh_std', 1.5)

    interpolation_method = kwargs.get('interpolation', 'linear')

    h, w = img.shape[:2]

    if kwargs.get("fit_interval_to_image", True):
        w_ratio = w / float(w_mesh_interval)
        h_ratio = h / float(h_mesh_interval)

        w_ratio = max(1, round(w_ratio))
        h_ratio = max(1, round(h_ratio))

        w_mesh_interval = w / w_ratio
        h_mesh_interval = h / h_ratio

    source = np.mgrid[0:h + h_mesh_interval:h_mesh_interval, 0:w + w_mesh_interval:w_mesh_interval]
    source = source.transpose(1, 2, 0).reshape(-1, 2)

    if kwargs.get("draw_grid_lines", False):
        if len(img.shape) == 2:
            color = 0
        else:
            color = np.array([0, 0, 255])
        for s in source:
            img[int(s[0]):int(s[0]) + 1, :] = color
            img[:, int(s[1]):int(s[1]) + 1] = color

    destination = source.copy()
    source_shape = source.shape[:1]
    destination[:, 0] = destination[:, 0] + random_state.normal(0.0, h_mesh_std, size=source_shape)
    destination[:, 1] = destination[:, 1] + random_state.normal(0.0, w_mesh_std, size=source_shape)

    grid_x, grid_y = np.mgrid[0:h, 0:w]
    grid_z = griddata(destination, source, (grid_x, grid_y), method=interpolation_method).astype(np.float32)
    map_x = grid_z[:, :, 1]
    map_y = grid_z[:, :, 0]
    warped = cv2.remap(img, map_x, map_y, INTERPOLATION[interpolation_method], borderValue=(255, 255, 255))

    return warped


PADDING_CONSTANT = 0


def collate(batch):
    batch = [b for b in batch if b is not None]
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0]
    dim1 = max([b['line_img'].shape[1] for b in batch])
    dim2 = batch[0]['line_img'].shape[2]

    all_labels = []
    label_lengths = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i, :, :b_img.shape[1], :] = b_img

        l = batch[i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))

    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0, 3, 1, 2])
    line_imgs = torch.from_numpy(line_imgs)
    labels = torch.from_numpy(all_labels.astype(np.int32))
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    return {
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch]
    }


class HwDataset(Dataset):
    def __init__(self, set_list, char_to_idx, augmentation=False, img_height=32, random_subset_size=None):

        self.img_height = img_height

        self.ids = set_list
        self.ids.sort()

        self.detailed_ids = []
        for ids_idx, paths in enumerate(self.ids):
            json_path, img_path = paths
            d = safe_load.json_state(json_path)
            if d is None:
                continue
            for i in range(len(d)):

                if 'hw_path' not in d[i]:
                    continue
                self.detailed_ids.append((ids_idx, i))

        if random_subset_size is not None:
            self.detailed_ids = random.sample(self.detailed_ids, min(random_subset_size, len(self.detailed_ids)))

        self.char_to_idx = char_to_idx
        self.augmentation = augmentation
        self.warning = False

    def __len__(self):
        return len(self.detailed_ids)

    def __getitem__(self, idx):
        ids_idx, line_idx = self.detailed_ids[idx]
        gt_json_path, img_path = self.ids[ids_idx]
        gt_json = safe_load.json_state(gt_json_path)
        if gt_json is None:
            return None

        if 'hw_path' not in gt_json[line_idx]:
            return None

        hw_path = gt_json[line_idx]['hw_path']

        hw_path = hw_path.split("/")[-1:]
        hw_path = "/".join(hw_path)

        hw_folder = os.path.dirname(gt_json_path)

        img = cv2.imread(os.path.join(hw_folder, hw_path))

        if img is None:
            return None

        if img.shape[0] != self.img_height:
            if img.shape[0] < self.img_height and not self.warning:
                self.warning = True
                print("WARNING: upsampling image to fit size")
            percent = float(self.img_height) / img.shape[0]
            img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)

        if img is None:
            return None

        if self.augmentation:
            img = augmentation.apply_random_color_rotation(img)
            img = augmentation.apply_tensmeyer_brightness(img)
            img = warp_image(img)

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0

        if len(gt) == 0:
            return None
        gt_label = string_utils.str2label_single(gt, self.char_to_idx)

        return {
            "line_img": img,
            "gt": gt,
            "gt_label": gt_label
        }
