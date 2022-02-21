import os
import json
import glob
import pdb
import pandas as pd
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
from util.box_ops import *


class TDWDataset(Dataset):
    def __init__(self, dataset_dir, training, delta_time=1, frame_idx=5):
        self.training = training
        self.frame_idx = frame_idx
        self.delta_time = delta_time

        meta_path = os.path.join(dataset_dir, 'meta.json')
        self.meta = json.loads(Path(meta_path).open().read())

        if self.training:
            self.file_list = glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-9]*', '*[0-8]'))
        else:
            self.file_list = glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-3]', '*9'))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        image_1 = self.read_frame(file_name, frame_idx=self.frame_idx)
        image_2 = self.read_frame(file_name, frame_idx=self.frame_idx+self.delta_time)
        images = torch.cat([image_1[None], image_2[None]], 0)
        segment_colors = self.read_frame(file_name.replace('/images/', '/objects/'), frame_idx=self.frame_idx)
        _, segment_map, gt_moving = self.process_segmentation_color(segment_colors, file_name)
        gt_moving = gt_moving.unsqueeze(0)

        h, w = image_1.shape[-2:]
        boxes = masks_to_boxes(gt_moving)
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        targets = {
            'mask': gt_moving,
            'boxes': boxes,
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'image_id': torch.as_tensor(idx),
            'labels': torch.as_tensor([1])
        }
        return images, targets

    @staticmethod
    def read_frame(path, frame_idx):
        image_path = os.path.join(path, format(frame_idx, '05d') + '.png')
        return read_image(image_path)

    @staticmethod
    def _object_id_hash(objects, val=256, dtype=torch.long):
        C = objects.shape[0]
        objects = objects.to(dtype)
        out = torch.zeros_like(objects[0:1, ...])
        for c in range(C):
            scale = val ** (C - 1 - c)
            out += scale * objects[c:c + 1, ...]
        return out

    def process_segmentation_color(self, seg_color, file_name):
        # convert segmentation color to integer segment id
        raw_segment_map = self._object_id_hash(seg_color, val=256, dtype=torch.long)
        raw_segment_map = raw_segment_map.squeeze(0)

        # remove zone id from the raw_segment_map
        meta_key = 'playroom_large_v3_images/' + file_name.split('/images/')[-1] + '.hdf5'
        zone_id = int(self.meta[meta_key]['zone'])
        raw_segment_map[raw_segment_map == zone_id] = 0

        # convert raw segment ids to a range in [0, n]
        _, segment_map = torch.unique(raw_segment_map, return_inverse=True)
        segment_map -= segment_map.min()

        # gt_moving_mask
        gt_moving = raw_segment_map == int(self.meta[meta_key]['moving'])

        return raw_segment_map, segment_map, gt_moving


def build_tdw_dataset(image_set, dataset_dir='/data2/honglinc/tdw_playroom_small'):
    if image_set == 'train':
        return TDWDataset(dataset_dir, training=True)
    elif image_set == 'val':
        return TDWDataset(dataset_dir, training=False)
    else:
        raise NotValueError


if __name__ == "__main__":
    dataset_dir = '/data2/honglinc/tdw_playroom_small'
    batch_size = 5

    train_dataset = TDWDataset(dataset_dir, training=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = TDWDataset(dataset_dir, training=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    image_1, image_2, segment_map, gt_moving = next(iter(train_dataloader))
    print(image_1.shape, image_2.shape, segment_map.shape)

    # Visualization
    for idx in range(batch_size):
        fig = plt.figure(figsize=(12, 3))
        plt.subplot(1, 4, 1)
        plt.imshow(image_1[idx].permute(1, 2, 0))
        plt.title('Frame 1')
        plt.subplot(1, 4, 2)
        plt.imshow(image_2[idx].permute(1, 2, 0))
        plt.title('Frame 2')
        plt.subplot(1, 4, 3)
        plt.imshow(segment_map[idx])
        plt.title('Segment')
        plt.subplot(1, 4, 4)
        plt.imshow(gt_moving[idx])
        plt.title('Moving')
        fig.tight_layout()
        plt.show()
        plt.close()
