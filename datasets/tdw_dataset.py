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
import torchvision.transforms as T
from PIL import Image


class TDWDataset(Dataset):
    def __init__(self, dataset_dir, training, supervision, delta_time=1, frame_idx=5):
        self.training = training
        self.frame_idx = frame_idx
        self.delta_time = delta_time
        self.supervision = supervision

        meta_path = os.path.join(dataset_dir, 'meta.json')
        self.meta = json.loads(Path(meta_path).open().read())

        if self.training:
            self.file_list = glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-9]*', '*[0-8]'))
        else:
            self.file_list = glob.glob(os.path.join(dataset_dir, 'images', 'model_split_[0-3]', '*9'))

        self.transform = T.Compose([
            T.ToTensor(),  # divided by 255.
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        raw_image_1, image_1 = self.read_frame(file_name, frame_idx=self.frame_idx, transform=self.transform)

        try:
            raw_image_2, image_2 = self.read_frame(file_name, frame_idx=self.frame_idx+self.delta_time, transform=self.transform)
        except Exception as e:
            image_2 = image_1.clone()
            raw_image_1 = raw_image_1.clone()
            print('Encounter error: ', e)

        images = torch.cat([image_1[None], image_2[None]], 0)
        raw_images =  torch.cat([raw_image_1[None], raw_image_2[None]], 0)
        segment_colors = self.read_frame(file_name.replace('/images/', '/objects/'), frame_idx=self.frame_idx)
        _, segment_map, gt_moving = self.process_segmentation_color(segment_colors, file_name)
        gt_moving = gt_moving.unsqueeze(0)

        h, w = image_1.shape[-2:]

        if self.supervision == 'sinobj':
            mask = gt_moving
            labels = torch.as_tensor([0]).long()
            if gt_moving.sum() == 0:
                return None

        elif self.supervision == 'allobj':
            unique = segment_map.unique()
            unique = unique[unique > 0]
            if len(unique) > 6: # invalid image file with more than 5 objects
                print('Having more than 6 objects')
                plt.imshow(segment_map)
                plt.savefig('%d.png' % sum(unique))
                plt.close()
                return None
            mask = unique[:, None, None] == segment_map
            labels = torch.as_tensor([0] * mask.shape[0]).long()
        else:
            raise ValueError

        # create bboxes from masks [N, H, W]
        boxes = masks_to_boxes(mask)
        raw_boxes = boxes.clone()
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)

        targets = {
            'mask': mask,
            'boxes': boxes,
            'raw_boxes': raw_boxes,
            'orig_size': torch.as_tensor([int(h), int(w)]),
            'image_id':  torch.as_tensor([idx]),
            'labels': labels,
            'raw_images': raw_images
        }
        return images, targets

    @staticmethod
    def read_frame(path, frame_idx, transform=None):
        image_path = os.path.join(path, format(frame_idx, '05d') + '.png')
        if transform is None:
            return read_image(image_path)
        else:
            raw_image = Image.open(image_path)
            return T.ToTensor()(raw_image), transform(raw_image)

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


def build_tdw_dataset(image_set, supervision, dataset_dir='/data2/honglinc/tdw_playroom_small'):
    if image_set == 'train':
        return TDWDataset(dataset_dir, training=True, supervision=supervision)
    elif image_set == 'val':
        return TDWDataset(dataset_dir, training=False, supervision=supervision)
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
