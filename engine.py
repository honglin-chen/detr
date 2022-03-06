# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import pdb
import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util.misc import NestedTensor
from util.miou_metric import measure_miou_metric
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, dataset: str, max_norm: float = 0, ):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, dataset):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = None # CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    miou_stats = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)

        processed_targets = []
        for t in targets:
            item = {}
            for k, v in t.items():
                item[k] = v if isinstance(v, str) else v.to(device)
            processed_targets.append(item)
        targets = processed_targets

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)


        '''
        
        # Visualization
        threshold = 0.85

        plt.figure(figsize=(10, 10))


        for idx in range(9):
            plt.subplot(3, 3, idx+1)
            scores = results[idx]['scores']
            mask_select = scores > threshold

            plt.imshow(targets[idx]['raw_images'][0].permute(1, 2, 0).cpu())
            pred_bboxes = results[idx]['boxes'][mask_select]
            tgt_bboxes = targets[idx]['raw_boxes']

            ax = plt.gca()
            for j in range(pred_bboxes.shape[0]):
                plt_boxes(ax, pred_bboxes[j].cpu(), 'b')

            for j in range(tgt_bboxes.shape[0]):
                plt_boxes(ax, tgt_bboxes[j].cpu(), 'r')
            plt.axis('off')
        plt.show()
        plt.close()
        '''

        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)


        
        # pdb.set_trace()
        def reorder_int_labels(x):
            _, y = torch.unique(x, return_inverse=True)
            y -= y.min()
            return y
        #
        #
        threshold = 0.

        for idx in range(len(results)):

            scores = results[idx]['scores']
            masks = results[idx]['masks']
            # mask_select = scores > threshold
            # masks = masks[mask_select]

            masks = masks.cuda() * scores[:, None, None, None]
            masks = torch.cat([torch.zeros_like(masks[0:1]), masks], dim=0)

            pred_segments = masks.argmax(0)
            gt_segments = targets[idx]['segment_map']
            pred_segments = reorder_int_labels(pred_segments)

            # pdb.set_trace()
            # print('Number of unique segments', len(pred_segments.unique()))
            # plt.subplot(1, 3, 1)
            #
            # plt.imshow(targets[idx]['raw_images'][0].permute(1, 2, 0).cpu())
            # plt.title('Image')
            # plt.axis('off')
            # plt.subplot(1, 3, 2)
            # plt.imshow(gt_segments.cpu())
            # plt.title('GT segments')
            # plt.axis('off')
            # plt.subplot(1, 3, 3)
            # plt.imshow(pred_segments[0].cpu())
            # plt.title('Predicted segments')
            # plt.axis('off')
            # plt.show()
            # plt.close()

            miou, vis = measure_miou_metric(pred_segment=pred_segments.int(), gt_segment=gt_segments.int().unsqueeze(-1))
            miou_stats.append(miou)


            pred, gt, iou = vis

            # plt.subplot(1, 2, 1)
            # plt.imshow(pred_segments.reshape(512, 512).cpu())
            # plt.subplot(1, 2, 2)
            # plt.imshow(gt_segments.cpu())
            # plt.show()
            # plt.close()


            # save_out = {
            #     'image': targets[idx]['raw_images'][0],
            #     'pred_segments': pred[0],
            #     'gt_segments': gt[0],
            # }
            #
            # file_name = targets[idx]['file_name'].split('/data2/honglinc/tdw_playroom_small/images/')[-1].replace('/', '-')
            # save_path = os.path.join('./output/TDW_Cylinder_DETR_RAFT', file_name+'.pt')
            # print('Save to', save_path)
            # torch.save(save_out, save_path)

        print('mIoU: ', np.mean(miou_stats), len(miou_stats), len(pred[0]))

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    print('Final mIoU: ', np.mean(miou_stats))
    # pdb.set_trace()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator


def plt_boxes(ax, box, color='r'):
    x0, y0, x1, y1 = box.unbind(-1)
    width = x1 - x0
    height = y1 - y0
    rect = patches.Rectangle((x0, y0), width, height, linewidth=1, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


