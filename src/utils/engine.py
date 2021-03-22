import math
import sys
import time
import torch
import torchvision
import numpy as np
import pandas as pd
import src.utils.utils as utils
import src.utils.helper as helper
from src.utils.evaluate import compute_mAP


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, df):
    """
    Trains one epoch.

    Args:
        model: -
        optimizer: -
        data_loader: -
        device: -
        epoch: -
        print_freq: int. Number of iterations to wait and print info
        df: empty pandas dataframe with the following columns ->
                                ['epoch', 'iteration', 'lr', 'time',
                                'loss_avg', 'loss_median', 'loss_max', 'loss_min',
                                'loss_bb_regression_avg', 'loss_bb_regression_median',
                                'loss_bb_regression_max', 'loss_bb_regression_min',
                                'loss_classifier_avg', 'loss_classifier_median',
                                'loss_classifier_max', 'loss_classifier_min',
                                'loss_rpn_bb_regression_avg', 'loss_rpn_bb_regression_median',
                                'loss_rpn_bb_regression_max', 'loss_rpn_bb_regression_min']

    Returns:
        A MetricLogger object containing epoch information.

    """

    # Set training flag
    model.train()

    # Loggers
    metric_logger = utils.MetricLogger(delimiter=" | ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    iteration = 0

    lr_scheduler = None
    # Uses default scheduler for the first epoch
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    # Goes through the Dataloader
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # Add dataframe row with iteration metric logs
        df = helper.df_add_iteration_log(df, epoch, iteration, metric_logger)

        iteration += 1

    return metric_logger, df


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, sequences=1):
    """
    Evaluates model mAP for IoU range of [0.5:.05:0.95].

    Args:
        model: -
        data_loader: -
        device: -
        sequences: the number of sequences of images to pass, if any

    Returns:
        mAP and AP list for each IoU threshold in range [0.5:.05:0.95]
    """

    # Set evaluation mode flag
    model.eval()
    # Create list with all object detection -> [set, frame, obj, [xmin,ymin,xmax,ymax], label, score]
    ground_truth = []
    predictions = []

    # Gather all targets and outputs on test set
    for image, targets in data_loader:
        image = [img.to(device) for img in image]
        outputs = model(image)
        for idx in range(len(outputs)):
            outputs[idx] = helper.apply_nms(outputs[idx], iou_thresh=0.5)

        # create list for targets and outputs to pass to compute_mAP()
        # lists have the following structure:  [sequence, frame, obj_idx, [xmin, ymin, xmax, ymax], label, score]
        for s in range(sequences):
            obj_gt = 0
            obj_target = 0
            for out, target in zip(outputs, targets):

                for i in range(len(target['boxes'])):
                    ground_truth.append([s, target['image_id'].detach().cpu().numpy()[0], obj_target,
                                         target['boxes'].detach().cpu().numpy()[i],
                                         target['labels'].detach().cpu().numpy()[i], 1])
                    obj_target += 1

                for j in range(len(out['boxes'])):
                    predictions.append([s, target['image_id'].detach().cpu().numpy()[0], obj_gt,
                                        out['boxes'].detach().cpu().numpy()[j],
                                        out['labels'].detach().cpu().numpy()[j],
                                        out['scores'].detach().cpu().numpy()[j]])
                    obj_gt += 1

    mAP, AP = compute_mAP(ground_truth, predictions, n_classes=4)
    print("mAP:{:.3f}".format(mAP))
    for ap_metric, iou in zip(AP, np.arange(0.5, 1, 0.05)):
        print("\tAP at IoU level [{:.2f}]: {:.3f}".format(iou, ap_metric))

    return mAP, AP

