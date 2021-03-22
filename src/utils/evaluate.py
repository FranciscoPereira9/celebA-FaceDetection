import os
import numpy as np
from collections import Counter
import pandas as pd
import copy
import csv
import sys


def IOU(box1, box2):
    '''
    Intersection over Union - IoU
    *------------
    |   (x2min,y2min)
    |   *----------
    |   | ######| |
    ----|------* (x1max,y1max)
        |         |
        ----------

    Args:
        box1: [xmin,ymin,xmax,ymax]
        box2: [xmin,ymin,xmax,ymax]

    Returns:
        iou -> value of intersection over union of the 2 boxes

    '''

    # Compute coordinates of intersection
    xmin_inter = max(box1[0], box2[0])
    ymin_inter = max(box1[1], box2[1])
    xmax_inter = min(box1[2], box2[2])
    ymax_inter = min(box1[3], box2[3])

    # calculate area of intersection rectangle
    inter_area = max(0, xmax_inter - xmin_inter + 1) * max(0, ymax_inter - ymin_inter + 1) # FIXME why plus one?
 
    # calculate boxes areas
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
 
    # compute IoU
    iou = inter_area / float(area1 + area2 - inter_area)
    assert iou >= 0
    return iou


def compute_AP(ground_truth, predictions, iou_thresh=0.5, n_classes=4):
    """
    Calculates Average Precision across all classes.

    Args:
        ground_truth: list with ground-truth objects. Needs to have the following format: [sequence, frame, obj, [xmin, ymin, xmax, ymax], label, score]
        predictions: list with predictions objects. Needs to have the following format: [sequence, frame, obj, [xmin, ymin, xmax, ymax], label, score]
        iou_thresh: IoU to which a prediction compared to a ground-truth is considered right.
        n_classes: number of existent classes

    Returns:
        Average precision for the specified threshold.
    """
    # Initialize lists
    APs = []
    class_gt = []
    class_predictions = []

    # AP is computed for each class
    for c in range(n_classes):
        # Find gt and predictions of the class
        for gt in ground_truth:
            if gt[4] == c:
                class_gt.append(gt)
        for predict in predictions:
            if predict[4] == c:
                class_predictions.append(predict)

        # Create dict with array of zeros for bb in each image
        gt_amount_bb = Counter([gt[1] for gt in class_gt])
        for key, val in gt_amount_bb.items():
            gt_amount_bb[key] = np.zeros(val)

        # Sort class predictions by their score
        class_predictions = sorted(class_predictions, key=lambda x: x[5], reverse=True)

        # Create arrays for Positives (True and False)
        TP = np.zeros(len(class_predictions))
        FP = np.zeros(len(class_predictions))
        # Number of true boxes
        truth = len(class_gt)

        # Initializing aux variables
        epsilon = 1e-6

        # Iterate over predictions in each image and compare with ground truth
        for predict_idx, prediction in enumerate(class_predictions):
            # Filter prediction image ground truths
            image_gt = [obj for obj in class_gt if obj[1] == prediction[1]]

            # Initializing aux variables
            best_iou = -1
            best_gt_iou_idx = -1

            # Iterate through image ground truths and calculate IoUs
            for gt_idx, gt in enumerate(image_gt):
                iou = IOU(prediction[3], gt[3])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_iou_idx = gt_idx

            # If the best IoU is greater that thresh than an TP prediction has been found
            if best_iou > iou_thresh and best_gt_iou_idx > -1:
                # Check if gt box was already covered
                if  gt_amount_bb[prediction[1]][best_gt_iou_idx] == 0:
                    gt_amount_bb[prediction[1]][best_gt_iou_idx] = 1  # set as covered
                    TP[predict_idx] = 1  # Count as true positive
                else:
                    FP[predict_idx] = 1
            else:
                FP[predict_idx] = 1

        # Calculate recall and precision
        TP_cumsum = np.cumsum(TP)
        FP_cumsum = np.cumsum(FP)
        recall = np.append([0], TP_cumsum / (truth + epsilon))
        precision = np.append([1], np.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon)))

        # Calculate the area precision/recall and add to list
        APs.append(np.trapz(precision, recall))

    return sum(APs)/len(APs)  # average of class precisions


def compute_mAP(ground_truth, predictions, n_classes):
    """
    Calls AP computation for different levels of IoUs, [0.5:.05:0.95].

    Args:
        ground_truth: list with ground-truth objects. Needs to have the following format: [sequence, frame, obj, [xmin, ymin, xmax, ymax], label, score]
        predictions: list with predictions objects. Needs to have the following format: [sequence, frame, obj, [xmin, ymin, xmax, ymax], label, score]
        n_classes: number of existent classes.

    Returns:
        mAp and list with APs for each IoU threshold.
    """
    # return mAP
    APs = [compute_AP(ground_truth, predictions, iou_thresh, n_classes) for iou_thresh in np.arange(0.5, 1.0, 0.05)]
    return np.mean(APs), APs


def compute_mAP_from_files(preds_file, labels_file):
    labels = list()
    # convert labels to the right format (dict of lists) [seq, frame] -> [box1, box2, ...]
    with open(labels_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for i, row in enumerate(reader):
            if i == 0 or len(row) == 2:
                continue
            labels.append([int(row[0]), int(row[1]), eval(row[2])])

    preds = list()
    # conver predictions to the right format (list) [seq, frame, [x, y, x+w, y+h], score]
    with open(preds_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for i, row in enumerate(reader):
            if row == "seq;frame;label;score".split(";"):
                continue
            preds.append([int(row[0]), int(row[1]), eval(row[2]), float(row[3])])
    return compute_mAP(preds, labels)


if __name__=="__main__":
    mAP, AP = compute_mAP_from_files("predictions.csv", "/home/master/dataset/test/labels.csv")
    print("mAP:{:.4f}".format(mAP))
    for ap_metric, iou in zip(AP, np.arange(0.5, 1, 0.05)):
        print("\tAP at IoU level [{:.2f}]: {:.4f}".format(iou, ap_metric))
