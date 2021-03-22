# Torch
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn
# Image Plots
from PIL import Image, ImageDraw, ExifTags, ImageColor, ImageFont
# Data management
import numpy as np
import time


def draw_bounding_boxes(img_tensor, target=None, prediction=None):
    """Draws bounding boxes in given images. Displays them

        Inputs:
          img:
            Image in tensor format.
          target:
            target dictionary containing bboxes list wit format -> [xmin, ymin, xmax, ymax]

        Returns:
          None
        """

    img = torchvision.transforms.ToPILImage()(img_tensor)

    # fetching the dimensions
    wid, hgt = img.size
    print(str(wid) + "x" + str(hgt))

    # Img to draw in
    draw = ImageDraw.Draw(img)

    if target:
        target_bboxes = target['boxes'].numpy().tolist()
        target_labels = decode_labels(target['labels'].numpy())

        for i in range(len(target_bboxes)):
            # Create Rectangle patches and add the patches to the axes
            draw.rectangle(target_bboxes[i], fill=None, outline='green', width=1)
            draw.text(target_bboxes[i][:2], target_labels[i], fill='green', font=None, anchor=None, spacing=4,
                      align='left', direction=None, features=None, language=None, stroke_width=0, stroke_fill=None,
                      embedded_color=False)

    if prediction:
        prediction_bboxes = prediction['boxes'].detach().cpu().numpy().tolist()
        prediction_labels = decode_labels(prediction['labels'].detach().cpu().numpy())
        for i in range(len(prediction_bboxes)):
            # Create Rectangle patches and add the patches to the axes
            draw.rectangle(prediction_bboxes[i], fill=None, outline='red', width=1)
            draw.text(prediction_bboxes[i][:2], prediction_labels[i], fill='red', font=None, anchor=None, spacing=4,
                      align='left', direction=None, features=None, language=None, stroke_width=0, stroke_fill=None,
                      embedded_color=False)

    img.show()


def encoded_labels(lst_labels):
    """Encodes label classes from string to integers.

        Labels are encoded accordingly:
            - background => 0
            - face => 1

            Args:
              lst_labels:
                A list with classes in string format (e.g. ['face', 'background'...]).

            Returns:
              encoded:
                A list with integers that represent each class.
            """

    encoded = []
    for label in lst_labels:
        if label == "face":
            code = 1
        else:
            code = 0
        encoded.append(code)
    return encoded


def decode_labels(lst_labels):
    """
    Decode label classes from integers to strings.
    Labels are encoded accordingly:
        - background => 0
        - face => 1

    Args:
      lst_labels:
        A list with classes in integer format (e.g. [1, 0, ...]).

    Returns:
        A list with strings that represent each class.
    """

    labels = []
    for code in lst_labels:
        if code == 1:
            label = "face"
        else:
            label = 'background'
        labels.append(label)
    return labels


def build_model(nclasses):
    """
    Builds model. Uses Faster R-CNN pre-trained on COCO dataset.

    Args:
      nclasses:
        number of classes

    Return:
      model: Faster R-CNN pre-trained model
    """
    # load pre-trained model on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, nclasses)

    return model


def train_model(model, loader, optimizer, scheduler, epochs, device):
    """
    Args:
        model: -
        loader: -
        optimizer: -
        scheduler: -
        epochs: -
        device: -

    Returns:
        model: -
        loss_list: list with mean loss per epoch. Epoch 1 is in idx 0.
    """
    # Create a loss list to keep epoch average loss
    loss_list = []
    # Epochs
    for epoch in range(epochs):
        print('Starting epoch...... {}/{} '.format(epoch + 1, epochs))
        iteration = 0
        loss_sub_list = []
        start = time.time()
        for images, targets in loader:
            # Agregate images in batch loader
            images = list(image.to(device) for image in images)

            # Agregate targets in batch loader
            targets = [{key: val.to(device) for key, val in target.items()} for target in targets]

            # Sets model to train mode (just a flag)
            model.train()

            # Output of model returns loss and detections
            optimizer.zero_grad()
            output = model(images, targets)

            # Calculate Cost
            losses = sum(loss for loss in output.values())
            loss_value = losses.item()
            loss_sub_list.append(loss_value)
            print('')

            # Update optimizer and learning rate
            losses.backward()
            optimizer.step()
            iteration += 1
            print('Iteration: {:d} --> Loss: {:.3f}'.format(iteration, loss_value))

        end = time.time()
        # update scheduler
        scheduler.step()
        # print the loss of epoch
        epoch_loss = np.mean(loss_sub_list)
        loss_list.append(epoch_loss)
        print('Epoch loss: {:.3f} , time used: ({:.1f}s)'.format(epoch_loss, end - start))

    return model, loss_list


def apply_nms(orig_prediction, iou_thresh):
    """
    Applies non max supression and eliminates low score bounding boxes.

      Args:
        orig_prediction: the model output. A dictionary containing element scores and boxes.
        iou_thresh: Intersection over Union threshold. Every bbox prediction with an IoU greater than this value
                      gets deleted in NMS.

      Returns:
        final_prediction: Resulting prediction
    """

    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    # Keep indices from nms
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def remove_low_score_bb(orig_prediction, score_thresh):
    """
    Eliminates low score bounding boxes.

    Args:
        orig_prediction: the model output. A dictionary containing element scores and boxes.
        score_thresh: Boxes with a lower confidence score than this value get deleted

    Returns:
        final_prediction: Resulting prediction
    """

    # Remove low confidence scores according to given threshold
    index_list_scores = []
    scores = orig_prediction['scores'].detach().cpu().numpy()
    for i in range(len(scores)):
        if scores[i] > score_thresh:
            index_list_scores.append(i)
    keep = torch.tensor(index_list_scores)

    # Keep indices from high score bb
    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction


def collate_fn(batch):
    # Collate function for Dataloader
    return tuple(zip(*batch))


def df_add_epoch_log(df, epoch, mAP, training_results):
    """
    Adds row to the epochs logs dataframe.

    Args:
        df: -
        epoch: -
        mAP: -
        training_results: MetricLogger object with information from training epoch.

    Returns:
        df: -

    """
    df = df.append({'epoch': epoch, 'lr': training_results.meters['lr'], 'time': 0, 'mAP': mAP,
                    'loss_avg': training_results.meters['loss'].avg,
                    'loss_median': training_results.meters['loss'].median,
                    'loss_max': training_results.meters['loss'].max,
                    'loss_min': min(training_results.meters['loss'].deque),
                    'loss_bb_regression_avg': training_results.meters['loss_box_reg'].avg,
                    'loss_bb_regression_median': training_results.meters['loss_box_reg'].median,
                    'loss_bb_regression_max': training_results.meters['loss_box_reg'].max,
                    'loss_bb_regression_min': min(training_results.meters['loss_box_reg'].deque),
                    'loss_classifier_avg': training_results.meters['loss_classifier'].avg,
                    'loss_classifier_median': training_results.meters['loss_classifier'].median,
                    'loss_classifier_max': training_results.meters['loss_classifier'].max,
                    'loss_classifier_min': min(training_results.meters['loss_classifier'].deque),
                    'loss_rpn_bb_regression_avg': training_results.meters['loss_rpn_box_reg'].avg,
                    'loss_rpn_bb_regression_median': training_results.meters['loss_rpn_box_reg'].median,
                    'loss_rpn_bb_regression_max': training_results.meters['loss_rpn_box_reg'].max,
                    'loss_rpn_bb_regression_min': min(training_results.meters['loss_rpn_box_reg'].deque)}, ignore_index=True)
    return df


def df_add_iteration_log(df, epoch, iteration, training_results):
    """
        Adds row to the iteration logs dataframe.

        Args:
            df: -
            epoch: -
            iteration: -
            training_results: MetricLogger object with information from training epoch.

        Returns:
            df: -

        """

    df = df.append({'epoch': epoch, 'iteration': iteration, 'lr': training_results.meters['lr'], 'time': 0,
                    'loss_avg': training_results.meters['loss'].avg,
                    'loss_median': training_results.meters['loss'].median,
                    'loss_max': training_results.meters['loss'].max,
                    'loss_min': min(training_results.meters['loss'].deque),
                    'loss_bb_regression_avg': training_results.meters['loss_box_reg'].avg,
                    'loss_bb_regression_median': training_results.meters['loss_box_reg'].median,
                    'loss_bb_regression_max': training_results.meters['loss_box_reg'].max,
                    'loss_bb_regression_min': min(training_results.meters['loss_box_reg'].deque),
                    'loss_classifier_avg': training_results.meters['loss_classifier'].avg,
                    'loss_classifier_median': training_results.meters['loss_classifier'].median,
                    'loss_classifier_max': training_results.meters['loss_classifier'].max,
                    'loss_classifier_min': min(training_results.meters['loss_classifier'].deque),
                    'loss_rpn_bb_regression_avg': training_results.meters['loss_rpn_box_reg'].avg,
                    'loss_rpn_bb_regression_median': training_results.meters['loss_rpn_box_reg'].median,
                    'loss_rpn_bb_regression_max': training_results.meters['loss_rpn_box_reg'].max,
                    'loss_rpn_bb_regression_min': min(training_results.meters['loss_rpn_box_reg'].deque)}, ignore_index=True)
    return df
