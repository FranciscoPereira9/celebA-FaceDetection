# Imports
import torch
from torch.utils.data import DataLoader
import pandas as pd
import os
import argparse
from DatasetClass import MyDataset
from datetime import datetime
from utils.engine import train_one_epoch, evaluate
import utils.helper as helper
import src.utils.utils as utils


# ----------------------------------------------- Default Arguments & Variables ----------------------------------------

# File name of this runtime
now = datetime.now()
filename = now.strftime("%Y_%b_%d_%Hh_%Mm")
# Make dir to save the resulting data from training
PATH = '../models/model_ces_' + filename
utils.mkdir(PATH)
# Defaults
batch_size = 1
epochs = 1
optimizer_type = 'sgd'
lr = 0.1
# Aux
best_mAP = 0

# ----------------------------------------------- Parsed Arguments -----------------------------------------------------

# Initiate the parser
parser = argparse.ArgumentParser()

# Add long and short argument
parser.add_argument("--batch_size", help="Set batch size.")
parser.add_argument("--epochs", help="Set number of epochs.")
parser.add_argument("--optimizer", help="Set optimizer. Can be 'sgd' or 'adam'.")
parser.add_argument("--learning_rate", help="Set learning rate.")

# Read arguments from the command line
args = parser.parse_args()

# Check arguments
print(33 * "-")
if args.batch_size:
    batch_size = int(args.batch_size)
out = "| Batch size: " + str(batch_size)
print(out, (30 - len(out)) * ' ', '|')
if args.epochs:
    epochs = int(args.epochs)
out = "| Number of epochs: " + str(epochs)
print(out, (30 - len(out)) * ' ', '|')
if args.optimizer:
    optimizer_type = args.optimizer
out = '| Optimizer type: ' + optimizer_type
print(out, (30 - len(out)) * ' ', '|')
if args.learning_rate:
    lr = float(args.learning_rate)
out = '| Learning rate: ' + str(lr)
print(out, (30 - len(out)) * ' ', '|')
print(33 * "-")

# ----------------------------------------------- Dataset Files --------------------------------------------------------

# Annotations directory path
ann_file = '../../../../cross-sensor/datasets/CelebA/list_bbox_celeba.txt'

# Image directory path
img_directory = '../../../../cross-sensor/datasets/CelebA/img_celeba'


# ----------------------------------------------- Create Data Pipeline -------------------------------------------------

# Training Data
dataset_train = MyDataset(ann_file, img_directory, mode='train')
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=helper.collate_fn)

# Validation Data
dataset_validation = MyDataset(ann_file, img_directory, mode='validation')
loader_validation = DataLoader(dataset_validation, batch_size=batch_size, shuffle=True, collate_fn=helper.collate_fn)

# ----------------------------------------------- Set Up the Model -----------------------------------------------------

# Setting up GPU device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Nº of classes: background, face
num_classes = 2
model = helper.build_model(num_classes)
model = model.to(device)

# Network params
params = [p for p in model.parameters() if p.requires_grad]

# Optimizers
if optimizer_type == 'adam':
    optimizer = torch.optim.Adam(params, lr=lr)
else:
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)

if epochs > 20:
    step_size = round(epochs / 7)
else:
    step_size = 3

# Learning Rate, lr decreases by half every step_size
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

# ----------------------------------------------- Train the Model ------------------------------------------------------

# Create Dataframe with epochs stats
columns_epochs = ['epoch', 'lr', 'time', 'mAP'
                  'loss_avg', 'loss_median', 'loss_max', 'loss_min',
                  'loss_bb_regression_avg', 'loss_bb_regression_median',
                  'loss_bb_regression_max', 'loss_bb_regression_min',
                  'loss_classifier_avg', 'loss_classifier_median',
                  'loss_classifier_max', 'loss_classifier_min',
                  'loss_rpn_bb_regression_avg', 'loss_rpn_bb_regression_median',
                  'loss_rpn_bb_regression_max', 'loss_rpn_bb_regression_min']

columns_iterations = ['epoch', 'iteration', 'lr', 'time',
                      'loss_avg', 'loss_median', 'loss_max', 'loss_min',
                      'loss_bb_regression_avg', 'loss_bb_regression_median',
                      'loss_bb_regression_max', 'loss_bb_regression_min',
                      'loss_classifier_avg', 'loss_classifier_median',
                      'loss_classifier_max', 'loss_classifier_min',
                      'loss_rpn_bb_regression_avg', 'loss_rpn_bb_regression_median',
                      'loss_rpn_bb_regression_max', 'loss_rpn_bb_regression_min']

train_epochs_log = pd.DataFrame(columns=columns_epochs)
train_iterations_log = pd.DataFrame(columns=columns_iterations)

# Train the network (saving the best model)
for epoch in range(0, epochs):
    # train for one epoch, printing every <print_freq> iterations
    training_results, train_iterations_log = train_one_epoch(model, optimizer, loader_train, device, epoch,
                                                             print_freq=10, df=train_iterations_log)

    # evaluate on the validation data set
    mAP = evaluate(model, loader_validation, device=device)

    # add epoch logs to df
    train_epochs_log = helper.df_add_epoch_log(train_epochs_log, epoch, mAP[0], training_results)

    # Check to keep best model
    if mAP[0] > best_mAP:
        best_mAP = mAP[0]
        # Save model
        torch.save(model.state_dict(), PATH + '/' + filename + '.pt')

    # update the learning rate
    lr_scheduler.step()

# ----------------------------------------------- Save Training Logs ---------------------------------------------------

# Save training logs
train_epochs_log.to_csv(PATH + '/' + filename + '_epochs.csv', index=False, header=True)
train_iterations_log.to_csv(PATH + '/' + filename + '_iterations.csv', index=False, header=True)
