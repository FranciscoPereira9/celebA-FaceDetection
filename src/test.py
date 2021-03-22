# Imports
import torch
from torch.utils.data import DataLoader
import os
import argparse
from DatasetClass import MyDataset
import utils.helper as helper
from utils.engine import evaluate
from random import seed, randint

# ----------------------------------------------- Default Arguments ----------------------------------------------------

batch_size = 1
PATH = 'model.pt'

# ----------------------------------------------- Parsed Arguments -----------------------------------------------------

# Initiate the parser
parser = argparse.ArgumentParser()

# Add long and short argument
parser.add_argument("--batch_size", help="Set batch size.")
parser.add_argument("--path", help="Set path to model file.")

# Read arguments from the command line
args = parser.parse_args()

# Check arguments
print(33*"-")
if args.batch_size:
    batch_size = int(args.batch_size)
out = "| Batch size: " + str(batch_size)
print(out, (30 - len(out))*' ', '|')
if args.path:
    PATH = args.path
out = "| PATH: " + PATH
print(out, (30 - len(out))*' ', '|')
print(33*"-")


# ----------------------------------------------- Dataset Files --------------------------------------------------------

# Annotations directory path
ann_directory = '../../../../cross-sensor/datasets/CelebA/list_bbox_celeba.txt'
# Listed directory
#ann_files = os.listdir(ann_directory)

# Image directory path
img_directory = '../../../../cross-sensor/datasets/CelebA/img_celeba'
# Listed directory
#img_files = os.listdir(img_directory)


# ----------------------------------------------- Create Data Pipeline -------------------------------------------------

# Test Data
dataset_test = MyDataset(ann_directory, img_directory, mode='test')
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True, collate_fn=helper.collate_fn)

# ----------------------------------------------- Set Up the Model -----------------------------------------------------

# Setting up GPU device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Nº of classes: background, face
num_classes = 2
model = helper.build_model(num_classes)
model = model.to(device)

# Get saved model
model.load_state_dict(torch.load(PATH))

# ----------------------------------------------- Evaluation & Predictions ---------------------------------------------

# put the model in evaluation mode
model.eval()

# Evaluate the model
evaluate(model, loader_test, device=device)

# Make prediction on random image
n = randint(0, dataset_test.len)
img, target = dataset_test[n]
with torch.no_grad():
    prediction = model([img.to(device)])[0]

# Non max suppression to reduce the number of bounding boxes
nms_prediction = helper.apply_nms(prediction, iou_thresh=0.5)
# Remove low score boxes
filtered_prediction = helper.remove_low_score_bb(nms_prediction, score_thresh=0.2)

# Draw bounding boxes
helper.draw_bounding_boxes(img.detach().cpu(), target=target, prediction=filtered_prediction)
