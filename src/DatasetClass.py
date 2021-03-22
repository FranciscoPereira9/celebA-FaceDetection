# Imports
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import math
import os
import xml.etree.ElementTree as ET
import re
import ast


# Create dataset object
class MyDataset(Dataset):

    # Constructor
    def __init__(self, ann_file, img_dir, transform=None, mode='train'):

        # Image directories
        self.ann_file = ann_file
        self.img_dir = img_dir

        # The transform that is going to be used on image
        self.transform = transform

        # If annotations have already been saved
        try:
            self.data = pd.read_csv('../dataloader/df.csv')
        except:
            print('No data in expected folder. Initialize data annotations reading...')

            # Create dataframe to hold info
            self.data = pd.DataFrame(columns=['Filename', 'BoundingBoxes', 'Labels', 'Area', 'N_Objects'])

            # Get annotations information in df
            ann_info = pd.read_csv(ann_file, delim_whitespace=True)

            # Append rows with image filename and respective bounding boxes to the df
            for file in enumerate(os.listdir(img_dir)):
                # Find image annotation row
                row = ann_info[ann_info['image_id'] == file[1]]

                # Create list of bounding boxes in an image
                list_bb = []
                list_labels = []
                list_area = []
                list = [row['x_1'].item(), row['y_1'].item(), row['x_1'].item() + row['width'].item(),
                        row['y_1'].item() + row['height'].item()]
                list_bb.append(list)
                list_labels.append(1)
                list_area.append(row['width'].item() * row['height'].item())

                # Create dataframe object with row containing [(Image file name),(Bounding Box List)]
                df = pd.DataFrame([[file[1], list_bb, list_labels, list_area, 1]],
                                  columns=['Filename', 'BoundingBoxes', 'Labels', 'Area', 'N_Objects'])
                self.data = self.data.append(df)

            # Save the collected information in the right format into a csv
            self.data.to_csv('../dataloader/df.csv', index=False, header=True)

        # Data division
        train_len = math.ceil(len(self.data) * 0.80)
        train_len = 5000
        val_len = math.floor(len(self.data) * 0.05)
        val_len = 20
        test_len = math.floor(len(self.data) * 0.15)
        test_len = 100

        if mode == 'train':
            self.data = self.data[:train_len]
        elif mode == 'validation':
            self.data = self.data[train_len:train_len + val_len]
        elif mode == 'test':
            self.data = self.data[train_len + val_len:train_len + val_len+test_len]

        # Number of images in dataset
        self.len = self.data.shape[0]

    # Get the length
    def __len__(self):
        return self.len

    # Getter
    def __getitem__(self, idx):

        # Image file path
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])

        # Open image file and tranform to tensor
        img = Image.open(img_name).convert('RGB')

        # Get bounding box coordinates (verify if array of numbers)
        if self.data.BoundingBoxes.dtype == 'O':
            bbox = torch.tensor(self.str2array(self.data.iloc[idx, 1]), dtype=torch.float32)
        else:
            bbox = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)

        # Get labels (verify if array of numbers)
        if self.data.Labels.dtype == 'O':
            labels = torch.tensor(self.str2array(self.data.iloc[idx, 2]), dtype=torch.int64)
        else:
            labels = torch.tensor(self.data.iloc[idx, 2], dtype=torch.int64)

        # Get bounding box areas (verify if array of numbers)
        if self.data.Area.dtype == 'O':
            area = torch.tensor(self.str2array(self.data.iloc[idx, 3]), dtype=torch.float32)
        else:
            area = torch.tensor(self.data.iloc[idx, 3], dtype=torch.float32)

        # Get number of objects in image (verify if int)
        num_objs = int(self.data.iloc[idx, 4])


        # If any, aplly tranformations to image and bounding box mask
        if self.transform:
            # Convert PIL image to numpy array
            img = np.array(img)
            # Apply transformations
            transformed = self.transform(image=img, bboxes=bbox)
            # Convert numpy array to PIL Image
            img = Image.fromarray(transformed['image'])
            # Get transformed bb
            bbox = torch.tensor(transformed['bboxes'])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Transform img to tensor
        img = torchvision.transforms.ToTensor()(img)

        # Build Target dict
        target = {"boxes": bbox, "labels": labels, "image_id": torch.tensor([idx]), "area": area, "iscrowd": iscrowd}

        return img, target

    # XML reader -> returns dictionary with image bounding boxes sizes
    def read_XML_classf(self, ann_file_path):
        bboxes = [{
            'file': ann_file_path,
            'labels': [],
            'objects': []
        }]

        # Reading XML file objects and print Bounding Boxes
        tree = ET.parse(ann_file_path)
        root = tree.getroot()
        objects = root.findall('object')

        for obj in objects:
            # label
            label = obj.find('name').text
            bboxes[0]['labels'].append(label)

            # bbox dimensions
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            bboxes[0]['objects'].append({'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})

        return bboxes

    def str2array(self, s):
        # Remove space after [
        s = re.sub('\[ +', '[', s.strip())
        # Replace commas and spaces
        s = re.sub('[,\s]+', ', ', s)
        return np.array(ast.literal_eval(s))