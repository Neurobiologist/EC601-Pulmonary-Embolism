# -*- coding: utf-8 -*-
from __future__ import division
import argparse
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms as transforms
import pandas as pd
import os
import pydicom
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import os
from os import listdir
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader,Dataset
from skimage.color import gray2rgb
import functools
from tqdm.auto import tqdm
import pydicom
import seaborn as sns
import scipy
import PIL
import json

class KagglePEDataset(torch.utils.data.Dataset):
    """Kaggle PE dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.pedataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        """ Return number of 2D images. (Each CT slice is an independent image.)"""
        return len(self.pedataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.pedataframe.StudyInstanceUID[idx],
                                self.pedataframe.SeriesInstanceUID[idx],
                                self.pedataframe.SOPInstanceUID[idx] + '.dcm')
        dicom_image = pydicom.dcmread(img_name) 
        image = dicom_image.pixel_array
        
        # in OSIC we find outside-scanner-regions with raw-values of -2000. 
        # Let's threshold between air (0) and this default (-2000) using -1000
        image[image <= -1000] = 0
        
        # convert to HU using DICOM information
        # HU is a number between -1000 and 1000 (generally)
        # good lung tissue is between -950 and -700 (approximately)
        intercept = dicom_image.RescaleIntercept
        slope = dicom_image.RescaleSlope
        
        if slope != 1:
            image = slope * image.astype(np.float64)
            
        image = image.astype(np.int16)
        image += np.int16(intercept)
        
        # Convert image from numpy array to PIL image (so that we can use pytorch transforms)
        image[image >= 500] = 500
        image[image <= -1000] = -1000
        image = (image + 1000)/1500
        image = image*255
        image = np.uint8(image)
        image = PIL.Image.fromarray(image).convert('RGB')

        # image is 512x512 RGB PIL image
        # pe_present_on_image is 0 or 1
        sample = {'image': image, 
                  'pe_present_on_image': int(self.pedataframe.pe_present_on_image[idx])}

        # Only apply transform to image.
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            
        return sample

data_dir = '/projectnb/ece601/kaggle-pulmonary-embolism/rsna-str-pulmonary-embolism-detection/'
train_csv = data_dir + 'train.csv'
train_dir = data_dir + 'train/'


# In[ ]:


pedataframe = pd.read_csv(train_csv)

for idx in range(1200000,len(pedataframe)):
    img_name = os.path.join(train_dir,
                            pedataframe.StudyInstanceUID[idx],
                            pedataframe.SeriesInstanceUID[idx],
                            pedataframe.SOPInstanceUID[idx] + '.dcm')
    dicom_image = pydicom.dcmread(img_name) 
    image = dicom_image.pixel_array
        
    # in OSIC we find outside-scanner-regions with raw-values of -2000. 
    # Let's threshold between air (0) and this default (-2000) using -1000
    image[image <= -1000] = 0
        
     # convert to HU using DICOM information
    # HU is a number between -1000 and 1000 (generally)
    # good lung tissue is between -950 and -700 (approximately)
    intercept = dicom_image.RescaleIntercept
    slope = dicom_image.RescaleSlope
        
    if slope != 1:
        image = slope * image.astype(np.float64)
            
    image = image.astype(np.int16)
    image += np.int16(intercept)
        
    # Convert image from numpy array to PIL image (so that we can use pytorch transforms)
    image[image >= 500] = 500
    image[image <= -1000] = -1000
    image = (image + 1000)/1500
    image = image*255
    image = np.uint8(image)
    image = PIL.Image.fromarray(image).convert('RGB')
    
    out_traindir = '/projectnb/ece601/kaggle-pulmonary-embolism/rsna-str-pulmonary-embolism-detection-265-jpeg/train/'
        
    directory = os.path.join(out_traindir,
                             pedataframe.StudyInstanceUID[idx],
                             pedataframe.SeriesInstanceUID[idx])
                             
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    out_filename = os.path.join(out_traindir,
                            pedataframe.StudyInstanceUID[idx],
                            pedataframe.SeriesInstanceUID[idx],
                            pedataframe.SOPInstanceUID[idx] + '.jpg')
    
    transform = T.Resize(256)
    image = transform(image)
    image.save(out_filename)
    if idx % 10000 == 0:
        print(idx)
                             


# In[ ]:




