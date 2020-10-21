from __future__ import division

import random
import os
from os import listdir
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision
from torchvision import models
import torchvision.datasets as dset
import torchvision.transforms as T
from torch.utils.data import TensorDataset,Dataset,DataLoader
from tqdm.auto import tqdm

from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms

import argparse
import pydicom
import glob
from matplotlib import pyplot as plt
from skimage.color import gray2rgb
import functools
import pydicom
import seaborn as sns
import scipy
import PIL

#ip install vtk
from sklearn.model_selection import KFold

import vtk
from vtk.util import numpy_support
from tqdm.auto import tqdm

#!pip install albumentations
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torchvision.models as models
model = models.vgg16(pretrained=True, progress=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

target_columns = ['pe_present_on_image', 'negative_exam_for_pe', 'rv_lv_ratio_gte_1', 
                  'rv_lv_ratio_lt_1','leftsided_pe', 'chronic_pe','rightsided_pe', 
                  'acute_and_chronic_pe', 'central_pe', 'indeterminate']

#vtk is used because dicom is giving some error

reader = vtk.vtkDICOMImageReader()
def get_img(path):
    reader.SetFileName(path)
    reader.Update()
    _extent = reader.GetDataExtent()
    ConstPixelDims = [_extent[1]-_extent[0]+1, _extent[3]-_extent[2]+1, _extent[5]-_extent[4]+1]

    ConstPixelSpacing = reader.GetPixelSpacing()
    imageData = reader.GetOutput()
    pointData = imageData.GetPointData()
    arrayData = pointData.GetArray(0)
    ArrayDicom = numpy_support.vtk_to_numpy(arrayData)
    ArrayDicom = ArrayDicom.reshape(ConstPixelDims, order='F')
    ArrayDicom = cv2.resize(ArrayDicom,(512,512))
    return ArrayDicom


def convert_to_rgb(array):
    array = array.reshape((512, 512, 1))
    return np.stack([array, array, array], axis=2).reshape((3,512, 512))

#data_dir = '/projectnb/ece601/kaggle-pulmonary-embolism/rsna-str-pulmonary-embolism-detection/'
data_dir = '/scratch/rsna-str-pulmonary-embolism-detection-265-jpeg/'
train_csv = data_dir + 'train.csv'
train_dir = data_dir + 'train/'

train = pd.read_csv(train_csv)
#test_data  = pd.read_csv(data_dir + "/test.csv")
#sample = pd.read_csv(data_dir + "/sample_submission.csv")

cols_ID = ["StudyInstanceUID","SeriesInstanceUID","SOPInstanceUID"]
train["ImagePath"] =train_dir+ train[cols_ID[0]]+"/"+train[cols_ID[1]]+"/"+train[cols_ID[2]]+".dcm"

class RsnaDataset(Dataset):
    
    def __init__(self,df,transforms=None):
        super().__init__()
        self.image_paths = df['ImagePath'].unique()
        self.df = df
        self.transforms = transforms
    
    def __getitem__(self,index):
        
        image_path = self.image_paths[index]
        data = self.df[self.df['ImagePath']==image_path]
        labels = data[target_columns].values.reshape(-1)
        image = get_img(image_path)
        image = convert_to_rgb(image)
        
        if self.transforms:
            image = self.transforms(image=image)['image']
            
        image = torch.tensor(image,dtype=torch.float)        
        labels = torch.tensor(labels,dtype=torch.float)
        
        return image,labels
           
    def __len__(self):
        return self.image_paths.shape[0]


classes = len(target_columns)
#in_features = model.fc.in_features
#model.fc = nn.Linear(in_features,classes)
model.classifier._modules['6'] = nn.Linear(4096, classes)

config={
       "learning_rate":0.001,
       "train_batch_size":32,
        "valid_batch_size":32,
        "test_batch_size":64,
       "epochs":10,
       "nfolds":3,
       "number_of_samples":7000
       }

transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def test():
    model.eval()
    all_prediction = np.zeros((test_data.shape[0],len(target_columns)))
    study_ids = list()
    sop_ids = list()
    for i in range(config["nfolds"]):
        #model.load_state_dict(torch.load(f"{model_path}model{i}.bin"))
        predictions = list()
        model.to(device)
        test_ds = RsnaDataset(test_data)
        test_dl = DataLoader(test_ds,
                        batch_size=config['test_batch_size'],
                        shuffle=False)
        
        tqdm_loader = tqdm(test_dl)
        
        with torch.no_grad():
            for inputs in tqdm_loader:
                images = inputs["image"].to(device, dtype=torch.float)
                outputs = model(images) 
                predictions.extend(outputs.cpu().detach().numpy())
                if i == 0:
                    study_ids.extend(inputs["study_id"])
                    sop_ids.extend(inputs["sop_id"])

        all_prediction += np.array(predictions)/config['nfolds']
        
    return all_prediction

sample = train.sample(n=config["number_of_samples"]).reset_index(drop=True)
test_data = sample

predictions = test()

