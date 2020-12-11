#!/usr/bin/env python
# coding: utf-8

# In[3]:


from __future__ import division
import pandas as pd
import os
import pydicom
import numpy as np
import h5py
import time
import pickle


# In[4]:


data_dir = '/projectnb/ece601/kaggle-pulmonary-embolism/rsna-str-pulmonary-embolism-detection/'
train_csv = data_dir + 'train.csv'
train_dir = data_dir + 'train/'
train = pd.read_csv(train_csv)


# In[5]:


train_pos = train[train.pe_present_on_image == 1]
train_neg = train[train.pe_present_on_image == 0]
print('Slice-wise PE positive: ', len(train_pos))
print('Slice-wise PE negative: ', len(train_neg))
print('Proportion: ', len(train_neg)/len(train_pos))


# In[ ]:

pedataframe = train_pos

h5py_file = h5py.File('/scratch/npy-pe-pos.hdf5', "w")
for i in range(len(pedataframe)):
    
    idx = pedataframe.index[i]
    img_name = os.path.join(train_dir,
                            pedataframe.StudyInstanceUID[idx],
                            pedataframe.SeriesInstanceUID[idx],
                            pedataframe.SOPInstanceUID[idx] + '.dcm')
    dicom_image = pydicom.dcmread(img_name) 

    try:
        # RuntimeError: The following handlers are available to decode the pixel ...
        # data however they are missing required dependencies: GDCM (req. GDCM)
        image = dicom_image.pixel_array
    except:
        print('Error parsing ', img_name)
        continue

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

    h5py_file.create_dataset(pedataframe.StudyInstanceUID[idx] + '/' + pedataframe.SOPInstanceUID[idx], data=image)

    if i % 10000 == 0:
        print(idx)

h5py_file.close()
