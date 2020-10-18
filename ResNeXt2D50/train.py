#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms as transforms
import pandas as pd
import os
import pydicom
import numpy as np
import glob
from os import listdir
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from skimage.color import gray2rgb
import functools
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
                                self.pedataframe.SOPInstanceUID[idx] + '.jpg')
        jpeg_image = PIL.Image.open(img_name) 

        # image is 256x256 RGB PIL image
        # pe_present_on_image is 0 or 1
        sample = {'image': jpeg_image, 
                  'pe_present_on_image': int(self.pedataframe.pe_present_on_image[idx])}

        # Only apply transform to image.
        if self.transform:
            sample['image'] = self.transform(sample['image'])
            
        return sample

#data_dir = '/projectnb/ece601/kaggle-pulmonary-embolism/rsna-str-pulmonary-embolism-detection-265-jpeg/'
data_dir = '/scratch/rsna-str-pulmonary-embolism-detection-265-jpeg/'
train_csv = data_dir + 'train.csv'
train_dir = data_dir + 'train/'

#resnext101 = models.resnext101_32x8d(pretrained=True, progress=True)
resnext50 = models.resnext50_32x4d(pretrained=True, progress=True)

# use values from sample image (but ideally this should be values from entire dataset)
global_mean = 0.5
global_std = 0.25

transform=T.Compose([T.Resize(256),
                     T.RandomCrop(224),
                     T.ToTensor(),
                     T.Normalize(mean=[global_mean, global_mean, global_mean], 
                                          std=[global_std, global_std, global_std]),
                    ])

transformed_dataset = KagglePEDataset(csv_file=train_csv, root_dir=train_dir, transform=transform)

# Replace last layer with number of outputs we need.
#resnext101.fc = torch.nn.Linear(resnext101.fc.in_features, 2)
resnext50.fc = torch.nn.Linear(resnext50.fc.in_features, 2)

batch_size = 64
validation_split = .2
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
# Be sure that the balance of the data in both sets are the same.
# i.e. they both have the same percentage of 
dataset_size = len(transformed_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(transformed_dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

print('Entire dataset:')
print(transformed_dataset.pedataframe['pe_present_on_image'].value_counts(normalize=True))

print('Training split:')
a = transformed_dataset.pedataframe['pe_present_on_image'][train_indices]
print(a.value_counts(normalize=True))

print('Validation split:')
b = transformed_dataset.pedataframe['pe_present_on_image'][val_indices]
print(b.value_counts(normalize=True))

epochs = 2
learning_rate = 0.1
gamma = 0.5
momentum = 0.9
decay = 0.0005
schedule = [20, 40, 60, 80, 100, 120, 140, 160]
ngpu = 1
prefetch = 2
log = './'
save = './snapshots'

# Init logger
if not os.path.isdir(log):
    os.makedirs(log)
log = open(os.path.join(log, 'log.txt'), 'w')
state = {'learning_rate':learning_rate,'decay':decay,'momentum':momentum}
log.write(json.dumps(state) + '\n')

# Init checkpoints
if not os.path.isdir(save):
    os.makedirs(save)

# Init model, criterion, and optimizer
#net = resnext101
net = resnext50

ngpu
if ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(ngpu)))

if ngpu > 0:
    net.cuda()

optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
                            weight_decay=state['decay'], nesterov=True)

# train function (forward, backward, update)
def train():
    net.train()
    loss_avg = 0.0
    for batch_idx, sample_batched in enumerate(train_loader):
        data = sample_batched['image'].cuda()
        target = sample_batched['pe_present_on_image'].cuda()

        # forward
        output = net(data.float())

        # backward
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        
        # update parameter weights
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.2 + float(loss) * 0.8

    state['train_loss'] = loss_avg

# test function (forward only)
def test():
    net.eval()
    loss_avg = 0.0
    correct = 0
    for batch_idx, sample_batched in enumerate(validation_loader):
        data = torch.autograd.Variable(sample_batched['image'].cuda())
        target = torch.autograd.Variable(sample_batched['pe_present_on_image'].cuda())

        # forward
        output = net(data.float())
        loss = F.cross_entropy(output, target)

        # accuracy
        pred = output.data.max(1)[1]
        correct += float(pred.eq(target.data).sum())

        # test loss average
        loss_avg += float(loss)

    state['test_loss'] = loss_avg / len(validation_loader)
    state['test_accuracy'] = correct / len(validation_loader.dataset)


# Main loop
best_accuracy = 0.0
for epoch in range(epochs):
    if epoch in schedule:
        state['learning_rate'] *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['learning_rate']

    state['epoch'] = epoch
    train()
    test()
    if state['test_accuracy'] > best_accuracy:
        best_accuracy = state['test_accuracy']
        torch.save(net.state_dict(), os.path.join(save, 'model.pytorch'))
    log.write('%s\n' % json.dumps(state))
    log.flush()
    print(state)
    print("Best accuracy: %f" % best_accuracy)

log.close()
