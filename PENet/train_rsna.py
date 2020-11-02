import util
import torch
from torch.utils.data import Dataset
from saver import ModelSaver
import pandas as pd
import random
import numpy as np
import time
from models.loss import *
import os, json
from torch import nn

epochs = 100
learning_rate = 0.001
gamma = 0.5
momentum = 0.6
decay = 0.0005
schedule = [20, 40, 60, 80]
schedule = []
ngpu = 1
prefetch = 2
BCE_weight = 6.42

class KagglePEDataset(Dataset):
    """Kaggle PE dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file.
            root_dir (string): Directory with preprocessed npy images.
            transform (callable, optional): Not handled
        """
        self.pedataframe = pd.read_csv(csv_file)
        self.studies = self.pedataframe.StudyInstanceUID.unique()
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess_tabular_data()

        # Uncomment below to load npy images into memory before training.
        # I didn't find this to help.
        '''
        self.data_cache = []
        for study_index in range(len(self)):
            studyId = self.studies[study_index]
            study_file = self.root_dir + studyId + '.npy' # load preprocessed 3D np array of the CT scan
            img = np.load(study_file) # Nx208x208 int 8
            self.data_cache.append(img)
        '''

    def __len__(self):
        """ Return number of studies"""
        #return len(self.pedataframe)
        #return len(self.studies)
        return 512 # Start with a small sample

    def preprocess_tabular_data(self):
        '''
        preprocess the pandas dataframe so it doesn't take too much time when loading data.
        '''
        self.df_cache = []
        df_i = 0

        for study_index in range(len(self)):
            studyId = self.studies[study_index]
            start_i = df_i
            assert self.pedataframe['StudyInstanceUID'][start_i] == self.studies[study_index]
            while self.pedataframe['StudyInstanceUID'][df_i] == self.studies[study_index]:
                df_i += 1

            # append (start_index, end_index) of study into data frame.
            self.df_cache.append((start_i, df_i))
            
    def __getitem__(self, idx):
        a = time.perf_counter()
        
        # filter dataframe by study (this is already ordered by z-position)
        studyId = self.studies[idx]

        #study_df = self.pedataframe[self.pedataframe['StudyInstanceUID'] == studyId]
        pe_present_in_slice_list = self.pedataframe['pe_present_on_image'][self.df_cache[idx][0] : self.df_cache[idx][1]]

        # retrieve preprocessed scan from cache
        #study_file = self.root_dir + studyId + '.npy'
        #img = self.data_cache[idx]
        study_file = self.root_dir + studyId + '.npy' # load preprocessed 3D np array of the CT scan
        img = np.load(study_file) # Nx208x208 int 8
        #img = util.process_from_npy(img) # Nx192x192 (center cropped) float 32

        # randomly choose a 24 slice window
        total_num_slices = img.shape[-3]
        start_slice_index = random.randint(0, img.shape[-3]-24)
        img = img[start_slice_index : start_slice_index + 24, :, :] # 24x192x192 float32
        
        # Network expectd Bx1x24x192x192 (I'm not sure why the extra 1 dimension)
        img = torch.unsqueeze(torch.from_numpy(img), 0)

        # Calculate target (ground truth). PE positive if >=4 slices in 24 slice window is +ve.
        #pe_present_in_slice_list = study_df['pe_present_on_image']
        if len(pe_present_in_slice_list) != total_num_slices:
            print('ERROR! Number of slices mismatched. {} != {}'.format(
                  len(pe_present_in_slice_list), 
                  total_num_slices))
        
        windowed = pe_present_in_slice_list[start_slice_index : start_slice_index + 24]
        pe_present_on_image = sum([int(i) for i in windowed]) >= 4
        
        sample = {'image': img, 
                  'pe_present_on_image': pe_present_on_image}
        
        return sample

def train(net, loader, optimizer):
    ''' train function (forward, backward, update) '''
    net.train()
    #loss_fn = BinaryFocalLoss()
    #loss_fn = nn.BCELoss()
    pos_weight = torch.Tensor([BCE_weight]).cuda() # Empirically calculate -/+ sample ratio
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # combines sigmoid with BCE Loss
    
    running_loss = 0.0
    for batch_idx, sample_batched in enumerate(loader):
        data = sample_batched['image'].cuda()
        target = sample_batched['pe_present_on_image'].cuda()
        optimizer.zero_grad()

        # forward
        output = net(data.float())
        output = torch.squeeze(output,-1)
        loss = loss_fn(output, target.float())
        
        #probability = torch.squeeze(torch.sigmoid(output),-1)

        # backward
        #loss = loss_fn(probability, target.float())
        loss.backward()
        
        # update parameter weights
        optimizer.step()

        running_loss += loss.item()
        
    return running_loss / len(loader)

# test function (forward only)
def test(net, loader):
    net.eval()
    total_samples = 0
    total_true_positive = 0;
    total_false_positive = 0;
    total_true_negative = 0;
    total_false_negative = 0;
    
    pos_weight = torch.Tensor([BCE_weight]).cuda() # Empirically calculate -/+ sample ratio
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # combines sigmoid with BCE Loss
    running_loss = 0.0
    
    with torch.no_grad():
        for batch_idx, sample_batched in enumerate(loader):
            data = sample_batched['image'].cuda()
            target = sample_batched['pe_present_on_image'].cuda()
            
            # forward
            output = net(data.float())
            output = torch.squeeze(output,-1)
            loss = loss_fn(output, target.float())
            probability = torch.sigmoid(output)

            # accuracy
            prediction = (probability >= 0.5).long()
            correct = (prediction == target)

            running_loss += loss.item()
            total_samples += prediction.shape[0]

            # confusion matrix
            for sample_index in range(prediction.shape[0]):
                if target[sample_index] == 1: # target +ve
                    if correct[sample_index] == 1:
                        total_true_positive += 1
                    else:
                        total_false_positive += 1
                else: # target -ve
                    if correct[sample_index] == 1:
                        total_true_negative += 1
                    else:
                        total_false_negative += 1
                    
    return [total_true_positive, total_false_positive, total_true_negative, total_false_negative], running_loss / len(loader)

def main(): 
    #random.seed(3)
    #log = './'
    #save = './snapshots'

    state = {'learning_rate':learning_rate,'decay':decay,'momentum':momentum}

    # Init logger
    #if not os.path.isdir(log):
    #    os.makedirs(log)
    #log = open(os.path.join(log, 'log.txt'), 'w')
    #log.write(json.dumps(state) + '\n')

    # Init checkpoints
    #if not os.path.isdir(save):
    #    os.makedirs(save)
    
    # Instantiate Model
    device = 'cuda'
    model_path = '/projectnb/ece601/kaggle-pulmonary-embolism/meganmp/ckpts/penet_best.pth.tar'
    model, ckpt_info = ModelSaver.load_model(model_path, [0])
    model = model.to(device)

    # Feeze all the layers
    for param in model.parameters():
        param.requires_grad = False

    # Some debug code to print out paramter names
    #for name, param in model.named_parameters():
    #    print(name)

    # unfreeze Last layer
    model.module.classifier.fc.weight.requires_grad = True
    model.module.classifier.fc.bias.requires_grad = True
    
    # Instantiate dataset
    #data_dir = '/scratch/rsna-str-pulmonary-embolism-detection-208-npy/'
    data_dir = '/projectnb/ece601/kaggle-pulmonary-embolism/rsna-str-pulmonary-embolism-detection-208-npy/'
    train_csv = data_dir + 'train.csv'
    train_dir = data_dir + 'train/'
    dataset = KagglePEDataset(csv_file=train_csv, root_dir=train_dir)
    
    # Instantiate dataloader
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    confusion, ave_loss = test(model, validation_loader)
    print('Confusion matrix: [True +, False +, True -, False-]')
    print('Before training: ', confusion, 'loss: ', ave_loss)

    # test/train model
    for epoch in range(epochs):
        if epoch in schedule:
            state['learning_rate'] = state['learning_rate'] * gamma
            print('Learning rate: ', state['learning_rate'])
            
        # Optimizer (SGD)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    state['learning_rate'],
                                    momentum=state['momentum'],
                                    weight_decay=state['decay'],
                                    nesterov=True)
        # Train
        ave_train_loss = train(model, validation_loader, optimizer)

        # Test
        confusion, ave_test_loss = test(model, validation_loader)
        print('Training loss: ', ave_train_loss,
              'Testing loss: ', ave_test_loss,
              'Confusion: ', confusion)

if __name__ == "__main__": 
    main()












