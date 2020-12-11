Model for efficientnetb0:  
2D-efficientb0-SGD.ipynb  
optimizer: SGD  
Loss function: BCEWithLogitsLoss 
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/efficientnetb0.PNG)

Best model path:    
/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/efficientnetb0/model-efficientb0-40.pth  

Model for LSTM when stage 1 is efficientnetb0 (1280 features):
LSTM1280_efficientnetb0.ipynb  
Features from efficientnetb0 path:  
Feature-Vector-Generation-efficientnetb0.ipynb  
/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/SequenceModeling/efficientb0_features.hdf5  
For Image Level:  
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/efficientnetb0_lstm_imagelevel.PNG)

/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/SequenceModeling/model-efficientb0-lstm1.pth (only trained few epoches since forgot to save)    
For Study Level:  
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/efficientnetb0_lstm_studylevel.PNG)
 
/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/SequenceModeling/model-efficientb0-lstm2.pth


Model for LSTM when stage 1 is combined resnet + efficientnetb0 (2048 + 1280 = 3328 features):   
LSTM_efficientnetb0+resnet.ipynb  
For Image Level:      
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/combined_lstm_imagelevel.PNG)

/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/combined_Sequencemodel/model-combined-lstm1.pth  
For Study Level:    
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/combined_lstm_studylevel.PNG)  

/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/combined_Sequencemodel/model-combined-lstm2.pth


Trained model weights: /projectnb/ece601/kaggle-pulmonary-embolism/cliao25/EC601-Pulmonary-Embolism/SequenceModeling/exp-4-SGD

Results from: 2D-ResNeXt-50-exp-4-SGD.ipynb

The training record at the bottom of the notebook tells you the validation loss for each epoch. You can take the model with smallest validation loss from above folder on SCC.

don't look at 2D-ResNeXt-50-exp-5-LSTM.ipynb.

2D Model training details
---
In /projectnb/ece601/kaggle-pulmonary-embolism/cliao25/EC601-Pulmonary-Embolism/SequenceModeling/,

you will find:

npy-1.hdf5
npy-2.hdf5
npy-3.hdf5
npy-4.hdf5
npy-5.hdf5
npy-6.hdf5
npy-7.hdf5
npy-8.hdf5
npy-pe-pos.hdf5

Each file stores 100,000 slices from the data. (Split into separate files to be more manageable)

npy-pe-pos.hdf5 stores all positive slices from the data. I put all the positive slices together, because there are very few of these.

There are a total of 96,540 positive slices.
We split this into 70,000 training and 26,540 validation.

Because we have more negative samples, we can rotate through them (not train on all the negative samples at once).

For each training epoch, we train on 140,000 samples (70,000+, 70,000-) and validate on (26,540+, 26,540-).
