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
