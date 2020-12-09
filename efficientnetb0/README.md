Model for efficientnetb0:  
optimizer: SGD  
Loss function: BCEWithLogitsLoss 
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/efficientnetb0/IMG/efficientnetb0.PNG)

Best model path:    
/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/efficientnetb0/model-efficientb0-40.pth  

Model for LSTM when stage 1 is efficientnetb0 (1280 features):  
Features from efficientnetb0 path:  
/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/SequenceModeling/efficientb0_features.hdf5  
For Image Level:  
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/efficientnetb0/IMG/efficientnetb0_lstm_imagelevel.PNG)

  
/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/SequenceModeling/model-efficientb0-lstm1.pth (only trained few epoches since forgot to save)    
For Study Level:  
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/efficientnetb0/IMG/efficientnetb0_lstm_studylevel.PNG)
 
/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/SequenceModeling/model-efficientb0-lstm2.pth

Still training:  
Model for LSTM when stage 1 is combined resnet + efficientnetb0 (2048 + 1280 = 3328 features):    
For Image Level:      
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/efficientnetb0/IMG/combined_lstm_imagelevel.PNG)
/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/combined_Sequencemodel/model-combined-lstm1.pth  
For Study Level:  
/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/combined_Sequencemodel/model-combined-lstm2.pth
