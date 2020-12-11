# CNN_LSTM Model

This file contains the code and related images on CNN_LSTM Model to detect PE.  
Trained models are saved in Model file outside.  

## Performance
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/ROC%20Curve%20for%20CNN_LSTM.png)

## Pipeline

We use a two stage training process as illustrated in the diagram below.

Follow the following steps:

1. Preprocess data into hdf5 files for fast access.
2. Train a Stage 1 resnext model.
3. Train a Stage 1 efficientnet model.
4. Run both Stage 1 models to generate feature vectors. Combine feature vectors into one vector.
5. Use combined feature vector to train Stage 2 LSTM.
6. Test the LSTM.

![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/CNN_LSTM%20pipeline.PNG)

## Overall Strategy

* 2D CNN Model (Resnext, Efficientnetb0) used for feature extraction per image.
* Combine the features and input into sequence model (lstm).

## Datastes and Preprocessing  

The following code is used to convert DICOM to numpy array:

```
 dicom_image = pydicom.dcmread(img_name)  
 image = dicom_image.pixel_array
```
 
 To batch process data on BU SCC into HDF5 files, run:
 
 ```
 qsub Make-HDF5-Batch.qsub
 ```
 
 This will generate one HDF5 for every 100,000 samples slices.
 
 This will genearte a separate HDF5 file for just the positive samples, since we need to use the positive samples for every epoch.
 
 ## Stage 1 - 2D Model training
 
There are a total of 96,540 positive slices.

We split this into 70,000 training and 26,540 validation.

Because we have more negative samples, we can rotate through them (not train on all the negative samples at once).

For each training epoch, we train on 140,000 samples (70,000+, 70,000-) and validate on (26,540+, 26,540-).

**Training Resnet Model**  
Run <code>2D-ResNeXt-50-train.ipynb</code>  
optimizer: SGD  
Loss function: BCEWithLogitsLoss 

**Training efficientnetb0 Model**    
Run <code>2D-efficientnetb0-train.ipynb</code>  
optimizer: SGD  
Loss function: BCEWithLogitsLoss  
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/efficientnetb0.PNG)  

Best model paths on SCC:    

Efficient Net: <code>/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/efficientnetb0/model-efficientb0-40.pth</code>    

ResNeXt: <code>/projectnb/ece601/kaggle-pulmonary-embolism/cliao25/EC601-Pulmonary-Embolism/SequenceModeling/exp-4-SGD/model-resnext-50-28.pth</code>

 ## Extract Features  

 1. Run <code>Feature-Vector-Generation-ResNeXt50.ipynb</code> to generate resnet_features.hdf5. Each slice is compressed to 2048 features.  
 2. Run <code>Feature-Vector-Generation-efficientnetb0.ipynb</code> to generate efficientnetb0_features.hdf5. Each slice is compressed to 1280 features. 
 
 Features saved on SCC: 
 
 Efficient Net: <code>/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/SequenceModeling/efficientb0_features.hdf5</code>  
 
 ResNeXt: <code>/projectnb/ece601/kaggle-pulmonary-embolism/cliao25/EC601-Pulmonary-Embolism/SequenceModeling/resnet_features.hdf5</code>
 
 ## Stage 2 - Sequence Model (LSTM)
 
 We train a total of 6 LSTMs.
 
 We experimented with using features from ResNeXt and Efficient Net, or combining both. Combining both features gives better performance.
 
 There are two flavors of LSTM models we train:
  1. Image Level: Output a binary value for each slice, indicating PE positive or negative.
  2. Study Level: Output 9 binary values per study. Each binary value is a study level attribute that can help PE diagnosis. See code for the names of each attribute.

**LSTM Model input features from Resnet**  
Run <code>LSTM_resnext_train.ipynb</code> 

**LSTM Model input features from efficientnetb0**  
Run <code>LSTM_efficientnetb0_train.ipynb</code>  
<code>/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/SequenceModeling/efficientb0_features.hdf5</code>    
* For Image Level:  
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/efficientnetb0_lstm_imagelevel.PNG)

Best model path on SCC:  
<code>/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/SequenceModeling/model-efficientb0-lstm1.pth</code>   
* For Study Level:  
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/efficientnetb0_lstm_studylevel.PNG)

Best model path on SCC:  
<code>/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/SequenceModeling/model-efficientb0-lstm2.pth</code>  


**Model for LSTM when stage 1 is combined resnet + efficientnetb0**     
Run <code>LSTM_efficientnetb0+resnet.ipynb</code>  
* For Image Level:      
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/combined_lstm_imagelevel.PNG)  

Best model path on SCC:  
<code>/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/combined_Sequencemodel/model-combined-lstm1.pth</code>    
* For Study Level:    
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/combined_lstm_studylevel.PNG)  

Best model path on SCC:  
<code>/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/combined_Sequencemodel/model-combined-lstm2.pth</code>    

