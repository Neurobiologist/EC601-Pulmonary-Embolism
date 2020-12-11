# CNN_LSTM Model

This file contains the code and related images on CNN_LSTM Model to detect PE.  
Trained models are saved in Model file outside.  

## Performance
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/tree/master/CNN-LSTM-Model/IMG)

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
 
 ## Stage 1 - 2D CNN Training

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

**Training Resnet Model**  
Run <code>2D-ResNeXt-50-train.ipynb</code>  
optimizer: SGD  
Loss function: BCEWithLogitsLoss 

**Training efficientnetb0 Model**    
Run <code>2D-efficientnetb0-train.ipynb</code>  
optimizer: SGD  
Loss function: BCEWithLogitsLoss  
![image](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/CNN-LSTM-Model/IMG/efficientnetb0.PNG)  

Best model path on SCC:    
<code>/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/efficientnetb0/model-efficientb0-40.pth</code>    


 ## Extract Features  

 1. Run <code>Feature-Vector-Generation-ResNeXt50.ipynb</code> to generate features.hdf5. Each sample contains 2048 features.  
 2. Run <code>Feature-Vector-Generation-efficientnetb0.ipynb</code> to generate features.hdf5. Each sample contains 1280 features. Features saved on SCC: <code>/projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/SequenceModeling/efficientb0_features.hdf5</code>  
 
 ## Stage 2 - Sequence Model (LSTM)   

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

