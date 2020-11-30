
Trained model weights: /projectnb/ece601/kaggle-pulmonary-embolism/cliao25/EC601-Pulmonary-Embolism/SequenceModeling/exp-4-SGD

Results from: 2D-ResNeXt-50-exp-4-SGD.ipynb

The training record at the bottom of the notebook tells you the validation loss for each epoch. You can take the model with smallest validation loss from above folder on SCC.

don't look at 2D-ResNeXt-50-exp-5-LSTM.ipynb.

2D Model training details
========================
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
