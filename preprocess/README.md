This folder contains some code specific to our data.

Kaggle RSNA Pulmonary Embolism dataset has around 1,790,000 512x512 DICOM files.

These files are big and expensive to process. This code takes DICOM files as input and outputs 256x256 JPEG files.

I created replicated the code across four files, with different ranges to make it run faster.

To submit on SCC:

```
qsub batch1.qsub
qsub batch2.qsub
qsub batch3.qsub
qsub batch4.qsub
```

This splits the data across 4 CPU batch jobs.

I found that the fourth batch has lots of images that need the GCDM python module to preprocess.
This module is not installed on the SCC. I need to create an anaconda environment for this.

To create the anaconda environment, do:

```
module load anaconda3
conda create -n kaggle
source activate kaggle
conda install -c conda-forge gdcm
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install -c conda-forge pydicom
conda install -c conda-forge pillow
conda install -c conda-forge numpy
conda install -c conda-forge pandas
```

To chow status of SCC job:
```
qstat -u <bu_username>
```
