# EC601-Project03

## Introduction
This project consists of (1) preprocessing steps from Kaggle notebooks and (2) adaptations to the PENet software to run our own data on the pre-trained model.

## PENet Instructions
Our data will not work on the unmodified files in the PENet Github repo. Necessary changes to the <code>environment.yml</code> and <code>requirements.txt</code> in this repository are necessary to run any of the following scripts so use the modified versions when setting up the environment. The checkpoints for the trained model are in <code>/projectnb/ec601/kaggle-pulmonary-embolism/meganmp/ckpts</code>.

**To run on dicom files:**

1. Modify paths in <code>test_from_dicom.sh</code> to <code>input_study</code> and <code>ckpt_path</code>
2. <code>sh test_from_dicom.sh</code>

Returns a probability of PE in that study. As of 10/19, we can successfully run this code on single studies.

**To generate class activation maps (CAMs):**

1. Modify paths in <code>get_cams.sh</code> to <code>data_dir</code>, <code>ckpt_pth</code>, and <code>cam_dir</code>
2. <code>sh get_cams.sh</code>

## Kaggle NB Preprocessing
Using the available Kaggle NB, we've done the following:

* Loading the CT scans per patient and probing the structure of the data
* Understanding Digital Imaging and Communications in Medicine (DICOM) files, Hounsfield units, CT windows, and CT levels; why it's important to concentrate 256 shades of grey into a reasonable range of densities based on our target tissue because our eyes can only detect a 6% difference in greyscale[[1]](#1)
* 

## References
<a id="1">[1]</a> https://www.youtube.com/watch?v=KZld-5W99cI
