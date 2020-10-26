# EC601-PENet

## Introduction
This project consists of (1) preprocessing steps from Kaggle notebooks and (2) adaptations to the PENet software to run our own data on the pre-trained model.

## PENet Instructions
Our data will not work on the unmodified files in the PENet Github repo. Necessary changes to the <code>environment.yml</code> and <code>requirements.txt</code> in this repository are necessary to run any of the following scripts so use the modified versions when setting up the environment. The checkpoints for the trained model are in <code>/projectnb/ec601/kaggle-pulmonary-embolism/meganmp/ckpts</code>.

**Set up environment:**
1. Run <code>conda env create -f environment.yml</code>

**To run on dicom files:**

1. Modify paths in <code>test_from_dicom.sh</code> to <code>input_study</code> and <code>ckpt_path</code>
2. <code>sh test_from_dicom.sh</code>

Returns a probability of PE in that study. As of 10/19, we can successfully run this code on single studies.

**To generate class activation maps (CAMs):**

1. Modify paths in <code>get_cams.sh</code> to <code>data_dir</code>, <code>ckpt_pth</code>, and <code>cam_dir</code>
2. <code>sh get_cams.sh</code>

**To test the model:**

1. Modify paths in <code>test.sh</code> to <code>data_dir</code>, <code>ckpt_pth</code>, and <code>results</code>
2. <code>sh test.sh</code>

**To train the model:**

*Currently consulting with corresponding author and co-author regarding the generation of a necessary pkl file. Updates forthcoming.*
