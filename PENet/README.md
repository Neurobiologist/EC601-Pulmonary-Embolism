# EC601-PENet

## Introduction
This project consists of (1) preprocessing steps from Kaggle notebooks and (2) adaptations to the PENet software to run our own data on the pre-trained model.

## PENet Instructions
Our data will not work on the unmodified files in the PENet Github repo. Necessary changes to the <code>environment.yml</code> and <code>requirements.txt</code> in this repository are necessary to run any of the following scripts so use the modified versions when setting up the environment. The checkpoints for the trained model are in <code>/projectnb/ec601/kaggle-pulmonary-embolism/meganmp/ckpts</code>.

**Set up environment:**
1. Run <code>conda env create -f environment.yml</code>

*N.B.: This might default to your home directory on the SCC. If so, try* <code>conda env create -f environment.yml -p \<project-dir\></code>



**To run on dicom files:**

1. Modify paths in <code>test_from_dicom.sh</code> to <code>input_study</code> and <code>ckpt_path</code>
2. <code>sh test_from_dicom.sh</code> 

Returns a probability of PE in that study. As of 10/19, we can successfully run this code on single studies.

**To generate pickle file and hdf5 file from Kaggle data:**

1. Modify <code>constants.py</code> in directory <code>/projectnb/ece601/kaggle-pulmonary-embolism/meganmp/PENet/rsna</code> to indicate proper directories for data, CSV files, and results.
2. Run <code>python parse_rsna_data.py</code> to generate <code>series_list.pkl</code> and <code>data.hdf5</code>

**To test the model on Kaggle data:**

1. Generate <code>series_list.pkl</code> from above instructions.
2. Modify paths in <code>run_test_rsna.sh</code>
3. Run <code>python test_rsna.py</code>

**To generate class activation maps (CAMs):**

1. Modify paths in <code>get_cams.sh</code> to <code>data_dir</code>, <code>ckpt_pth</code>, and <code>cam_dir</code>
2. <code>sh get_cams.sh</code>

**To test the model:**

1. Modify paths in <code>test.sh</code> to <code>data_dir</code>, <code>ckpt_pth</code>, and <code>results</code>
2. <code>sh test.sh</code>





## Model Training

(1) Preprocess data using <code>experiments/preprocess.py</code>. Do this on SCC using <code>qsub experiments/preprocess.qsub</code>

This script preprocesses the DICOM files, and stores each study as a single .npy file (3D int8 numpy array).

(2) Train the PENet model using preprocessed data. Execute <code>python3 train_rsna.py</code>

This script trains the last layer of the PENet on 24 slice windows randomly extracted from the 3D volume.
It uses Weighted Binary Cross Entropy (BCE) loss to address class imbalance.
The script trains and tests on a 512 studies. The loss and confusion matrix is printed out for each epoch.

Training output for BCE Weight=6.42
```
/projectnb/ece601/kaggle-pulmonary-embolism/meganmp/ckpts/penet_best.pth.tar
Confusion matrix: [True +, False +, True -, False-]
Before training:  [0, 77, 435, 0] loss:  2.2430192943429574
Training loss:  1.5486452913610265 Testing loss:  1.221583612728864 Confusion:  [18, 50, 309, 135]
Training loss:  1.2574980861973017 Testing loss:  1.1993182031437755 Confusion:  [30, 38, 258, 186]
Training loss:  1.1881880860310048 Testing loss:  1.2456099740229547 Confusion:  [29, 46, 293, 144]
Training loss:  1.2585547482594848 Testing loss:  1.2536943764425814 Confusion:  [30, 44, 266, 172]
Training loss:  1.1689363338518888 Testing loss:  1.2303415923379362 Confusion:  [18, 53, 331, 110]
Training loss:  1.2548361558001488 Testing loss:  1.1966853835619986 Confusion:  [30, 39, 284, 159]
```

Training output for BCE Weight=8.0
```
Confusion matrix: [True +, False +, True -, False-]
Before training:  [0, 71, 441, 0] loss:  2.577443141490221
Training loss:  1.4621236596722156 Testing loss:  1.443847042741254 Confusion:  [9, 61, 354, 88]
Training loss:  1.3440967523492873 Testing loss:  1.3258668668568134 Confusion:  [31, 34, 243, 204]
Training loss:  1.3442665673792362 Testing loss:  1.3798291217535734 Confusion:  [43, 29, 177, 263]
Training loss:  1.3371383384801447 Testing loss:  1.3516062246635556 Confusion:  [52, 18, 142, 300]
Training loss:  1.4031586004421115 Testing loss:  1.3335414826869965 Confusion:  [58, 11, 96, 347]
Training loss:  1.370939995162189 Testing loss:  1.38674187194556 Confusion:  [43, 29, 159, 281]
Training loss:  1.4512228891253471 Testing loss:  1.3377065197564662 Confusion:  [70, 0, 0, 442]
```

![alt text](https://github.com/Neurobiologist/EC601-Pulmonary-Embolism/blob/master/PENet/Train_Graphic.png)
