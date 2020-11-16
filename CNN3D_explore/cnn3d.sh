#!/bin/bash -l

#$ -l h_rt=10:00:00   # Specify the hard time limit for the job
#$ -N train           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m e
#$ -l gpus=1
#$ -l gpu_c=7.0

cp /projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/rsna-str-pulmonary-embolism-detection-265-jpeg.zip /scratch
cd /scratch
unzip -q -o rsna-str-pulmonary-embolism-detection-265-jpeg.zip
cd /projectnb/ece601/kaggle-pulmonary-embolism/jiamingy/notebook

module load python3
module load torch
module load pytorch
python3 cnn3d.py
echo done
