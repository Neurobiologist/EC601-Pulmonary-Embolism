import util
import numpy as np
import pandas as pd
import os

data_dir = '/projectnb/ece601/kaggle-pulmonary-embolism/rsna-str-pulmonary-embolism-detection/'
train_csv = data_dir + 'train.csv'
train_dir = data_dir + 'train/'

# get list of unqieu studies
pedataframe = pd.read_csv(train_csv)
studies = pedataframe.StudyInstanceUID.unique()
print('Total number of studies to process: ' + str(len(studies)))

# This is the directory where I will put the preprocessed 3D .npy files
out_traindir = '/projectnb/ece601/kaggle-pulmonary-embolism/rsna-str-pulmonary-embolism-detection-208-npy/train/'

i = 0
for study in studies:
    study_path = train_dir + study
    files = os.listdir(study_path)
    if len(files) != 1:
        print("Error: more than one images in study!.")

    image_dir = study_path + '/' + files[0] + '/'
    
    # preprocess
    study_arr = util.dicom_2_npy(image_dir, 'CTPA')
    study_preprocessed = util.preprocess_img(study_arr)
    
    #save 3D numpy array
    with open(out_traindir + '{}.npy'.format(study),'wb') as f:
        np.save(f, study_preprocessed)

    i+=1
    if i%100 == 0:
        print('Processed {} studies'.format(i))
