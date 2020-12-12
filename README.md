# EC601-Pulmonary-Embolism
This repository contains our contributions to the Kaggle competition [Pulmonary Embolism Detection](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection).

Folder | Contents 
--- | --- 
**CNN-LSTM-Model** | Code and images on CNN_LSTM Model to detect PE
**Model** | Trained model for CNN_LSTM Model
**PENet** | Code and information on experiments with [PENet](https://github.com/marshuang80/penet) [[1]](#1)
**Website** | Code for the physician interface
**Documents** | Data Exploration, Literature Review, Sprint #1 #2 #3 #4 #5 and Poster

## Development of a Deep Learning Model to Diagnose Pulmonary Embolism
A pulmonary embolism (PE) is a potentially life-threatening obstruction of the pulmonary artery. The diverse clinical presentation and symptomatology of PE can pose challenges in prompt and accurate diagnosis, and complications can rapidly escalate in severity. The goal of this project was to develop a deep learning model using the RSNA-STR Pulmonary Embolism CT (RSPECT) Dataset to enable the accurate automatic identification of PE in computed tomography pulmonary angiogragphy (CTPA) images [[2]](#2). CTPA is currently the gold standard method of diagnosis for PE, but the size and complexity of the imaging data can lead to human error or delays in diagnosis. Advancements in the automated diagnosis of PE have the potential to expedite diagnosis, improve accuracy of PE detection, and improve patient outcomes.

## References
<a id="1">[1]</a> Huang, S. C., Kothari, T., Banerjee, I., Chute, C., Ball, R. L., Borus, N., ... & Dunnmon, J. (2020). PENet—a scalable deep-learning model for automated diagnosis of pulmonary embolism using volumetric CT imaging. npj Digital Medicine, 3(1), 1-9.

<a id="2">[2]</a> RSNA-STR Pulmonary Embolism CT (RSPECT) Dataset, Copyright RSNA, 2020: https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-pe-detection-challenge-2020
