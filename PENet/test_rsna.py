import cv2
import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as sk_metrics
import torch
import util
import pandas as pd
import os
import sys
sys.path.append(os.getcwd())

from args import TestArgParser
from data_loader import CTDataLoader
from collections import defaultdict
from logger import TestLogger
from PIL import Image
from saver import ModelSaver
from tqdm import tqdm
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score

def test(args):
    print ("Stage 1")
    model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)
    print ("Stage 2")
    args.start_epoch = ckpt_info['epoch'] + 1
    model = model.to(args.device)
    print ("Stage 3")
    model.eval()
    print ('This should be false: {}'.format(model.training))
    print ("Stage 4")
    data_loader = CTDataLoader(args, phase=args.phase, is_training=False)
    #print('data_loader={}'.format(data_loader))
    #print('data_loader.dataset={}'.format(data_loader.dataset))
    study2slices = defaultdict(list)
    study2probs = defaultdict(list)
    study2labels = {}
    logger = TestLogger(args, len(data_loader.dataset), data_loader.dataset.pixel_dict)
    print("Stage 5")
    f = open('/projectnb/ece601/kaggle-pulmonary-embolism/meganmp/train/series_list.pkl','rb')
    data_labels = pickle.load(f)

    # Create list to manually process labels
    #with open('positive.txt') as f:
         #pos_labels = f.readlines()
    #pos_labels = [x.strip() for x in pos_labels]
    ispos = [x.is_positive for x in data_labels]
    isposidx = [x.study_num for x in data_labels]
    label_dict = {}
    for i in range(len(ispos)):
        label_dict[isposidx[i]] = ispos[i]

    for key in label_dict.keys():
        print('label_dict={}\t{}'.format(key, label_dict[key]))
    

    # Get model outputs, log to TensorBoard, write masks to disk window-by-window
    util.print_err('Writing model outputs to {}...'.format(args.results_dir))
    with tqdm(total=len(data_loader.dataset), unit=' windows') as progress_bar:
        for i, (inputs, targets_dict) in enumerate(data_loader):
            with torch.no_grad():
                cls_logits = model.forward(inputs.to(args.device))
                cls_probs = torch.sigmoid(cls_logits)

            if args.visualize_all:
                logger.visualize(inputs, cls_logits, targets_dict=None, phase=args.phase, unique_id=i)

            max_probs = cls_probs.to('cpu').numpy()
            for study_num, slice_idx, prob in \
                    zip(targets_dict['study_num'], targets_dict['slice_idx'], list(max_probs)):
                #print('targets_dict[studynum]={}'.format(targets_dict['study_num']))
                #print('targets_dict[sliceidx]={}'.format(targets_dict['slice_idx']))
                # Convert to standard python data types
                study_num = study_num #.item()
                #study_num = int(study_num)
                slice_idx = int(slice_idx)

                # Save series num for aggregation
                study2slices[study_num].append(slice_idx)
                study2probs[study_num].append(prob.item())


                series = data_loader.get_series(study_num)
                if study_num not in study2labels:
                    print('study_num={}'.format(study_num))
                    print('series.is_positive={}'.format(label_dict[study_num]))
                    study2labels[study_num] = label_dict[study_num]
                    #if study_num in pos_labels:
                        #print('DEBUG -------=1?-------------------')
                        #print('POS LABEL')
                        #print('study_num={}'.format(study_num))
                        #study2labels[study_num] = 1
                    #else:
                        #print('Not in study2labels. series = {}'.format(study_num))
                        #print('series.is_positive={}'.format(series.is_positive))
                        #study2labels[study_num] = int(series.is_positive)
                        #print('study2labels: {}'.format(study2labels[study_num]))

            progress_bar.update(inputs.size(0))

    print('study2labels={}'.format(study2labels))

    # Combine masks
    util.print_err('Combining masks...')
    max_probs = []
    labels = []
    predictions = {}
    print("Get max prob")
    for study_num in tqdm(study2slices):

        # Sort by slice index and get max probability
        slice_list, prob_list = (list(t) for t in zip(*sorted(zip(study2slices[study_num], study2probs[study_num]),
                                                              key=lambda slice_and_prob: slice_and_prob[0])))
        study2slices[study_num] = slice_list
        study2probs[study_num] = prob_list
        max_prob = max(prob_list)
        print('study={}\tmax_prob={}'.format(study_num, max_prob))
        max_probs.append(max_prob)
        label = study2labels[study_num]
        labels.append(label)
        predictions[study_num] = {'label':label, 'pred':max_prob}

    #Save predictions to file, indexed by study number
    print("Saving predictions to pickle files")
    with open('{}/preds.pickle'.format(args.results_dir),"wb") as fp:
        pickle.dump(predictions,fp)

    results_series = [k for k,_ in predictions.items()]
    results_pred = [v['pred'] for _,v in predictions.items()]
    results_label = [v['label'] for _,v in predictions.items()]
    print('roc_auc_score={}'.format(roc_auc_score(results_label, results_pred)))

    # Create dataframe summary
    TRAIN_CSV = '/projectnb/ece601/kaggle-pulmonary-embolism/rsna-str-pulmonary-embolism-detection/train.csv'
    train_df = pd.read_csv(TRAIN_CSV)
    train_df = train_df[['SeriesInstanceUID', 'negative_exam_for_pe']]
    train_df = train_df.groupby('SeriesInstanceUID').aggregate(list)
    train_df['pe_label'] = train_df['negative_exam_for_pe'].apply(lambda x: 0 if 1 in x else 1)

    results_dict = {
        'series': results_series,
        'pred': results_pred
    }
    results_df = pd.DataFrame.from_dict(results_dict)

    results_df = results_df.set_index('series')
    results_df = results_df.join(train_df, how='left').reset_index().rename({'index': 'series'})
    print('roc_auc_score={}'.format(roc_auc_score(results_df['pe_label'], results_df['pred'])))
    
    # Calculate confusion matrix
    results_df['interpretation'] = results_df['pred'].apply(lambda x: 0 if x < 0.5 else 1)
    print(results_df.head(10))
    tn, fp, fn, tp = confusion_matrix(results_df['pe_label'], results_df['interpretation']).ravel()
    print('confusion_matrix: [{} {} {} {}]'.format(tp, fp, fn, tn))

    
def save_for_xgb(results_dir, series2probs, series2labels):
    """Write window-level and series-level features to train an XGBoost classifier.
    Args:
        results_dir: Path to results directory for writing outputs.
        series2probs: Dict mapping series numbers to probabilities.
        series2labels: Dict mapping series numbers to labels.
    """

    # Convert to numpy
    xgb_inputs = np.zeros([len(series2probs), max(len(p) for p in series2probs.values())])
    xgb_labels = np.zeros(len(series2labels))
    for i, (series_num, probs) in enumerate(series2probs.items()):
        xgb_inputs[i, :len(probs)] = np.array(probs).ravel()
        xgb_labels[i] = series2labels[series_num]

    # Write to disk
    os.makedirs(os.path.join(results_dir, 'xgb'), exist_ok=True)
    xgb_inputs_path = os.path.join(results_dir, 'xgb', 'inputs.npy')
    xgb_labels_path = os.path.join(results_dir, 'xgb', 'labels.npy')
    np.save(xgb_inputs_path, xgb_inputs)
    np.save(xgb_labels_path, xgb_labels)


if __name__ == '__main__':
    util.set_spawn_enabled()
    parser = TestArgParser()
    args_ = parser.parse_args()
    test(args_)
