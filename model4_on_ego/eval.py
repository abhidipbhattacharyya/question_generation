import os
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import Counter, namedtuple, defaultdict
from transformers import BartTokenizer
import pandas as pd
from collections import Counter, namedtuple
import evaluate
import argparse
from datasets import load_metric
# Constants
data_dir = '/media/abhidip/2F1499756FA9B1151/data/CU-stroy-QA-Data/'
source_texts = data_dir + 'source_texts.csv'
train_path = data_dir + 'train_split.csv'
val_path = data_dir + 'val_split.csv'
test_path = data_dir + 'test_split.csv'

def read_prediction(filename):
    ret_dict = {}
    with open(filename,'r') as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines]
    for l in lines:
        info = l.split("|||")
        id = info[0].strip()
        decoder_ip= info[1].strip()
        decoder_op= info[2].strip()
        ret_dict.setdefault(int(id), decoder_op)

    return ret_dict


def reading_GTfile(csv_file, pred_dict):
    df = pd.read_csv(csv_file)
    cols = list(df.columns)
    alldata = []
    for ind in df.index:
        data_i = {k:df[k][ind] for k in cols}
        #print(data_i)
        gendata = pred_dict[data_i["pair_id"]]
        data_i["gen_data"] = gendata
        alldata.append(data_i)

    return alldata


def compute_metric_for_results(eval_pred, metric_string):
    preds = eval_pred[0]
    labels = eval_pred[1]
    metric = load_metric(metric_string)
    result = metric.compute(predictions=preds, references=labels)

    return result


def evaluate_metric(alldata, task='ask_question', metric = 'rouge'):
    #print('hete')
    references = []
    predictions= []
    rouge = evaluate.load(metric)
    if task == 'ask_question':
        col = 'question'
    else:
        col= 'answer'

    for d in alldata:
        references.append( d[col])
        predictions.append(d["gen_data"])
        print('{}\t{}'.format(d[col], d["gen_data"]))

    result = rouge.compute(predictions=predictions, references=references)
    eval_pred=predictions,references
    #result = compute_metric_for_results(eval_pred, metric)
    print(result)


def evaluate_metric_by_attr(alldata, task='ask_question', attr='', metric = 'rouge'):
    references = []
    predictions= []
    #print(evaluate.list_evaluation_modules())
    rouge = evaluate.load(metric)
    if task == 'ask_question':
        col = 'question'
    else:
        col= 'answer'

    for d in alldata:
        if d['attribute1'] == attr:
            references.append( d[col])
            predictions.append(d["gen_data"])
            #print('{}\t{}'.format(d[col], d["gen_data"]))

    result = rouge.compute(predictions=predictions, references=references)
    eval_pred=predictions,references
    #result = compute_metric_for_results(eval_pred, metric)
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_file', default='/media/abhidip/2F1499756FA9B1151/QG/model/model4_batch_padding_better/ask_question/predict_m4.csv',
                        help='Path to save the generated op.')
    parser.add_argument('--csv_file', default='/media/abhidip/2F1499756FA9B1151/data/CU-stroy-QA-Data/test_split.csv',
                        help='path to datasets')
    parser.add_argument('--metric', default='rouge',
                        help='bleu|rouge| check evaluate.list_evaluation_modules()')
    args = parser.parse_args()

    pred_data = read_prediction(args.predict_file)
    #print(pred_data)
    alldata =reading_GTfile(args.csv_file, pred_data)
    attrs = ['action', 'feeling', 'setting', 'character', 'causal relationship', 'outcome resolution', 'prediction']
    evaluate_metric(alldata, metric =args.metric)
    #print(attr)
    #evaluate_metric_by_attr(alldata, task='ask_question', attr='prediction', metric = 'rouge')
