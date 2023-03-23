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
        ret_dict.setdefault(id, decoder_op)

    return ret_dict


def reading_tsefile(csv_file, pred_dict):
    df = pd.read_csv(csv_file)
    alldata = []
    for ind in df.index:
        data_i = {k:df[k][ind] for k in cols}
        gendata = pred_dict[data_i["pair_id"]]
        data_i["gen_data"] = gendata
        alldata.append(data_i)

    return alldata


def evaluate(alldata, task='ask_question'):
    references = []
    predictions= []
    rouge = evaluate.load('rouge')
    if task == 'ask_question':
        col = 'question'
    else:
        col= 'answer'

    for d in alldata:
        references.append( d[col])
        predictions.append(d["gen_data"])

    result = rouge.compute(predictions=predictions, references=references)
    print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict_file', default='/media/abhidip/2F1499756FA9B1151/QG/model/predict_m4_bothTrModel.csv',
                        help='Path to save the generated op.')
    args = parser.parse_args()
