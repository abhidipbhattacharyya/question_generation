import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter, namedtuple, defaultdict

import pandas as pd


#------from helper.py by maggie-------##
''' i will write some functions. That will eventually go to helper.py'''
def extract_section_text_for_row(fairytale_row):
    source_texts_df = pd.read_csv(source_texts)
    title = fairytale_row['source_title']
    sections = fairytale_row['cor_section'].split(",")
    if len(sections) > 1:
        # If multiple sections are available, combine their text and use as input
        combined_section_text = ""
        for s in sections:
            # Make sure there are no whitespaces in the section
            int_s = int(s.replace(" ", ""))
            source_texts_row = source_texts_df.loc[(source_texts_df['source_title']==title) & (source_texts_df['cor_section'] == int_s)]
            s_text = source_texts_row['text'].iloc[0]
            combined_section_text = combined_section_text + " " + s_text
        section_text = combined_section_text
    else:
        section = sections[0]
        source_texts_row = source_texts_df.loc[(source_texts_df['source_title']==title) & (source_texts_df['cor_section']== int(section))]
        source_text = source_texts_row['text'].iloc[0]
        section_text = source_text

    fairytale_row['section_text'] = section_text
    return fairytale_row

'''encoder text. It will have the story piece and the task desc'''
def format_encoder_text_for_bart(task = "ask_question"):
    input_text = "{}: context: {}" % (task, example["section_text"])
    example["input_text"] = input_text
    return example

'''decoder text. it will have attribute and question or answer'''
def format_decoder_text_for_bart(task = "ask_question"):
    if 'attribute1' in example:
        attribute = example['attribute1']
    else:
        attribute = example['attribute']

    if task == "ask_question":
        decoder_text = "attribute:{} query:{}".format(attribute, example["answer"])
    elif task == "ask_answer":
        decoder_text = "attribute:{} query:{}".format(attribute, example["question"])
    else:
        decoder_text = "attribute:{} query:<s>".format(attribute)
    example['decoder_text'] = decoder_text
    return example

class fairytale_dataset(Dataset):
    def __init__(self,csv_file, tokenizer=None, story_file=None, task = "ask_question"):
        df = pd.read_csv(csv_file)
        cols = list(df.columns)
        self.rw_data = []
        self.tokenizer = tokenizer
        for ind in df.index:
            data_i = {k:df[k][ind] for k in cols}
            data_i =extract_section_text_for_row(data_i)
            data_i =format_encoder_text_for_bart(data_i, task)
            data_i = format_decoder_text_for_bart(data_i, task)
            
            self.rw_data.append(data_i)

    def __len__(self):
        return len(self.rw_data)

    def __getitem__(self, item):
        # image reading

        return self.rw_data[item]

    def collate_fn(self, batch):


if __name__ == '__main__':
    dataset = fairytale_dataset('small_example.csv')
    print(dataset[0])
