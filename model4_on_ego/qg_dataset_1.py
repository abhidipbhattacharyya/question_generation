import os
import json
import torch
import random
from torch.utils.data import Dataset, DataLoader
from collections import Counter, namedtuple, defaultdict
from transformers import BartTokenizer
import pandas as pd
from collections import Counter, namedtuple
import argparse

#------from helper.py by maggie-------##

# Constants
data_dir = '/media/abhidip/2F1499756FA9B1151/data/CU-stroy-QA-Data/'
source_texts = data_dir + 'source_texts.csv'
train_path = data_dir + 'train_split.csv'
val_path = data_dir + 'val_split.csv'
test_path = data_dir + 'test_split.csv'

''' i will write some functions. That will eventually go to preprocessing.py'''
# Remove blank space characters from the source texts
def clean_source_text(raw_text):
    clean_text = raw_text.replace('\n',' ').replace('\r',' ').replace('\t',' ').replace('\s',' ').replace('\t+',' ').replace('\s+',' ').lower().strip(' .') + ' .'
    return clean_text

def extract_section_text_for_row(fairytale_row, source_texts_df):
    #source_texts_df = pd.read_csv(source_texts)
    title = fairytale_row['source_title']
    #print(fairytale_row['cor_section'])
    sections = str(fairytale_row['cor_section']).split(",")
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

    fairytale_row['section_text'] = clean_source_text(section_text)
    return fairytale_row


def format_encoder_text_for_bart(example, task = "ask_question"):
    '''encoder text. It will have the story piece and the task desc.'''
    input_text = "{}: context: {}".format(task, example["section_text"])
    example["input_text"] = input_text
    return example


def format_decoder_text_for_bart(example, task = "ask_question"):
    '''decoder text. it will have attribute and question or answer.'''
    if 'attribute1' in example:
        attribute = example['attribute1']
    else:
        attribute = example['attribute']

    if task == "ask_question":
        decoder_text = "</s><s>{}: <a>{}</a> <q>".format(attribute, example["answer"])
    elif task == "ask_answer":
        decoder_text = "</s><s>{}: <q>{}</q> <a>".format(attribute, example["question"])
    else:
        decoder_text = "</s><s>{}: <q>".format(attribute)
        #decoder_text = "</s><s>{}: <a>{}</a> <q>".format(attribute, example["answer"])#temporary
        #decoder_text = "</s><s>{}: <q>{}</q> <a>".format(attribute, example["question"])#temporary
    example['decoder_text'] = decoder_text
    return example

def format_output_text_for_bart(example, task = "ask_question"):
    '''creating the target.'''
    if task == "ask_question":
        example["output_text"] = "<s>{}: <a>{}</a> <q>{}</q></s>".format(example['attribute1'],example["answer"],example["question"])
    elif task == "ask_answer":
        example["output_text"] = "<s>{}: <q>{}</q> <a>{}</a></s>".format(example['attribute1'],example["question"],example["answer"])
    else:
        random.seed()
        if random.random()>0.5:
            example["output_text"] = "<s>{}: <q>{}</q> <a>{}</a></s>".format(example['attribute1'],example["question"],example["answer"])#"<s>"+ example['attribute1']+": question: "+ example["question"]+" "+ "answer: "+example["answer"]+"</s>"
        else:
            example["output_text"] =  "<s>{}: <a>{}</a> <q>{}</q></s>".format(example['attribute1'],example["answer"],example["question"])#"<s>"+ example['attribute1']+": answer: "+ example["answer"]+" "+ "question: "+example["question"]+"</s>"
    return example


batch_fields = [
    'pair_ids',
    'encoder_ids',
    'encoder_attention_masks',
    'decoder_ids',
    'target_ids',
    'decoder_attention_masks'
]
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))

class QGTensorizer(object):
    ''' to tensorize the inputs'''
    def __init__(self,tokenizer, max_encoder_input_length = 1024, max_decoder_input_length = 128, max_target_length = 128):
        self.tokenizer = tokenizer
        self.max_encoder_input_length =  max_encoder_input_length
        self.max_decoder_input_length = max_decoder_input_length
        self.max_target_length = max_target_length


    def tensorize_example(self, en_txt, de_txt, tar_txt = None):
        en_ids = self.tokenizer(en_txt,  return_tensors='pt',max_length= self.max_encoder_input_length, truncation=True, padding='max_length')
        #self.tokenizer.padding_side = 'left'
        de_ids = self.tokenizer(de_txt,  return_tensors='pt',max_length= self.max_decoder_input_length, truncation=True, padding='max_length',add_special_tokens = False)
        tar_ids = None
        #self.tokenizer.padding_side = 'right'
        if tar_txt:
            tar_ids = self.tokenizer(text_target=tar_txt,  return_tensors='pt',max_length= self.max_target_length, truncation=True, padding='max_length', add_special_tokens = False)
        #self.tokenizer.padding_side = 'right'
        return en_ids, de_ids, tar_ids

''' dataset with collate_fn'''
class fairytale_dataset(Dataset):
    def __init__(self,csv_file, tokenizer=None, istraining=True, max_encoder_input_length = 1024, max_decoder_input_length = 128, max_target_length = 128, story_file=None, task = "ask_question"):
        df = pd.read_csv(csv_file)
        cols = list(df.columns)
        self.rw_data = []
        self.tokenizer = tokenizer
        self.istraining = istraining
        self.task = task
        self.tensorizer = QGTensorizer(tokenizer, max_encoder_input_length, max_decoder_input_length, max_target_length)
        source_texts_df = pd.read_csv(source_texts)
        print("self.istraining:{}".format(self.istraining))
        self.mil = max_encoder_input_length
        self.mdil= max_decoder_input_length
        self.mtil = max_target_length
        for ind in df.index:
            data_i = {k:df[k][ind] for k in cols}
            data_i = extract_section_text_for_row(data_i, source_texts_df)
            data_i = format_encoder_text_for_bart(data_i, self.task)
            data_i = format_decoder_text_for_bart(data_i, self.task)
            #data_i = format_output_text_for_bart(data_i, self.task)
            self.rw_data.append(data_i)

    def __len__(self):
        return len(self.rw_data)

    @property
    def pair_ids(self):
        return [data_item['pair_id'] for data_item in self.rw_data]
    def __getitem__(self, item):
        data_item = self.rw_data[item]
        if self.istraining:
            tar_txt = format_output_text_for_bart(data_item, self.task)["output_text"]
        else:
            tar_txt = None
        #print(data_item)
        #print([data_item["input_text"],data_item['decoder_text'], tar_txt])
        #en_ids, de_ids, tar_ids=self.tensorizer.tensorize_example(data_item["input_text"],data_item['decoder_text'], tar_txt = tar_txt)
        #encoder_id = en_ids['input_ids'].squeeze(0)
        #encoder_att = en_ids['attention_mask'].squeeze(0)
        #decoder_id = de_ids['input_ids'].squeeze(0)
        #print(encoder_id.size())
        #if tar_txt:
            #target_id = tar_ids['input_ids'].squeeze(0)
        #else:
            #target_id = None

        eitem = {'pair_id':data_item['pair_id'],
                'encoder_id':data_item["input_text"],
                #'encoder_attention_mask':encoder_att,
                'decoder_id':data_item['decoder_text'],
                #'decoder_attention_mask':de_ids['attention_mask'].squeeze(0),#think about it
                'target_id':tar_txt
                }
        return eitem

    def collate_fn(self, batch):

        pair_ids = [b['pair_id'] for b in batch]
        encoder_ids = [b['encoder_id'] for b in batch]
        decoder_ids =[b['decoder_id'] for b in batch]
        target_ids = [b['target_id'] for b in batch]

        encoder_ids = self.tokenizer(encoder_ids,  return_tensors='pt', max_length=self.mil, truncation=True, padding=True)
        decoder_ids = self.tokenizer(decoder_ids,  return_tensors='pt', max_length=self.mdil, truncation=True, padding=True,add_special_tokens = False)
        target_ids = self.tokenizer(text_target=target_ids,  return_tensors='pt', max_length=self.mtil, truncation=True, padding=True, add_special_tokens = False)


        return Batch(
                pair_ids = pair_ids,
                encoder_ids= encoder_ids['input_ids'],
                encoder_attention_masks=encoder_ids['attention_mask'],
                decoder_ids=decoder_ids['input_ids'],
                target_ids=target_ids['input_ids'],
                decoder_attention_masks=decoder_ids['attention_mask']
            )
'''
    def collate_fn(self, batch):
        encoder_ids = []
        encoder_attention_masks = []
        decoder_ids = []
        target_ids = []
        pair_ids = []
        decoder_attention_masks = []
        if self.istraining:
            target_ids = []

        mil=0
        mdil=0
        mtil=0
        for b in batch:
            il = len(b['encoder_id'].split(" "))+50# adding to handle estimation errs made by space counting
            dl = len(b['decoder_id'].split(" "))+5
            tl = len(b['target_id'].split(" "))+10
            if il>mil:
                mil=il
            if dl>mdil:
                mdil=dl
            if tl>mtil:
                mtil=tl

        mil = min(mil,self.mil)
        mdil= min(mdil,self.mdil)
        mtil = min(mtil, self.mtil)
        for b in batch:
            pair_ids.append(b['pair_id'])
            en_id =  self.tokenizer(b['encoder_id'],  return_tensors='pt',max_length= mil, truncation=True, padding='max_length')
            encoder_ids.append(en_id['input_ids'].squeeze(0))
            encoder_attention_masks.append(en_id['attention_mask'].squeeze(0))

            de_id= self.tokenizer(b['decoder_id'],  return_tensors='pt',max_length= mdil, truncation=True, padding='max_length',add_special_tokens = False)
            decoder_ids.append(de_id['input_ids'].squeeze(0))
            decoder_attention_masks.append(de_id['attention_mask'].squeeze(0))
            if self.istraining:
                tar_id = self.tokenizer(text_target=b['target_id'],  return_tensors='pt',max_length= mtil, truncation=True, padding='max_length', add_special_tokens = False)
                target_ids.append( tar_id['input_ids'].squeeze(0))


        encoder_ids = torch.stack(encoder_ids, dim=0)
        decoder_ids = torch.stack(decoder_ids, dim=0)
        encoder_attention_masks = torch.stack(encoder_attention_masks, dim=0)
        decoder_attention_masks = torch.stack(decoder_attention_masks, dim=0)
        if self.istraining:
            target_ids = torch.stack(target_ids, dim=0)

        return Batch(
                pair_ids = pair_ids,
                encoder_ids= encoder_ids,
                encoder_attention_masks=encoder_attention_masks,
                decoder_ids=decoder_ids,
                target_ids=target_ids,
                decoder_attention_masks=decoder_attention_masks
            )

'''
def build_dataset(args,  tokenizer, is_train=True):
    dataset = fairytale_dataset(args.csv_file,
                tokenizer=tokenizer,
                istraining=is_train,
                max_encoder_input_length = args.max_encoder_input_length,
                max_decoder_input_length = args.max_decoder_input_length,
                max_target_length = args.max_target_length,
                task = args.task)
    return dataset

if __name__ == '__main__':
    tokenizer = BartTokenizer.from_pretrained("/media/abhidip/2F1499756FA9B1151/QG/model/bart-base/")
    dataset = fairytale_dataset(train_path,
        tokenizer=tokenizer,
        max_encoder_input_length = 1024,
        max_decoder_input_length = 128,
        max_target_length = 128,
        story_file=None,
        task = "ask_question"
    )
    #print(dataset[0])
    loader = DataLoader(
        dataset, batch_size=4,
        collate_fn=dataset.collate_fn,
        pin_memory=False
    )
    print(len(dataset))
    for batch_idx, batch in enumerate(loader):
        encoder_ids= batch.encoder_ids
        encoder_attention_masks=batch.encoder_attention_masks
        decoder_ids=batch.decoder_ids
        target_ids=batch.target_ids
        print(batch.pair_ids)
        print(encoder_ids.size())
        print(encoder_attention_masks.size())
        print(decoder_ids.size())
        print(target_ids.size())
        print("----------------------")
        break

    #print(dataset.rw_data[0])
    #print(dataset[0])
