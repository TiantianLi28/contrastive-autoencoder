import torch
from random import sample
from random import choice
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from transformers import BartForConditionalGeneration, AutoTokenizer
import json

#load paths
config = json.load(open('config.json', 'r'))

def main():
    sentenceTrans = SentenceTransformer('paraphrase-albert-small-v2')


    tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")

    # by default encoder-attention is `block_sparse` with num_random_blocks=3, block_size=64
    model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")


    cohen=pd.read_csv(config['cohen_path'], index_col=0)


    clopid=pd.read_csv(config['clopid_path'])


    clopid.drop(['MeSH terms'], axis=1, inplace=True)


    clopid.dropna(inplace=True)


    anemia=pd.read_csv(config['anemia_path'])


    anemia.drop(['MeSH terms'], axis=1, inplace=True)


    anemia.dropna(inplace=True)


    proton_beam=pd.read_csv(config['proton_beam_path'])


    proton_beam.drop(['MeSH terms'], axis=1, inplace=True)


    proton_beam.dropna(inplace=True)


    unaj=pd.read_excel(config['hyperglycemia_path'])


    unaj=unaj[['Article ID', 'Title', 'Abstract', 'Decision (Y/N)']]



    unaj.dropna(inplace=True)


    unaj=unaj.rename({'Article ID': 'pmid','Abstract':'abstract'}, axis=1)


    dataset=clopid
    aug=[]
    for abst in dataset.index:
        ori=dataset.loc[abst, 'abstract']
        pos=ori.split(". ")
        #sample from positive pool
        sample_size = int(len(pos)*0.7)
        pos = [pos[i] for i in sorted(sample(range(len(pos)), sample_size))]
        
        pos='. '.join(pos)
        #choose random abstract other than anchor
        neg_idx=choice(dataset.index)
        while neg_idx == abst:
            neg_idx=choice(dataset.index)
        neg=dataset.loc[neg_idx, 'abstract']
        neg=neg.split(". ")
        #sample form negative pool
        sample_size = int(len(neg)*0.7)
        neg = [neg[i] for i in sorted(sample(range(len(neg)), sample_size))]
        
        neg='. '.join(neg)
        aug.append([dataset.loc[abst, 'pmid'], ori, pos, neg])

    aug=pd.DataFrame(aug)
    aug.columns=['pmid', 'anchor', 'positive', 'negative']
    aug=aug.set_index('pmid')


    aug = aug[~aug.index.duplicated(keep='first')]


    #tokenize augmented abstract dataset
    aug_tok=[]
    for i in tqdm(aug.index):
    
        aug_tok.append([i, tokenizer(aug.loc[i, 'anchor'], return_tensors='pt'), tokenizer(aug.loc[i, 'positive'], return_tensors='pt'), tokenizer(aug.loc[i, 'negative'], return_tensors='pt')])
    
    for i in range(len(aug_tok)):
        for j in range(1, 4):
            if len(aug_tok[i][j]['input_ids'][0]) >1024:
                aug_tok[i][j]['input_ids']=aug_tok[i][j]['input_ids'][:, :1024]
                aug_tok[i][j]['attention_mask']=aug_tok[i][j]['attention_mask'][:, :1024]

    aug_paraphrased=[]
    for i in tqdm(range (len(aug_tok))): 
        aug_paraphrased.append([model.generate(**aug_tok[i][1]), model.generate(**aug_tok[i][2]), model.generate(**aug_tok[i][3]), aug_tok[i][0]])
        
    aug_df=pd.DataFrame(aug_paraphrased)

    list_aug=aug_df.values.tolist()

    paraphrased = [[tokenizer.batch_decode(list_aug[i][0]), tokenizer.batch_decode(list_aug[i][1]), 
                    tokenizer.batch_decode(list_aug[i][2]), list_aug[i][3]]for i in range(len(list_aug))]


    paraphrased=pd.DataFrame(paraphrased, columns=['anchor', 'positive', 'negative', 'pmid'])

    paraphrased.to_csv(config['vectorized_path'])

    return 0





