import fasttext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from sklearn.model_selection import KFold
from os import listdir
from os.path import isfile, join
import json

#load paths
config = json.load(open('config.json', 'r'))

def tok_sentence(sent, model):
    
    veclist=[]
    wordlist=sent.split()
    for w in wordlist:
        veclist.append(model[w])
    veclist=np.array(veclist).T
    
    veclist=np.mean(veclist, axis = 1)

    return veclist

def main():

    clopid=pd.read_csv(config['clopid_paraphrased'])
    anemia=pd.read_csv(config['anemia_paraphrased'])
    pb=pd.read_csv(config['proton_beam_paraphrased'])
    unja=pd.read_csv(config['hyperglycemia_paraphrased'])
    cohen=pd.read_csv(config['cohen_paraphrased'])




    datasets=pd.DataFrame()
    path=config['path_to_percentage_datasets']
    for f in listdir(path):
        print(f)
        datasets=pd.concat([datasets, pd.read_csv(path +'/'+ f, index_col=0)]) 




    datasets=pd.concat([clopid, anemia])
    datasets=pd.concat([datasets, clopid])
    datasets=pd.concat([datasets, pb])
    datasets=pd.concat([datasets, unja])
    datasets=pd.concat([datasets, cohen])



    combined_string=datasets.anchor.str.cat(sep=', ')




    text_file = open("augmented_percent.txt", "w")
    n=text_file.write(combined_string)
    text_file.close()

    # Skipgram model :
    model = fasttext.train_unsupervised('augmented_percent.txt', model='skipgram')




    print(len(model['findings'])) # get the vector of the word 'findings'



    data_vec=pd.DataFrame()

    data_vec['pos_vec']=datasets['positive'].apply(tok_sentence)
    data_vec['neg_vec']=datasets['negative'].apply(tok_sentence)
    data_vec['anc_vec']=datasets['anchor'].apply(tok_sentence)
    data_vec['pmid']=datasets['pmid']
    data_vec=data_vec.reset_index(drop=True)
    lis=[]

    for row in tqdm(data_vec.index) :
        vec=np.append(data_vec.loc[row, 'pmid'], data_vec.loc[row, 'pos_vec'])
        vec=np.append(vec, data_vec.loc[row, 'neg_vec'])
        vec=np.append(vec, data_vec.loc[row, 'anc_vec'])
        lis.append(vec)
    lis=pd.DataFrame(lis)
    lis=lis.set_index(0)
    lis.index=lis.index.astype(int)


    lis.to_csv(config['fasttext_output'])

    return 0

