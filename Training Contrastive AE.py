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
import json

#load paths
config = json.load(open('config.json', 'r'))

# batch generator
def get_batches(dataset, batch_size):
    X= dataset
    n_samples = X.shape[0]

    # Shuffle at the start of epoch
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        batch_idx = indices[start:end]

        yield X.iloc[batch_idx]


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=100.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        
        return (x1 - x2).pow(2).sum()
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        #distance might be too similar
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        #print('distance_positive:',distance_positive, 'distance_negative:', distance_negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        #print('loss:', losses)
        return losses.mean()

def initialize_weight(m):
        torch.nn.init.xavier_uniform_(m.weight.data, gain=torch.nn.init.calculate_gain('relu'))
        
class AE(torch.nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.encoder_hidden_layer = torch.nn.Linear(
            in_features=input_shape, out_features= output_shape
        )
        self.encoder_output_layer = torch.nn.Linear(
            in_features=output_shape, out_features=output_shape
        )
        self.decoder_hidden_layer = torch.nn.Linear(
            in_features=output_shape, out_features=output_shape
        )
        self.decoder_output_layer = torch.nn.Linear(
            in_features=output_shape, out_features=input_shape
        )
        initialize_weight(self.encoder_hidden_layer)
        initialize_weight(self.encoder_output_layer)
        initialize_weight(self.decoder_hidden_layer)
        initialize_weight(self.decoder_output_layer)
        
    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed
    
    def reduce(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        return code

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def main():
    dataset=pd.read_csv(config['vectorized_path'])

    model = torch.load('fasttext_AE_2')
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

    criterion = torch.nn.TripletMarginLoss()


    model = AE(input_shape=768, output_shape=768)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-7)

    criterion = torch.nn.TripletMarginLoss()


    anchor=dataset.iloc[:, 2*768:3*768]
    negative=dataset.iloc[:, 768:2*768]
    positive=dataset.iloc[:, 0:768]

    anchor=torch.tensor(anchor.values, dtype=torch.float)
    positive=torch.tensor(positive.values, dtype=torch.float)
    negative=torch.tensor(negative.values, dtype=torch.float)

    early_stopping=EarlyStopping(patience = 10, verbose=True, delta=1e-4)

    for epoch in tqdm(range(200)):
        loss=0
        
        #batches=get_batches(dataset, int(dataset.shape[0]/500))
        #for xtrain in batches:
        optimizer.zero_grad()
        for i, sample in enumerate(anchor): 
            anc=sample
            pos=positive[i]
            neg=negative[i]

            anc_outputs = model(anc)
            pos_outputs = model(pos)
            neg_outputs = model(neg)

            train_loss = criterion(anc_outputs, pos_outputs, neg_outputs)
            train_loss.backward()
            optimizer.step()
            loss+=train_loss.item()
        print('loss:', loss)
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print('Early Stopping')
            break

    anchor=dataset.iloc[:, 1536:2304]
    negative=dataset.iloc[:, 768:1536]
    positive=dataset.iloc[:, 0:768]

    anchor=torch.tensor(anchor.values, dtype=torch.float)
    positive=torch.tensor(positive.values, dtype=torch.float)
    negative=torch.tensor(negative.values, dtype=torch.float)



    torch.save(model, config['ae_path'])


