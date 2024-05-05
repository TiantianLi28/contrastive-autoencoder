import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from collections import defaultdict
import json

#load paths
config = json.load(open('config.json', 'r'))

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


class SoftmaxRegression(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
        # initialize weights to zeros
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()
        
        
    def forward(self, x):
        logits = self.linear(x)
        # using softmax to calculate probabilities
        probas = F.softmax(logits, dim=1)
        return logits, probas
    
    def evaluate(self, x, y):
        labels = self.predict_labels(x).float()
        scores = self.predict_probas(x).float().detach().numpy()[:,1]
        
        accuracy = torch.sum(labels.view(-1) == y.float()).item() / y.size(0)
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = roc_auc_score(y, scores)
        self.plot_roc_curve(fpr, tpr, roc_auc)
        
        return accuracy, roc_auc
    
    def predict_labels(self, x):
        logits, probas = self.forward(x)
        labels = torch.argmax(probas, dim=1)
        return labels    
    
    def predict_probas(self, x):
        logits, probas = self.forward(x)
        #print(logits)
        return probas    

    def plot_roc_curve(self, fpr, tpr, roc):
        label = "auc=" + str(round(roc, 4))
        plt.plot(fpr, tpr, linewidth=2, label=label)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.title("ROC Curve")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc=4)
        plt.show()


def comp_accuracy(true_labels, pred_labels):
    accuracy = torch.sum(true_labels.view(-1).float() == 
                         pred_labels.float()).item() / true_labels.size(0)
    return accuracy


def def_value():
    return 0

def train_val_test(X_train, y_train, X_val, y_val, X_test, y_test):    
    numF=768
    acclis=defaultdict(def_value)
    auclis=defaultdict(def_value)
    lrlis=[0.001,0.01,0.1,1,10]
    epoch_lis={0.001: 5000, 0.01: 500, 0.1: 100, 1:100, 10:100}

    for lrate in lrlis:

        pred_model = SoftmaxRegression(num_features=numF, num_classes=2)
        optimizer = torch.optim.SGD(pred_model.parameters(), lr=lrate)
        num_epochs=epoch_lis[lrate]
        
        
        for epoch in range(num_epochs):
            #### Compute outputs ####
            logits, probas = pred_model(X_train)

            #### Compute gradients ####
            cost = F.cross_entropy(logits, y_train.long())
            optimizer.zero_grad()
            cost.backward()

            #### Update weights ####  
            optimizer.step()

            #### Logging ####      
            logits, probas = pred_model(X_train)
            acc = comp_accuracy(y_train, torch.argmax(probas, dim=1))

        val_acc, val_auc = pred_model.evaluate(X_val, y_val)
        acclis[lrate]=val_acc
        auclis[lrate]=val_auc

    bestlr=max(auclis, key=auclis.get)
    print(bestlr)
    pred_model = SoftmaxRegression(num_features=numF, num_classes=2)
    optimizer = torch.optim.SGD(pred_model.parameters(), lr=bestlr)
    num_epochs = epoch_lis[bestlr]

    for epoch in range(num_epochs):

        #### Compute outputs ####
        logits, probas = pred_model(X_train)

        #### Compute gradients ####
        cost = F.cross_entropy(logits, y_train.long())
        optimizer.zero_grad()
        cost.backward()

        #### Update weights ####  
        optimizer.step()

        #### Logging ####      
        logits, probas = pred_model(X_train)
        acc = comp_accuracy(y_train, torch.argmax(probas, dim=1))

    val_acc, val_auc = pred_model.evaluate(X_test, y_test)
    yhat=pred_model.predict_probas(X_test).float().detach().numpy()[:,1]
    wssvalue=wss(yhat.tolist(),y_test.tolist())
    testauclis=val_auc
    '''
    fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test,pred_model.predict_labels(X_test).detach().numpy()),
                                    show_absolute=True,show_normed=True,colorbar=True)
    plt.title("Confusion matrix ")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    '''
    return testauclis, wssvalue

def wss(yhat, ytrue):
   
    totalPositive=sum(ytrue)
    
    tp=0
    tn=0
    
    yhat=pd.DataFrame(yhat, columns=['prob'])
    ytrue=pd.DataFrame(ytrue, columns=['label'])
    yhat=yhat.sort_values(by=['prob'], ascending=False)
    #print(yhat)
    rank=0
    for i in yhat.index:
        # if document at position = index is positive (i.e., equal to 1.0) then
        # increment the number of true positives
        if ytrue.loc[i,'label'] == 1.0:
            tp += 1
        else:
            tn += 1
        recall = tp / totalPositive
        if recall >= 0.95:
            last_pos_doc = rank + 1
            precision = tp / (tp + tn)
            #print(f"precision at 0.95 recall = {precision}")
            wss = (float(yhat.shape[0] - last_pos_doc) / float(yhat.shape[0])) - 0.05
            #print(wss)
            return wss
        rank+=1
   


def main():

    clopid=pd.read_csv(config['clopid_path'])
    anemia=pd.read_csv(config['anemia_path'])
    proton_beam=pd.read_csv(config['proton_beam_path'])
    hyperglycemia=pd.read_excel(config['hyperglycemia_path'])
    hyperglycemia=hyperglycemia.rename({'Article ID': 'pmid'}, axis=1)

    lis=pd.read_csv(config['vectorized_path'])

    lis=lis.set_index('0')

    lis=lis[~lis.index.duplicated(keep='first')]

    model_cp=lis[lis.index.isin(clopid.pmid)]
    model_an=lis[lis.index.isin(anemia.pmid)]
    model_pb=lis[lis.index.isin(proton_beam.pmid)]
    model_un=lis[lis.index.isin(hyperglycemia.pmid)]

    ###changing -1 to 0 
    anemia['label']=anemia.apply(lambda x: 1.0 if x["label"] == 1 else 0,axis=1)
    ###changing -1 to 0 
    clopid['label']=clopid.apply(lambda x: 1.0 if x["label"] == 1 else 0,axis=1)
    ###changing -1 to 0 
    proton_beam['label']=proton_beam.apply(lambda x: 1.0 if x["label"] == 1 else 0,axis=1)
    ###changing Y and N to 1 and 0 for hyperglycemia dataset
    hyperglycemia['label']=hyperglycemia.apply(lambda x: 1.0 if x["Decision (Y/N)"] == 'Y' else 0,axis=1)
    hyperglycemia.drop(['Decision (Y/N)'], axis=1, inplace=True)


    Y=pd.concat([anemia[['label','pmid']], clopid[['label','pmid']]], axis=0)
    Y=pd.concat([Y, proton_beam[['label','pmid']]], axis=0)
    Y=pd.concat([Y, hyperglycemia[['label','pmid']]], axis=0)
    Y=Y.set_index('pmid')

    Y=Y[~Y.index.duplicated(keep='first')]

    model_cp_y=Y[Y.index.isin(clopid.pmid)]
    model_an_y=Y[Y.index.isin(anemia.pmid)]
    model_pb_y=Y[Y.index.isin(proton_beam.pmid)]
    model_un_y=Y[Y.index.isin(hyperglycemia['pmid'])]

    model_un_y=model_un_y.reindex(model_un.index)
    model_cp_y=model_cp_y.reindex(model_cp.index)
    model_an_y=model_an_y.reindex(model_an.index)
    model_pb_y=model_pb_y.reindex(model_pb.index)

    testacclis=[]
    testauclis=[]

    model=torch.load(config['ae_path'])

    wsslis=defaultdict(def_value)
    auclis=defaultdict(def_value)

    ylis= [model_an_y, model_un_y, model_cp_y, model_pb_y]
    xlis= [model_an, model_un, model_cp, model_pb]
    vec_len=768
    for ds in range(0,4):
        
        kf = KFold(n_splits=5, shuffle=True, random_state=300)
        feat=xlis[ds]
        y=ylis[ds].squeeze()
        for i, (train_index, test_index) in enumerate(kf.split(feat)):
        
            X_train=feat.iloc[train_index]
            X_test=feat.iloc[test_index]
            y_train=y.iloc[train_index]
            y_test=y.iloc[test_index]
            
            X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.25,random_state=300)
            
            X_train=X_train.iloc[:, 2*vec_len:3*vec_len]
            X_test=X_test.iloc[:, 2*vec_len:3*vec_len]
            X_val=X_val.iloc[:, 2*vec_len:3*vec_len]
            
            X_train=torch.tensor(X_train.values, dtype=torch.float)
            X_test=torch.tensor(X_test.values, dtype=torch.float)
            X_val=torch.tensor(X_val.values, dtype=torch.float)
            
            y_train=torch.tensor(y_train.values, dtype=torch.float)
            y_val=torch.tensor(y_val.values, dtype=torch.float)
            y_test=torch.tensor(y_test.values, dtype=torch.float)
            
            X_train=model.reduce(X_train)
            X_test=model.reduce(X_test)
            X_val=model.reduce(X_val)
            
            px = pd.DataFrame(X_train.detach().numpy())
            pt = pd.DataFrame(X_test.detach().numpy())
            pv = pd.DataFrame(X_val.detach().numpy())
            
            X_train=torch.tensor(px.values, dtype=torch.float)
            X_test=torch.tensor(pt.values, dtype=torch.float)
            X_val=torch.tensor(pv.values, dtype=torch.float)
            
            ac,ws=train_val_test(X_train,y_train,X_val, y_val,X_test, y_test)
            auclis[ds] += ac/5
            wsslis[ds] += ws/5

    cohen=pd.read_csv(config['cohen_path'])
    cohen=cohen[['drug','abstract','pmid','label']]
    cohen.dropna(inplace=True)
    cohenlis=lis
    cohenlis['pmid']=cohenlis.index

    cohen=cohen.set_index('pmid')

    cohenauc=defaultdict(def_value)
    cohenwss=defaultdict(def_value)

    for drug in cohen.drug.value_counts().index:
        
        dataset=cohen[cohen['drug']==drug]
        y=dataset[['label']].squeeze()
        vec=cohenlis[cohenlis['pmid'].isin(dataset.index)]
        vec=vec.reindex(y.index)
        vec=vec.iloc[:, 768*2:768*3]
        
        kf = KFold(n_splits=5, shuffle=True, random_state=300)
        for i, (train_index, test_index) in enumerate(kf.split(vec)):
            X_train=vec.iloc[train_index]
            X_test=vec.iloc[test_index]
            y_train=y.iloc[train_index]
            y_test=y.iloc[test_index]

            X_train, X_val, y_train, y_val= train_test_split(X_train, y_train, test_size=0.25,random_state=300)
            
            X_train=torch.tensor(X_train.values, dtype=torch.float)
            X_test=torch.tensor(X_test.values, dtype=torch.float)
            X_val=torch.tensor(X_val.values, dtype=torch.float)
            
            y_train=torch.tensor(y_train.values, dtype=torch.float)
            y_val=torch.tensor(y_val.values, dtype=torch.float)
            y_test=torch.tensor(y_test.values, dtype=torch.float)
            
            X_train=model.reduce(X_train)
            X_test=model.reduce(X_test)
            X_val=model.reduce(X_val)
            
            px = pd.DataFrame(X_train.detach().numpy())
            pt = pd.DataFrame(X_test.detach().numpy())
            pv = pd.DataFrame(X_val.detach().numpy())
            
            X_train=torch.tensor(px.values, dtype=torch.float)
            X_test=torch.tensor(pt.values, dtype=torch.float)
            X_val=torch.tensor(pv.values, dtype=torch.float)
            
            ac,ws=train_val_test(X_train,y_train,X_val, y_val,X_test, y_test)

            cohenauc[drug] += ac/5
            cohenwss[drug] += ws/5

    print('Results:')
    print('Anemia: wss:', wsslis[0], 'auc:', auclis[0])
    print('clopid: wss:', wsslis[2], 'auc:', auclis[2])
    print('proton_beam: wss:', wsslis[3], 'auc:', auclis[3])
    print('hyperglycemia: wss:', wsslis[1], 'auc:', auclis[1])
    print('Cohen wss:', cohenwss)
    print('Cohen auc:', cohenauc)

    return 0.0



