import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

import pickle
import random

from torch.utils.data import Dataset


## Data----------------------------------------------------------------------------------------
class Tabledata(Dataset):
    def __init__(self, data, scale='minmax'):
        for c in ["age", "dis", "danger", "CT_R", "CT_E"]:
            # minmax_col(data, c)
            meanvar_col(data, c)
        for i in data['cluster'].unique():
            max_diff_days = data[data['cluster'] == i]['diff_days'].max()
            data.loc[data['cluster'] == i, 'd'] = max_diff_days
        
        if scale == 'minmax':
            self.miny, self.maxy = minmax_col(data,"y")
            self.mind, self.maxd = minmax_col(data,"d")
        elif scale =='normalization':
            self.meany, self.vary = meanvar_col(data, "y")
            self.meand, self.vard = meanvar_col(data, "d")
        self.binary_X = data.iloc[:, 1:4].values.astype('float32')
        self.cont_X = data.iloc[:, 4:9].values.astype('float32') # diff_days 는 사용 x
        self.cat_X = data.iloc[:, 10:14].astype('category')
        self.y = data.iloc[:, -2:].values.astype('float32')

        # 범주형 데이터 처리 - 0~n 까지로 맞춰줌
        self.cat_cols = self.cat_X.columns
        self.cat_map = {col: {cat: i for i, cat in enumerate(self.cat_X[col].cat.categories)} for col in self.cat_cols}
        self.cat_X = self.cat_X.apply(lambda x: x.cat.codes)
        self.cat_X = torch.from_numpy(self.cat_X.to_numpy()).long()
        self.embeddings = nn.ModuleList([nn.Embedding(len(self.cat_map[col]), 4) for col in self.cat_cols])

    def __len__(self):
        return len(self.cont_X)

    def __getitem__(self, index):
        cont_X = torch.from_numpy(self.cont_X[index])
        cat_X = self.cat_X[index]
        # batch size 가 1일 때 예외 처리
        if len(cat_X.shape) == 1:
            embeddings = [embedding(cat_X[i].unsqueeze(0)) for i, embedding in enumerate(self.embeddings)]
        else:
            embeddings = [embedding(cat_X[:, i]) for i, embedding in enumerate(self.embeddings)]
        cat_X = torch.cat(embeddings, 1).squeeze()
        binary_X = torch.from_numpy(self.binary_X[index])
        y = torch.tensor(self.y[index])
        return binary_X, cont_X, cat_X, y

class Seqdata(Dataset):
    def __init__(self, data):
        for c in ["age", "dis", "danger", "CT_R", "CT_E"]:
            # minmax_col(data, c)
            meanvar_col(data, c)
        # miny, maxy = minmax_col(data,"y")
        # meany, vary = meanvar_col(data, "y")

        sub_y = []
        for _, group in data.groupby('cluster'):
            group['sub_y'] = group['diff_days'].map(group['diff_days'].value_counts())
            sub_y.extend(group['sub_y'].tolist())
        data['sub_y'] = sub_y
    
        data=data[['cluster', 'age',  'CT_R', 'CT_E', 'gender', 'is_korean', 'primary case', 'job_idx','rep_idx', # patient datas
                   'dis', 'danger', 'place_idx', 'add_idx', 'y', 'diff_days', 'sub_y']]                           # cluster datas
        data = data.sort_values(by=["cluster", "diff_days"], ascending=[True, True])
        
        self.cont_P = data.iloc[:, 1:4].values.astype('float32') 
        self.disc_P = data.iloc[:, 4:9].astype('category')
        self.cont_C = data.iloc[:, 9:11].values.astype('float32') 
        self.disc_C = data.iloc[:, 11:13].astype('category')
        
        self.y = data.iloc[:, -2:].values.astype('float32')

        # 범주형 데이터 처리 - 0~n 까지로 맞춰줌
        self.disc_P_cols = self.disc_P.columns
        self.disc_P_map = {col: {cat: i for i, cat in enumerate(self.disc_P[col].cat.categories)} for col in self.disc_P_cols}
        self.disc_P = self.disc_P.apply(lambda x: x.cat.codes)
        self.disc_P = torch.from_numpy(self.disc_P.to_numpy()).long()
        
        self.disc_C_cols = self.disc_C.columns
        self.disc_C_map = {col: {cat: i for i, cat in enumerate(self.disc_C[col].cat.categories)} for col in self.disc_C_cols}
        self.disc_C = self.disc_C.apply(lambda x: x.cat.codes)
        self.disc_C = torch.from_numpy(self.disc_C.to_numpy()).long()

    def __len__(self):
        return len(self.cont_P)

    def __getitem__(self, index):
        cont_C = torch.from_numpy(self.cont_C[index])
        cont_P = torch.from_numpy(self.cont_P[index])
        disc_C = self.disc_C[index]
        disc_P = self.disc_P[index]
        # # batch size 가 1일 때 예외 처리
        # if len(disc_C.shape) == 1:
        #     embeddings = [embedding(disc_C[i].unsqueeze(0)) for i, embedding in enumerate(self.embeddings)]
        # else:
        #     embeddings = [embedding(disc_C[:, i]) for i, embedding in enumerate(self.embeddings)]
        # disc_C = torch.cat(embeddings, 1).squeeze()
        # if len(disc_P.shape) == 1:
        #     embeddings = [embedding(disc_P[i].unsqueeze(0)) for i, embedding in enumerate(self.embeddings)]
        # else:
        #     embeddings = [embedding(disc_P[:, i]) for i, embedding in enumerate(self.embeddings)]
        # disc_P = torch.cat(embeddings, 1).squeeze()
        y = torch.tensor(self.y[index])
        return cont_P, disc_P, cont_C, disc_C, y

## MinMax Scaling Functions ------------------------------------
def minmax_col(data,name):
    minval , maxval = data[name].min(), data[name].max()
    data[name]=(data[name]-data[name].min())/(data[name].max()-data[name].min())
    return minval, maxval

def restore_minmax(data, minv, maxv):
    data = (data * (maxv - minv)) + minv
    return data
# ---------------------------------------------------------------

## Normalization Scaling Functions ---------------------------------
def meanvar_col(data,name):
    mean_val = data[name].mean()
    std_val = data[name].var()
    data[name]=(data[name]-data[name].mean())/data[name].var()
    return mean_val, std_val

def restore_meanvar(data, mean, var):
    data = data * var + mean
    return data
# ----------------------------------------------------------------



## Loss ----------------------------------------------------------------------------------------
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.mse = nn.MSELoss()
        self.eps = 1e-12

    def forward(self, target, pred):
        return torch.sqrt(self.mse(target, pred) + self.eps)
# ---------------------------------------------------------------------------------------------




## Train --------------------------------------------------------------------------------------
def train(data, model, optimizer, criterion, lamb=0.0):
    model.train()
    if model.__class__.__name__ != "TSTransformer" :
        binary_X, cont_X, cat_X, y = data
        data_x = torch.cat((binary_X, cont_X, cat_X), dim=1).cuda()
        y=y.cuda()
        optimizer.zero_grad()
        batch_num = data_x.shape[0]
        out = model(data_x)
        
        loss1 = criterion(out[:,0], y[:,0])
        loss2 = criterion(out[:,1], y[:,1])
        loss = loss1 + loss2
        
        # Add Penalty term for ridge regression
        if lamb != 0.0:
            loss += lamb * torch.norm(model.linear1.weight, p=2)
        
        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()
            return loss.item(), batch_num, out, y
        else:
            return 0, batch_num, out, y
    else :
        cont_P, disc_P, cont_C, disc_C, y = data
        cont_P, disc_P, cont_C, disc_C, y = cont_P.to("cuda"), disc_P.to("cuda"), cont_C.to("cuda"), disc_C.to("cuda"), y.to("cuda")
        optimizer.zero_grad()
        batch_num = cont_P.shape[0]
        out = model(cont_P, disc_P, cont_C, disc_C)
        
        # loss1 = criterion(out[:,0], y[:,0])
        # loss2 = criterion(out[:,1], y[:,1])
        # loss = loss1 + loss2
        
        # # Add Penalty term for ridge regression
        # if lamb != 0.0:
        #     loss += lamb * torch.norm(model.linear1.weight, p=2)
        
        if not torch.isnan(loss):
        #     loss.backward()
        #     optimizer.step()
            return loss.item(), batch_num, out, y
        else:
            return 0, batch_num, out, y

## Validation --------------------------------------------------------------------------------
@torch.no_grad()
def valid(data, model, eval_criterion):
    model.eval()
    binary_X, cont_X, cat_X, y = data
    y=y.cuda()
    data_x = torch.cat((binary_X, cont_X, cat_X), dim=1).cuda()

    batch_num = data_x.shape[0]
    out = model(data_x)

    loss1 = eval_criterion(out[:,0], y[:,0])
    loss2 = eval_criterion(out[:,1], y[:,1])
    loss = loss1 + loss2
    if not torch.isnan(loss):
        return loss.item(), batch_num, out, y
    else:
        return 0, batch_num, out, y
    


## Test ----------------------------------------------------------------------------------------
@torch.no_grad()
def test(data, model, eval_criterion):    
    model.eval()
    binary_X, cont_X, cat_X, y = data
    y=y.cuda()
    data_x = torch.cat((binary_X, cont_X, cat_X), dim=1).cuda()

    batch_num = data_x.shape[0]
    out = model(data_x)

    loss1 = eval_criterion(out[:,0], y[:,0])
    loss2 = eval_criterion(out[:,1], y[:,1])
    # loss = eval_criterion(out, y)
    loss = loss1 + loss2
    if not torch.isnan(loss):
        return loss.item(), batch_num, out, y
    else:
        return 0, batch_num, out, y




def set_seed(random_seed=1000):
    '''
    Set Seed for Reproduction
    '''
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def save_checkpoint(file_path, epoch, **kwargs):
    '''
    Save Checkpoint
    '''
    state = {"epoch": epoch}
    state.update(kwargs)
    torch.save(state, file_path)
