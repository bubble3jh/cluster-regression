import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

import pickle
import random
import math
from torch.utils.data import Dataset


## Data----------------------------------------------------------------------------------------
class Tabledata(Dataset):
    def __init__(self, data, scale='minmax', ratio=0.5):
        self.ratio = ratio
        self.cont_tensor = torch.zeros([124,5])
        self.cat_tensor = torch.zeros([124,7])
        data=data[['cluster', 'age', 'CT_R', 'CT_E', 'dis', 'danger','gender', 'is_korean', 'primary case', 
                   'job_idx', 'rep_idx', 'place_idx', 'add_idx', 'diff_days','y']]

        data['cluster'] = data['cluster'].map({value: idx for idx, value in enumerate(sorted(data['cluster'].unique()))})
        for c in ["age", "dis", "danger", "CT_R", "CT_E"]:
            # minmax_col(data, c)
            meanvar_col(data, c)
        grouped = data.groupby('cluster')['diff_days'].agg(['max', 'min'])
        data['d'] = data['cluster'].map(lambda x: (grouped.loc[x, 'max'] - grouped.loc[x, 'min']) + 1)
        if scale == 'minmax':
            self.a_y, self.b_y = minmax_col(data,"y")
            self.a_d, self.b_d = minmax_col(data,"d")
        elif scale =='normalization':
            self.a_y, self.b_y = meanvar_col(data, "y")
            self.a_d, self.b_d = meanvar_col(data, "d")
        self.cluster = data.iloc[:,0].values.astype('float32')
        self.cont_X = data.iloc[:, 1:6].values.astype('float32')
        self.cat_X = data.iloc[:, 6:13].astype('category')
        self.y = data.iloc[:, -2:].values.astype('float32')
        # 범주형 데이터 처리 - 0~n 까지로 맞춰줌
        self.cat_cols = self.cat_X.columns
        self.cat_map = {col: {cat: i for i, cat in enumerate(self.cat_X[col].cat.categories)} for col in self.cat_cols}
        self.cat_X = self.cat_X.apply(lambda x: x.cat.codes)
        self.cat_X = torch.from_numpy(self.cat_X.to_numpy()).long()

    def __len__(self):
        return len(np.unique(self.cluster))

    def __getitem__(self, index):
        cont_X = torch.from_numpy(self.cont_X[self.cluster == index])
        cont_X_del = delete_rows_by_ratio(cont_X, self.ratio)
        data_len = cont_X_del.shape[0]
        # 0인 tensor 복제해서 구역 할당
        cont_tensor = self.cont_tensor.clone()
        cont_tensor[:cont_X_del.shape[0],] = cont_X_del

        cat_X = self.cat_X[self.cluster == index]
        cat_X = delete_rows_by_ratio(cat_X, self.ratio)
        cat_tensor = self.cat_tensor.clone()
        cat_tensor[:cat_X.shape[0],] = cat_X
        cat_tensor_p = cat_tensor[:, :5]
        cat_tensor_c = cat_tensor[:, 5:]
        cont_tensor_p = cont_tensor[:, :3]
        cont_tensor_c = cont_tensor[:, 3:]
        y = torch.tensor(self.y[self.cluster == index])
        return cont_tensor_p,cont_tensor_c, cat_tensor_p, cat_tensor_c, data_len, y[0]

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
        print(data)
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
        y = torch.tensor(self.y[index])
        return cont_P, disc_P, cont_C, disc_C, y

def delete_rows_by_ratio(tensor, ratio):
    tensor_size = tensor.size()
    
    num_rows_to_delete = math.ceil(tensor_size[0] * ratio)
    num_rows_to_delete = min(num_rows_to_delete, tensor_size[0] - 1) # 길이가 1보다 작아지지 않도록 함
    tensor = tensor[:tensor_size[0] - num_rows_to_delete]
    
    return tensor

## MinMax Scaling Functions ------------------------------------
def minmax_col(data, name):
    minval , maxval = data[name].min(), data[name].max()
    data[name]=(data[name]-data[name].min())/(data[name].max()-data[name].min())
    return minval, maxval

def restore_minmax(data, minv, maxv):
    data = (data * (maxv - minv)) + minv
    return data
# ---------------------------------------------------------------

## Normalization Scaling Functions ---------------------------------
def meanvar_col(data, name):
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
    batch_num, cont_p, cont_c, cat_p, cat_c, len, y = data_load(data)
    optimizer.zero_grad()
    out = model(cont_p, cont_c, cat_p, cat_c, len)
    loss_d = criterion(out[:,0], y[:,0])
    loss_y = criterion(out[:,1], y[:,1])
    loss = loss_d + loss_y
    
    # Add Penalty term for ridge regression
    if lamb != 0.0:
        loss += lamb * torch.norm(model.linear1.weight, p=2)
    
    if not torch.isnan(loss):
        loss.backward()
        optimizer.step()
        return loss.item(), batch_num, out, y
    else:
        return 0, batch_num, out, y

## Validation --------------------------------------------------------------------------------
@torch.no_grad()
def valid(data, model, eval_criterion, scaling, a_y, b_y, a_d, b_d):
    '''
    a_y : min_y or mean_y
    b_y : max_y or var_y
    a_d : min_d or min_d
    b_d : max_d or var_d
    '''
    model.eval()
    batch_num, cont_p, cont_c, cat_p, cat_c, len, y = data_load(data)
    out = model(cont_p, cont_c, cat_p, cat_c, len)
    
    if scaling=="minmax":
        pred_y = restore_minmax(out[:, 0], a_y, b_y)
        pred_d = restore_minmax(out[:, 1], a_d, b_d)
        gt_y = restore_minmax(y[:,0], a_y, b_y)
        gt_d = restore_minmax(y[:,1], a_d, b_d)
        
    elif scaling == "normalization":
        pred_y = restore_meanvar(out[:, 0], a_y, b_y)
        pred_d = restore_meanvar(out[:, 1], a_d, b_d)
        gt_y = restore_meanvar(y[:,0], a_y, b_y)
        gt_d = restore_meanvar(y[:,1], a_d, b_d)
        
    loss_y = eval_criterion(pred_y, gt_y)
    loss_d = eval_criterion(pred_d, gt_d)
    loss = loss_y + loss_d
    if not torch.isnan(loss):
        return loss.item(), batch_num, out, y
    else:
        return 0, batch_num, out, y
    


## Test ----------------------------------------------------------------------------------------
@torch.no_grad()
def test(data, model, eval_criterion, scaling, a_y, b_y, a_d, b_d):    
    model.eval()
    batch_num, cont_p, cont_c, cat_p, cat_c, len, y = data_load(data)
    out = model(cont_p, cont_c, cat_p, cat_c, len)
    
    if scaling=="minmax":
        pred_y = restore_minmax(out[:, 0], a_y, b_y)
        pred_d = restore_minmax(out[:, 1], a_d, b_d)
        gt_y = restore_minmax(y[:,0], a_y, b_y)
        gt_d = restore_minmax(y[:,1], a_d, b_d)
        
    elif scaling == "normalization":
        pred_y = restore_meanvar(out[:, 0], a_y, b_y)
        pred_d = restore_meanvar(out[:, 1], a_d, b_d)
        gt_y = restore_meanvar(y[:,0], a_y, b_y)
        gt_d = restore_meanvar(y[:,1], a_d, b_d)
        
    loss_y = eval_criterion(pred_y, gt_y)
    loss_d = eval_criterion(pred_d, gt_d)
    loss = loss_y + loss_d
    if not torch.isnan(loss):
        return loss.item(), batch_num, out, y
    else:
        return 0, batch_num, out, y

def data_load(data):
    cont_p,cont_c, cat_p, cat_c, len, y = data
    return cont_p.shape[0], cont_p.cuda(), cont_c.cuda(), cat_p.cuda(), cat_c.cuda(), len.cuda(), y.cuda()


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
