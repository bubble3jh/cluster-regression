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
    def __init__(self, data, date_cutoff, scale='minmax'):
        # Hyperparameters and padding tensors
        self.date_cutoff = date_cutoff
        self.cont_tensor = torch.zeros([124,5])
        self.cat_tensor = torch.zeros([124,7])
        
        # Preprocessing data
        data=data[['cluster', 'age', 'CT_R', 'CT_E', 'dis', 'danger','gender', 'is_korean', 'primary case', 
                   'job_idx', 'rep_idx', 'place_idx', 'add_idx', 'diff_days','y']]
        for c in ["age", "dis", "danger", "CT_R", "CT_E"]:
            # minmax_col(data, c)
            if scale == 'minmax':
                minmax_col(data, c)
            elif scale =='normalization':
                meanvar_col(data, c)
        min_diff_days = data.groupby('cluster')['diff_days'].min()
        data['diff_days'] = data.apply(lambda row: row['diff_days'] - min_diff_days[row['cluster']], axis=1)
        grouped = data.groupby('cluster')['diff_days'].max()
        data['d'] = data['cluster'].map(lambda x: grouped[x] + 1)
        data = data.sort_values(by=['cluster', 'diff_days'], ascending=[True, True])
        data.drop(data[(data['d'] <= self.date_cutoff) | (data['diff_days'] > self.date_cutoff)].index, inplace=True)
        data['cluster'] = data['cluster'].map({value: idx for idx, value in enumerate(sorted(data['cluster'].unique()))})
        
        if scale == 'minmax':
            self.a_y, self.b_y = minmax_col(data,"y")
            self.a_d, self.b_d = minmax_col(data,"d")
        elif scale =='normalization':
            self.a_y, self.b_y = meanvar_col(data, "y")
            self.a_d, self.b_d = meanvar_col(data, "d")

        # Split data by type
        self.cluster = data.iloc[:,0].values.astype('float32')
        self.cont_X = data.iloc[:, 1:6].values.astype('float32')
        self.cat_X = data.iloc[:, 6:13].astype('category')
        self.y = data.iloc[:, -2:].values.astype('float32')
        self.cat_cols = self.cat_X.columns
        self.cat_map = {col: {cat: i for i, cat in enumerate(self.cat_X[col].cat.categories)} for col in self.cat_cols}
        self.cat_X = self.cat_X.apply(lambda x: x.cat.codes)
        self.cat_X = torch.from_numpy(self.cat_X.to_numpy()).long()
        print(f"Number of Clusters : {len(np.unique(self.cluster))}")
    def __len__(self):
        return len(np.unique(self.cluster))

    def __getitem__(self, index):
        cont_X = torch.from_numpy(self.cont_X[self.cluster == index])
        # cont_X = delete_rows_by_ratio(cont_X, self.ratio)
        data_len = cont_X.shape[0]
        # 0인 tensor 복제해서 구역 할당
        cont_tensor = self.cont_tensor.clone()
        cont_tensor[:cont_X.shape[0],] = cont_X

        cat_X = self.cat_X[self.cluster == index]
        # cat_X = delete_rows_by_ratio(cat_X, self.ratio)
        cat_tensor = self.cat_tensor.clone()
        cat_tensor[:cat_X.shape[0],] = cat_X
        cat_tensor_p = cat_tensor[:, :5]
        cat_tensor_c = cat_tensor[:, 5:]
        cont_tensor_p = cont_tensor[:, :3]
        cont_tensor_c = cont_tensor[:, 3:]
        y = torch.tensor(self.y[self.cluster == index])
        return cont_tensor_p, cont_tensor_c, cat_tensor_p, cat_tensor_c, data_len, y[0]

# def delete_rows_by_ratio(tensor, ratio):
#     tensor_size = tensor.size()
    
#     num_rows_to_delete = math.ceil(tensor_size[0] * ratio)
#     num_rows_to_delete = min(num_rows_to_delete, tensor_size[0] - 1) # 길이가 1보다 작아지지 않도록 함
#     tensor = tensor[:tensor_size[0] - num_rows_to_delete]
    
#     return tensor

def delete_df_rows_by_ratio(df, ratio):
    cluster_groups = df.groupby('cluster')  # 'cluster' 열을 기준으로 그룹화
    for _, group in cluster_groups:
        group_size = len(group)
        num_rows_to_delete = math.ceil(group_size * ratio)
        num_rows_to_delete = min(num_rows_to_delete, group_size - 1) # 그룹의 크기보다 크게 삭제하지 않도록 함
        df.drop(group.tail(num_rows_to_delete).index, inplace=True)  # 뒤에서 일정 개수의 행 삭제
    df = df.reset_index(drop=True)
    return df

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
    def __init__(self, reduction):
        super(RMSELoss,self).__init__()
        self.mse = nn.MSELoss(reduction=reduction)
        self.eps = 1e-12

    def forward(self, target, pred):
        x = torch.sqrt(self.mse(target, pred) + self.eps)
        # print(x.shape)
        return x
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
        return loss_d.item(), loss_y.item(), batch_num, out, y
    else:
        return 0, batch_num, out, y

## Validation --------------------------------------------------------------------------------
@torch.no_grad()
def valid(data, model, eval_criterion, scaling, a_y, b_y, a_d, b_d):
    model.eval()
    batch_num, cont_p, cont_c, cat_p, cat_c, len, y = data_load(data)
    out = model(cont_p, cont_c, cat_p, cat_c, len)
    
    pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, out, y, a_y, b_y, a_d, b_d)
       
    loss_y = eval_criterion(pred_y, gt_y)
    loss_d = eval_criterion(pred_d, gt_d)
    loss = loss_y + loss_d
    if not torch.isnan(loss):
        return loss_d.item(), loss_y.item(), batch_num, out, y
    else:
        return 0, batch_num, out, y

## Test ----------------------------------------------------------------------------------------
@torch.no_grad()
def test(data, model, scaling, a_y, b_y, a_d, b_d):
    
    criterion_mae = nn.L1Loss(reduction="sum")
    criterion_rmse = nn.MSELoss(reduction="sum")
    
    model.eval()
    batch_num, cont_p, cont_c, cat_p, cat_c, len, y = data_load(data)
    out = model(cont_p, cont_c, cat_p, cat_c, len)
    
    pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, out, y, a_y, b_y, a_d, b_d)
    
    # MAE
    mae_y = criterion_mae(pred_y, gt_y)
    mae_d = criterion_mae(pred_d, gt_d)
    mae = mae_y + mae_d
    # RMSE
    rmse_y = criterion_rmse(pred_y, gt_y)
    rmse_d = criterion_rmse(pred_d, gt_d)
    rmse = rmse_y + rmse_d
    
    if not torch.isnan(mae) and not torch.isnan(rmse):
        return mae_d.item(), mae_y.item(), rmse_d.item(), rmse_y.item(), batch_num, out, y
    else:
        return 0, batch_num, out, y

def data_load(data):
    cont_p,cont_c, cat_p, cat_c, len, y = data
    return cont_p.shape[0], cont_p.cuda(), cont_c.cuda(), cat_p.cuda(), cat_c.cuda(), len.cuda(), y.cuda()

def reverse_scaling(scaling, out, y, a_y, b_y, a_d, b_d):
    '''
    a_y : min_y or mean_y
    b_y : max_y or var_y
    a_d : min_d or min_d
    b_d : max_d or var_d
    '''
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
    return pred_y, pred_d, gt_y, gt_d

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

def data_split_num(dataset, tr=0.8, val=0.1, te=0.1):
    train_length = int(tr * len(dataset))
    val_length = int(val * len(dataset))
    test_length = len(dataset) - train_length - val_length

    return train_length, val_length, test_length
