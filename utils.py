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
from torch.distributions import Normal
from collections import defaultdict
from prettytable import PrettyTable
## Data----------------------------------------------------------------------------------------
class Tabledata(Dataset):
    def __init__(self, args, data, scale='minmax', use_treatment=False, binary_t=False):
        self.use_treatment = use_treatment
        # padding tensors
        self.diff_tensor = torch.zeros([124,1])
        if use_treatment:
            self.cont_tensor = torch.zeros([124,4])
        else:
            self.cont_tensor = torch.zeros([124,5])
        self.cat_tensor = torch.zeros([124,7])
        yd=[]
        for _, group in data.groupby('cluster'):
            yd.append(group[['y', 'd']].tail(1))
        yd = pd.concat(yd)

        ## 데이터 전처리 ##
        # 연속 데이터 정규화 #
        for c in ["age", "dis", "danger", "CT_R", "CT_E"]:
            if scale == 'minmax':
                minmax_col(data, c)
            elif scale =='normalization':
                meanvar_col(data, c)
        
        # 정규화 데이터 역변환을 위한 값 저장 #
        if scale == 'minmax':
            self.a_y, self.b_y = minmax_col(yd,"y")
            self.a_d, self.b_d = minmax_col(yd,"d")
        elif scale =='normalization':
            self.a_y, self.b_y = meanvar_col(yd, "y")
            self.a_d, self.b_d = meanvar_col(yd, "d")

        ## 데이터 특성 별 분류 및 저장 ##
        self.cluster = data.iloc[:,0].values.astype('float32')
        if use_treatment:
            if not binary_t:
                self.dis = data['dis'].values.astype('float32')
            else:
                print("use binary t")
                self.dis = (data['dis'].values >= 0.5).astype('float32')
            self.cont_X = data.iloc[:, 1:6].drop(columns=['dis']).values.astype('float32')
        else:
            self.dis=None
            self.cont_X = data.iloc[:, 1:6].values.astype('float32')
        self.cat_X = data.iloc[:, 6:13].astype('category')
        self.diff_days = data.iloc[:, 13].values.astype('float32')

        # y label tukey transformation
        # self.y = yd.values.astype('float32')
        y = torch.tensor(yd['y'].values.astype('float32'))
        d = torch.tensor(yd['d'].values.astype('float32'))
        if args.tukey:
            y = tukey_transformation(y, args)
            d = tukey_transformation(d, args)
        
        self.y = torch.stack([y, d], dim=1)

        # 이산 데이터 정렬 및 저장#
        self.cat_cols = self.cat_X.columns
        self.cat_map = {col: {cat: i for i, cat in enumerate(self.cat_X[col].cat.categories)} for col in self.cat_cols}
        self.cat_X = self.cat_X.apply(lambda x: x.cat.codes)
        self.cat_X = torch.from_numpy(self.cat_X.to_numpy()).long()
    def __len__(self):
        return len(np.unique(self.cluster))

    def __getitem__(self, index):
        '''
            [batch x padding x embedding]
            cont_tensor_p : 패딩이 씌워진 환자 관련 연속 데이터  
            cont_tensor_c : 패딩이 씌워진 클러스터 관련 연속 데이터  
            cat_tensor_p : 패딩이 씌워진 환자 관련 이산 데이터  
            cat_tensor_c : 패딩이 씌워진 클러스터 관련 이산 데이터  
            data_len : 클러스터별 유효 환자수 반환 데이터
            y : 정답 label
            diff_tensor : 클러스터별 유효 날짜 반환 데이터
        '''
        diff_days = torch.from_numpy(self.diff_days[self.cluster == index]).unsqueeze(1)
        diff_tensor = self.diff_tensor.clone()
        diff_tensor[:diff_days.shape[0]] = diff_days
        cont_X = torch.from_numpy(self.cont_X[self.cluster == index])
        data_len = cont_X.shape[0]
        cont_tensor = self.cont_tensor.clone()
        cont_tensor[:cont_X.shape[0],] = cont_X
        cat_X = self.cat_X[self.cluster == index]
        cat_tensor = self.cat_tensor.clone()
        cat_tensor[:cat_X.shape[0],] = cat_X
        cat_tensor_p = cat_tensor[:, :5]
        cat_tensor_c = cat_tensor[:, 5:]
        cont_tensor_p = cont_tensor[:, :3]
        cont_tensor_c = cont_tensor[:, 3:]
        y = self.y[index]
        if not self.use_treatment:
            return cont_tensor_p, cont_tensor_c, cat_tensor_p, cat_tensor_c, data_len, y, diff_tensor
        else:
            dis = torch.tensor(self.dis[index])
            # return cont_tensor, cat_tensor, data_len, y, diff_tensor, dis
            return cont_tensor_p, cont_tensor_c, cat_tensor_p, cat_tensor_c, data_len, y, diff_tensor, dis

class CEVAEdataset():
    def __init__(self, data, scale='minmax', t_type="multi"):
        columns = ['cluster', 'dis', 'danger','age', 'CT_R', 'CT_E', 'gender', 'is_korean',
           'primary case', 'job_idx', 'rep_idx', 'place_idx', 'add_idx', 'diff_days',
           'y', 'd', 'cut_date']
        data=data[columns]
        yd=[]
        for _, group in data.groupby('cluster'):
            yd.append(group[['y', 'd']].tail(1))
        yd = pd.concat(yd)

        for c in ["age", "dis", "danger", "CT_R", "CT_E"]:
            if scale == 'minmax':
                minmax_col(data, c)
            elif scale =='normalization':
                meanvar_col(data, c)
        
        if scale == 'minmax':
            self.a_y, self.b_y = minmax_col(yd,"y")
            self.a_d, self.b_d = minmax_col(yd,"d")
        elif scale =='normalization':
            self.a_y, self.b_y = meanvar_col(yd, "y")
            self.a_d, self.b_d = meanvar_col(yd, "d")

        self.x=torch.tensor(data.iloc[:,2:13].values).to(torch.float32)
        self.y=torch.tensor(data.iloc[:,14:16].values).squeeze()
        self.t=torch.tensor(data.iloc[:,1].values)
        if t_type=="binary":
            self.t = torch.where(self.t >= 0.5, torch.tensor(1), torch.tensor(0))
        elif t_type=="multi":
            self.t = self.t * 6
    def get_data(self):
        return self.x, self.y, self.t
    
    def get_rescale(self):
        return self.a_y, self.b_y, self.a_d, self.b_d
        

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
        return x
# ---------------------------------------------------------------------------------------------




## Train --------------------------------------------------------------------------------------
def train(data, model, optimizer, criterion, epoch, warmup_iter=0, lamb=0.0, aux_criterion=None, use_treatment=False, eval_criterion = None, scaling="minmax",a_y=None, b_y=None, a_d=None, b_d=None, pred_model="enc", binary_t=False, lambdas=[1,1,1]):
    eval_loss_y = None; eval_loss_d=None
    model.train()
    optimizer.zero_grad()
    batch_num, cont_p, cont_c, cat_p, cat_c, len, y, diff_days, *rest = data_load(data, use_treatment)
    out = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
    if use_treatment:
        t = rest[0]
        # x, x_reconstructed, z_mu, z_logvar, enc_preds, dec_preds, warm_yd = out
        x, x_reconstructed, enc_preds, dec_preds = out
        enc_yd_pred, enc_t_pred = enc_preds
        dec_yd_pred, dec_t_pred = dec_preds
        # loss, *ind_losses = cevae_loss_function(x_reconstructed, x, enc_t_pred, enc_yd_pred[:, 0], enc_yd_pred[:, 1], dec_t_pred, dec_yd_pred[:, 0], dec_yd_pred[:, 1], z_mu, z_logvar, t, y[:,0] , y[:,1],warm_yd ,criterion, aux_criterion, binary_t)
                                              #x_reconstructed, x, enc_t_pred, enc_y_pred, enc_d_pred, dec_t_pred, dec_y_pred, dec_d_pred, z_mu, z_logvar, t, y , d, criterion, lamdas
        loss, *ind_losses = cetransformer_loss(x_reconstructed, x, enc_t_pred, enc_yd_pred[:, 0], enc_yd_pred[:, 1], dec_t_pred, dec_yd_pred[:, 0], dec_yd_pred[:, 1], None, None, t.unsqueeze(1), y[:,0] , y[:,1], criterion, lambdas)
        # loss, *ind_losses = cetransformer_loss(x_reconstructed, x, enc_t_pred, 0, 0, dec_t_pred, dec_yd_pred[:, 0], dec_yd_pred[:, 1], None, None, t.unsqueeze(1), y[:,0] , y[:,1], criterion, lambdas)
        
        # (warmup_loss_y, warmup_loss_d), (enc_loss_y, enc_loss_d), (dec_loss_y, dec_loss_d) = ind_losses
        (enc_loss_y, enc_loss_d), (dec_loss_y, dec_loss_d) = ind_losses
        if False: # TODO: hardcode
            loss_y = enc_loss_y
            loss_d = enc_loss_d
            out = enc_yd_pred
            dec_out = dec_yd_pred
        else:
            loss_y = dec_loss_y
            loss_d = dec_loss_d
            out = dec_yd_pred
    else:
        loss_d = criterion(out[:,0], y[:,0])
        loss_y = criterion(out[:,1], y[:,1])    
        loss = loss_d + loss_y

    if eval_criterion != None:
        pred_y, pred_d, gt_y, gt_d = reverse_scaling(scaling, out, y, a_y, b_y, a_d, b_d)
        eval_loss_y = eval_criterion(pred_y, gt_y)
        eval_loss_d = eval_criterion(pred_d, gt_d)
    # Add Penalty term for ridge regression
    if lamb != 0.0:
        loss += lamb * torch.norm(model.linear1.weight, p=2)
    if not torch.isnan(loss):
        if epoch >= warmup_iter:  
            loss.backward()
        else:
            warmup_loss = warmup_loss_y + warmup_loss_d
            warmup_loss.backward()
        optimizer.step()
        return loss_d.item(), loss_y.item(), batch_num, out, y, eval_loss_y, eval_loss_d
    else:
        # return 0, batch_num, out, y
        raise ValueError("Loss raised nan.")

## Validation --------------------------------------------------------------------------------
@torch.no_grad()
def valid(data, model, eval_criterion, scaling, a_y, b_y, a_d, b_d, use_treatment=False):
    model.eval()
    
    batch_num, cont_p, cont_c, cat_p, cat_c, len, y, diff_days, *rest = data_load(data, use_treatment)
    out = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
    if use_treatment:
        t = rest[0]
        # x, x_reconstructed, z_mu, z_logvar, enc_preds, dec_preds, warm_yd = out
        x, x_reconstructed, enc_preds, dec_preds = out
        enc_yd_pred, enc_t_pred = enc_preds
        dec_yd_pred, dec_t_pred = dec_preds
        # loss, *ind_losses = cevae_loss_function(x_reconstructed, x, enc_t_pred, enc_yd_pred[:, 0], enc_yd_pred[:, 1], dec_t_pred, dec_yd_pred[:, 0], dec_yd_pred[:, 1], z_mu, z_logvar, t, y[:,0] , y[:,1], criterion, aux_criterion)
        # (enc_loss_y, enc_loss_d), (dec_loss_y, dec_loss_d) = ind_losses
        if False: # TODO: hardcode
            # loss_y = enc_loss_y
            # loss_d = enc_loss_d
            out = enc_yd_pred
            # out = dec_yd_pred
        else:
            # loss_y = dec_loss_y
            # loss_d = dec_loss_d
            out = dec_yd_pred

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
def test(data, model, scaling, a_y, b_y, a_d, b_d, use_treatment=False):
    
    criterion_mae = nn.L1Loss(reduction="sum")
    criterion_rmse = nn.MSELoss(reduction="sum")
    
    model.eval()

    batch_num, cont_p, cont_c, cat_p, cat_c, len, y, diff_days, *rest = data_load(data, use_treatment)
    out = model(cont_p, cont_c, cat_p, cat_c, len, diff_days)
    if use_treatment:
        t = rest[0]
        # x, x_reconstructed, z_mu, z_logvar, enc_preds, dec_preds, warm_yd = out
        x, x_reconstructed, enc_preds, dec_preds = out
        enc_yd_pred, enc_t_pred = enc_preds
        dec_yd_pred, dec_t_pred = dec_preds
        # loss, *ind_losses = cevae_loss_function(x_reconstructed, x, enc_t_pred, enc_yd_pred[:, 0], enc_yd_pred[:, 1], dec_t_pred, dec_yd_pred[:, 0], dec_yd_pred[:, 1], z_mu, z_logvar, t, y[:,0] , y[:,1], criterion, aux_criterion)
        # (enc_loss_y, enc_loss_d), (dec_loss_y, dec_loss_d) = ind_losses
        if False: # TODO: hardcode
            # loss_y = enc_loss_y
            # loss_d = enc_loss_d
            out = enc_yd_pred
            # out = dec_yd_pred
        else:
            # loss_y = dec_loss_y
            # loss_d = dec_loss_d
            out = dec_yd_pred
    if out.shape == torch.Size([2]):
        out = out.unsqueeze(0)
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
    
# # TODO: 이거 아님, ori-1~6 ori+1~6으로 가고, ori=1이라면? 식의 코드 변경
# @torch.no_grad()
# def estimate_counterfactuals(model, dataloader, a_y, b_y, a_d, b_d, scaling="minmax", use_treatment=True, binary_t=False):
#     # counter_factual_classes = 2 if binary_t else 7
#     counter_factual_classes = 7
#     model.eval()  # Set the model to evaluation mode
#     all_counterfactual_differences = []

#     for data in dataloader:
#         batch_num, cont_p, cont_c, cat_p, cat_c, len_, y, diff_days, *rest = data_load(data, use_treatment)
        
#         if use_treatment:
#             if binary_t:
#                 t_original = rest[0].long()    
#             else:
#                 t_original = 6*rest[0].long()
#             batch_counterfactual_differences = []

#             # Get the original predictions
#             out_original = model(cont_p, cont_c, cat_p, cat_c, len_, diff_days, t_original)
#             _, _, _, _, enc_preds_original, dec_preds_original, warmup_original = out_original # TODO:assume encoder output
#             pred_yd_original, _ = enc_preds_original
            
#             for t_value in range(counter_factual_classes):  
#                 t = torch.full_like(t_original, fill_value=t_value).long()  # Create a tensor with the new t value
#                 out = model(cont_p, cont_c, cat_p, cat_c, len_, diff_days, t)
#                 _, _, _, _, enc_preds_intervene, dec_preds_intervene, _ = out
#                 pred_yd_intervene, _ = enc_preds_intervene
                
#                 ori_y, ori_d, int_y, int_d = reverse_scaling(scaling, pred_yd_original, pred_yd_intervene, a_y, b_y, a_d, b_d)

#                 # Compute the differences from the original predictions
#                 diff_y = int_y - ori_y
#                 diff_d = int_d - ori_d
                
#                 # Store the t value difference and the prediction differences
#                 t_diff = t_value - t_original
#                 differences_dict = {
#                     "t_diff": t_diff.cpu().numpy(),
#                     "diff_y": diff_y.cpu().numpy(),
#                     "diff_d": diff_d.cpu().numpy()
#                 }
#                 batch_counterfactual_differences.append(differences_dict)
            
#             all_counterfactual_differences.append(batch_counterfactual_differences)
#         else:
#             raise ValueError("The use_treatment argument should be True for counterfactual estimation.")

#     return all_counterfactual_differences


# def organize_counterfactuals(all_counterfactual_differences):
#     organized_counterfactuals = defaultdict(lambda: {"diff_y": [], "diff_d": []})

#     for batch_counterfactual_differences in all_counterfactual_differences:
#         for differences_dict in batch_counterfactual_differences:
#             t_diff_values = differences_dict['t_diff']
#             diff_y_values = differences_dict['diff_y']
#             diff_d_values = differences_dict['diff_d']
            
#             # Assuming all arrays have the same length
#             for i in range(len(t_diff_values)):
#                 t_diff = t_diff_values[i]
#                 key = f't_diff_{int(t_diff)}'
#                 organized_counterfactuals[key]["diff_y"].append(diff_y_values[i])
#                 organized_counterfactuals[key]["diff_d"].append(diff_d_values[i])
#     # Convert lists to numpy arrays for consistency
#     for key, value in organized_counterfactuals.items():
#         organized_counterfactuals[key]["diff_y"] = np.array(value["diff_y"])
#         organized_counterfactuals[key]["diff_d"] = np.array(value["diff_d"])

#     return organized_counterfactuals

# def compute_average_differences(organized_counterfactuals):
#     average_differences = {}

#     for t_diff_key, values in organized_counterfactuals.items():
#         diff_y_avg = np.mean(values['diff_y'])
#         diff_d_avg = np.mean(values['diff_d'])

#         average_differences[t_diff_key] = {
#             "avg_diff_y": diff_y_avg,
#             "avg_diff_d": diff_d_avg
#         }
    
#     return average_differences
@torch.no_grad()
def estimate_counterfactuals(model, dataloader, a_y, b_y, a_d, b_d, scaling="minmax", use_treatment=True):
    model.eval()
    all_counterfactual_differences = []

    for data in dataloader:
        batch_num, cont_p, cont_c, cat_p, cat_c, len_, y, diff_days, *rest = data_load(data, use_treatment)
        if use_treatment:
            t_original = rest[0]

            batch_counterfactual_differences = []
            out_original = model(cont_p, cont_c, cat_p, cat_c, len_, diff_days, t_original)
            pred_yd_original = out_original

            for delta_t in range(-6, 7):
                t_adjusted = t_original + delta_t
                # 범위 내에 있는 요소만 필터링합니다.
                valid_indices = (t_adjusted >= 0) & (t_adjusted <= 6)
                t_valid = t_adjusted[valid_indices]

                if t_valid.numel() > 0:
                    out = model(cont_p[valid_indices], cont_c[valid_indices], cat_p[valid_indices], cat_c[valid_indices], len_[valid_indices], diff_days[valid_indices], t_valid)
                    pred_yd_intervene = out

                    ori_y, ori_d, int_y, int_d = reverse_scaling(scaling, pred_yd_original[valid_indices], pred_yd_intervene, a_y, b_y, a_d, b_d)
                    diff_y = int_y - ori_y
                    diff_d = int_d - ori_d

                    differences_dict = {
                        "delta_t": delta_t,
                        "diff_y": diff_y.cpu().numpy(),
                        "diff_d": diff_d.cpu().numpy()
                    }
                    batch_counterfactual_differences.append(differences_dict)
            
            all_counterfactual_differences.append(batch_counterfactual_differences)
        else:
            raise ValueError("The use_treatment argument should be True for counterfactual estimation.")

    return all_counterfactual_differences


def organize_counterfactuals(all_counterfactual_differences):
    organized_counterfactuals = defaultdict(lambda: {"diff_y": [], "diff_d": []})

    # 각 배치의 차이를 모읍니다.
    for batch_counterfactual_differences in all_counterfactual_differences:
        for differences_dict in batch_counterfactual_differences:
            delta_t = differences_dict["delta_t"]
            key = f't_diff_{delta_t}'
            organized_counterfactuals[key]["diff_y"].extend(differences_dict["diff_y"])
            organized_counterfactuals[key]["diff_d"].extend(differences_dict["diff_d"])

    # TODO 디버깅 필요
    # 각 diff_y와 diff_d 리스트를 하나의 NumPy 배열로 결합합니다.
    for key, value in organized_counterfactuals.items():
        organized_counterfactuals[key]["diff_y"] = np.concatenate(value["diff_y"])
        organized_counterfactuals[key]["diff_d"] = np.concatenate(value["diff_d"])

    return organized_counterfactuals


def compute_average_differences(organized_counterfactuals):
    average_differences = {}

    for t_diff_key, values in organized_counterfactuals.items():
        diff_y_avg = np.mean(values['diff_y'], axis=0)
        diff_d_avg = np.mean(values['diff_d'], axis=0)

        average_differences[t_diff_key] = {
            "avg_diff_y": diff_y_avg,
            "avg_diff_d": diff_d_avg
        }
    
    return average_differences

def print_average_differences(average_differences):
    # Create a table with the column names
    table = PrettyTable()
    table.field_names = ["t_diff", "avg_diff_y", "avg_diff_d"]
    
    # Sort the keys to ensure t_diff is ordered
        # Sort the keys to ensure t_diff is ordered
    def sort_key(key):
        return int(key.split('_')[2])
    sorted_keys = sorted(average_differences.keys(), key=lambda x: int(x.split('_')[2]))
    
    # Add rows to the table for each t_diff value
    for t_diff_key in sorted_keys:
        values = average_differences[t_diff_key]
        row = [t_diff_key, f"{values['avg_diff_y']:.6f}", f"{values['avg_diff_d']:.6f}"]
        table.add_row(row)
    
    # Print the table
    print(table)

def data_load(data, use_treatment=False):
    # Move all tensors in the data tuple to GPU at once
    data = tuple(tensor.cuda() for tensor in data)
    
    if use_treatment:
        cont_p, cont_c, cat_p, cat_c, len, y, diff_days, dis = data
        return cont_p.shape[0], cont_p, cont_c, cat_p, cat_c, len, y, diff_days, dis
    else:
        cont_p, cont_c, cat_p, cat_c, len, y, diff_days, *rest = data
        return cont_p.shape[0], cont_p, cont_c, cat_p, cat_c, len, y, diff_days, None

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

def patient_seq_to_date_seq(non_padded_cluster, non_padded_days ):
    days_uniq=non_padded_days.unique()
    result = torch.zeros(days_uniq.size()[0], non_padded_cluster.shape[-1])  

    for i, value in enumerate(days_uniq):
        indices = torch.where(non_padded_days == value)[0].unsqueeze(1)
        mean_value = torch.mean(non_padded_cluster[indices], dim=0)  
        result[i] = mean_value

    return result, days_uniq.size()[0]

def reduction_cluster(x, diff_days, len, reduction):
    cluster = []
    for i in range(x.shape[0]):
        pad_tensor = torch.zeros([5,x.shape[-1]]).cuda()
        m = len[i].item()
        non_padded_cluster = x[i, :m, :]  
        ## 클러스터 기준 평균 ##
        if reduction == "mean":
            non_padded_cluster = torch.mean(non_padded_cluster, dim=0)
        ## 클러스터 내 날짜 기준 평균 ##
        elif reduction == "date":
            non_padded_days = diff_days[i, :m, :]
            non_padded_cluster, new_len = patient_seq_to_date_seq(non_padded_cluster, non_padded_days)
            len[i]=new_len
            pad_tensor[:non_padded_cluster.shape[0]] = non_padded_cluster
            non_padded_cluster=pad_tensor
        cluster.append(non_padded_cluster)

    return torch.stack(cluster, dim=0)

### Tukey transformation
def tukey_transformation(data, args):
    epsilon = 1e-8  
    
    if args.tukey:
        data[data == 0] = epsilon
        if args.beta != 0:
            data = torch.pow(data, args.beta)
        elif args.beta == 0:
            data = torch.log(data)
        else:
            data = (-1) * torch.pow(data, args.beta)
        data[torch.isnan(data)] = 0.0
        
    return data

def inverse_tukey_transformation(data, args):
    epsilon = 1e-8
    
    if args.tukey:
        # Handle NaNs (these would have been zeros in the original data)
        data[torch.isnan(data)] = 0.0

        # Inverse transform based on beta
        if args.beta != 0:
            data = torch.pow(data, 1 / args.beta)
        elif args.beta == 0:
            data = torch.exp(data)
        
        # Restore zeros (these were converted to epsilon in the original data)
        data[torch.abs(data - epsilon) < 1e-8] = 0.0
    
    return data

## for VAE
def reparametrize(mu, logvar):
    # Calculate standard deviation
    std = torch.exp(0.5 * logvar)
    
    # Create a standard normal distribution
    epsilon = Normal(torch.zeros_like(mu), torch.ones_like(std)).rsample()
    
    # Reparametrization trick
    z = mu + epsilon * std
    return z

# def cevae_loss_function(x_reconstructed, x, enc_preds, dec_preds, z_mu, z_logvar, y, t, aux_criterion):

#     # Loss = logp(x,t∣z) + logp(y∣t,z) + logp(d∣t,z) + logp(z) − logq(z∣x,t,y,d) + logq(t=t∗∣x∗) + logq(y=y∗∣x∗,t∗) + logq(d=d∗∣x∗,t∗)

#     # Reconstruction loss for x
#     recon_loss = F.mse_loss(x_reconstructed, x)

#     # Predictive loss for observed data
#     enc_yd_pred, enc_t_pred = enc_preds
#     dec_t_pred, dec_yd_pred = dec_preds

#     pred_loss_y = F.mse_loss(enc_yd_pred[:, 1], y[:, 1])   # or use any appropriate loss
#     pred_loss_d = F.mse_loss(enc_yd_pred[:, 0], y[:, 0])   # or use any appropriate loss

#     # Use cross-entropy for categorical prediction of t
#     pred_loss_t_enc = aux_criterion(enc_t_pred, t.long())
#     pred_loss_t_dec = aux_criterion(dec_t_pred, t.long())

#     # KL divergence
#     kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

#     # Combining the terms based on the provided formula
#     total_loss = recon_loss + pred_loss_y + pred_loss_d + kl_loss + pred_loss_t_enc + pred_loss_t_dec

#     return total_loss

# def cevae_loss_function(x_reconstructed, x, enc_t_pred, enc_y_pred, enc_d_pred, dec_t_pred, dec_y_pred, dec_d_pred, z_mu, z_logvar, t, y , d, warm_yd, criterion, aux_criterion, binary_t): 
#     # 0. Warmup Loss
#     warmup_loss_y = criterion(warm_yd[:,0], y)  
#     warmup_loss_d = criterion(warm_yd[:,1], d)  
#     warmup_loss = warmup_loss_y + warmup_loss_d
#     # 1. Reconstruction Loss
#     ## mse method
#     recon_loss_x = F.mse_loss(x_reconstructed, x)
#     recon_loss_t = criterion(dec_t_pred, t)
#     recon_loss_y = criterion(dec_y_pred, y)
#     recon_loss_d = criterion(dec_d_pred, d)
#     ## log prob method
#     # x_dist = torch.distributions.Normal(x_reconstructed, torch.exp(0.5 * torch.ones_like(x_reconstructed)))
#     # t_dist = torch.distributions.Categorical(logits=dec_t_pred)
#     # y_dist = torch.distributions.Normal(dec_y_pred, torch.exp(0.5 * torch.ones_like(dec_y_pred)))
#     # d_dist = torch.distributions.Normal(dec_d_pred, torch.exp(0.5 * torch.ones_like(dec_d_pred)))
    
#     # recon_loss_x = -x_dist.log_prob(x).sum()
#     # recon_loss_t = -t_dist.log_prob(t.long()).sum() if binary_t else -t_dist.log_prob((t * 6).long()).sum()
#     # recon_loss_y = -y_dist.log_prob(y).sum()
#     # recon_loss_d = -d_dist.log_prob(d).sum()
    
#     recon_loss = recon_loss_x + recon_loss_t + recon_loss_y + recon_loss_d

#     # 2. KL Divergence
#     kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())

#     # 3. Auxiliary Loss (Using the predicted values t*, y*, and d*)
#     # aux_loss_t = aux_criterion(enc_t_pred, t.long()) if binary_t else aux_criterion(enc_t_pred, (t * 6).long())  
#     aux_loss_t = criterion(enc_t_pred, t) 
#     aux_loss_y = criterion(enc_y_pred, y)  
#     aux_loss_d = criterion(enc_d_pred, d)  
#     aux_loss = aux_loss_t + aux_loss_y + aux_loss_d
#     # Combine the losses
#     total_loss = recon_loss + kl_loss + aux_loss
#     if torch.isnan(total_loss):
#         import pdb;pdb.set_trace()
#     return total_loss, (warmup_loss_y, warmup_loss_d), (aux_loss_y, aux_loss_d), (recon_loss_y, recon_loss_d)

def nan_filtered_loss(pred, target, criterion):
    valid_indices = torch.where(~torch.isnan(pred))[0]
    return criterion(pred[valid_indices], target[valid_indices])

def cetransformer_loss(x_reconstructed, x, enc_t_pred, enc_y_pred, enc_d_pred, dec_t_pred, dec_y_pred, dec_d_pred, z_mu, z_logvar, t, y , d, criterion, lambdas):
    # Encoder Prediction Loss
    enc_y_loss = criterion(enc_y_pred, y)
    enc_d_loss = criterion(enc_d_pred, d)
    enc_t_loss = criterion(enc_t_pred, t)
    enc_loss = enc_y_loss + enc_d_loss + enc_t_loss

    # Decoder Prediction Loss
    dec_y_loss = criterion(dec_y_pred, y)
    dec_d_loss = criterion(dec_d_pred, d)
    dec_t_loss = criterion(dec_t_pred, t)
    dec_loss = dec_y_loss + dec_d_loss + dec_t_loss

    # Reconstruction Loss
    #recon_loss = criterion(x_reconstructed, x)

    # Encoder Prediction Loss
    # enc_y_loss = nan_filtered_loss(enc_y_pred, y, criterion)
    # enc_d_loss = nan_filtered_loss(enc_d_pred, d, criterion)
    # enc_t_loss = nan_filtered_loss(enc_t_pred, t, criterion)
    # enc_loss = enc_y_loss + enc_d_loss + enc_t_loss

    # # Decoder Prediction Loss
    # dec_y_loss = nan_filtered_loss(dec_y_pred, y, criterion)
    # dec_d_loss = nan_filtered_loss(dec_d_pred, d, criterion)
    # dec_t_loss = nan_filtered_loss(dec_t_pred, t, criterion)
    # dec_loss = dec_y_loss + dec_d_loss + dec_t_loss

    # # Reconstruction Loss
    # recon_loss = nan_filtered_loss(x_reconstructed, x, criterion)

    total_loss = lambdas[0]*enc_loss + lambdas[1]*dec_loss #+ lambdas[2]*recon_loss
    return total_loss, (0, 0), (dec_y_loss, dec_d_loss)