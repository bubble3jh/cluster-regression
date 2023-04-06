import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

import pickle
import random

from torch.utils.data import Dataset

class Tabledata(Dataset):
    def __init__(self, data):
        self.binary_X = data.iloc[:, 1:4].values.astype('float32')
        self.cont_X = data.iloc[:, 4:10].values.astype('float32')
        self.cat_X = data.iloc[:, 10:14].astype('category')
        self.y = data.iloc[:, -1].values.astype('float32')

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


## Data----------------------------------------------------------------------------------------
def min_max_scale(df, col, min = 0, max = 37):
    '''
    Adapt Min-Max Scaling
    '''
    df[col] = (df[col] - min) / (max - min)
    return df[col]


def restore_min_max(scaled_dat, min = 0, max = 37):
    '''
    Restore Min-Max Scaled Data
    '''
    dat = (max - min) * scaled_dat + min
    return dat



def full_load_data(data_path = './data/20221122_raw_data_utf-8.csv',
                num_features = 17,
                target_order = 3,
                train_ratio = 0.2,
                classification = True,
                device = 'cpu',
                model_name = 'MagNet'):
    '''
    Description:
        Load Data from bottom
    Input :
        - data_path : Path to load data
        - num_features : Number of features on one node
        - target_order : The order which we want to predict
        - train_ratio : Ratio of train data
        - device : Running device (cuda / cpu)
    Output :
        - final_data_list : the data we use to learning
        - y_min : minimum value of y label 
        - y_max : maximum value of y label
    '''
    df_dat, unique_group = load_data(data_path = data_path, classification = classification)
    df_dat, y_min, y_max = scaling_y(df = df_dat)

    # aggregate group (for each graph)
    group_df = df_dat.groupby("transmission_route").agg(list)

    data_x_list, data_edge_list, data_y_list, data_order_list, num_graph = split_graphs(group_df = group_df, unique_group = unique_group, num_features = num_features, model_name = model_name)

    new_data_x_list, new_data_edge_list, new_data_y_list, new_data_order_list, new_data_tr_mask_list, new_data_val_mask_list, new_data_te_mask_list = masking(data_x_list = data_x_list,
                                                                                                                            data_edge_list = data_edge_list,
                                                                                                                            data_y_list = data_y_list,
                                                                                                                            data_order_list = data_order_list,
                                                                                                                            target_order = target_order,
                                                                                                                            train_ratio = train_ratio)

    final_data_x_list, final_data_edge_list, final_data_y_list, final_data_order_list, final_data_tr_mask_list, final_data_val_mask_list, final_data_te_mask_list = del_zero_edge(new_data_x_list = new_data_x_list,
                                                                                                                            new_data_edge_list = new_data_edge_list,
                                                                                                                            new_data_y_list = new_data_y_list,
                                                                                                                            new_data_order_list = new_data_order_list,
                                                                                                                            new_data_tr_mask_list = new_data_tr_mask_list,
                                                                                                                            new_data_val_mask_list = new_data_val_mask_list,
                                                                                                                            new_data_te_mask_list = new_data_te_mask_list,
                                                                                                                            num_graph = num_graph)
    # with open('y_list.pkl', 'wb') as fp:
    #     pickle.dump(final_data_y_list, fp)

    # with open('order_list.pkl', 'wb') as fp:
    #     pickle.dump(final_data_order_list, fp)
    # pdb.set_trace()
    check_graphs(final_data_x_list = final_data_x_list,
                final_data_edge_list = final_data_edge_list,
                final_data_y_list = final_data_y_list,
                final_data_order_list = final_data_order_list, 
                final_data_tr_mask_list = final_data_tr_mask_list,
                final_data_val_mask_list = final_data_val_mask_list,
                final_data_te_mask_list = final_data_te_mask_list)

    final_data_list = to_tensor(final_data_x_list = final_data_x_list,
                            final_data_edge_list = final_data_edge_list,
                            final_data_y_list = final_data_y_list,
                            final_data_order_list = final_data_order_list,
                            final_data_tr_mask_list = final_data_tr_mask_list,
                            final_data_val_mask_list = final_data_val_mask_list,
                            final_data_te_mask_list = final_data_te_mask_list,
                            device = device)

    return final_data_list, y_min, y_max


def load_data(data_path='./data/20221122_raw_data_utf-8.csv', classification = True):
    # load data
    df = pd.read_csv(data_path, index_col=0)       # [8844, 37]

    # Making y label
    cdc_count = df['index_id_cdc'].value_counts()
    cdc_count = cdc_count.to_frame().reset_index()
    cdc_count.columns = ['id_inch', 'index_id_cdc_num']

    if classification:
        over = cdc_count['index_id_cdc_num']>=6
        cdc_count['index_id_cdc_num'][over] = 6

    df = pd.merge(df, cdc_count, how='left', on='id_inch')      # left outer join
    df['index_id_cdc_num'] = df['index_id_cdc_num'].fillna(0)   # 결측값 0으로 대치


    # Delete nan rows
    df.dropna(how='any', subset=['contact_n'], inplace=True)    # 차수가 없는것 삭제
    df = df.drop(df[df.contact_n == '?'].index)

    df.dropna(how='any', subset=['sex'], inplace=True)          # 성별 없는것 삭제

    df.dropna(how='any', subset=['a1','a2','a3','a4','a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15'], inplace=True)   # 증상 (a1-15) 없는것 삭제

    # index reset
    df = df.reset_index(drop=True)                        # [1154, 38]
    unique_group = pd.unique(df['transmission_route'])    # 155
    return df, unique_group


def scaling_y(df):
    # normalizing y label (min-max scaling)
    y_min = df['index_id_cdc_num'].min()
    y_max = df['index_id_cdc_num'].max()

    df['index_id_cdc_num'] = min_max_scale(df, 'index_id_cdc_num', y_min, y_max)
    return df, y_min, y_max


def split_graphs(group_df, unique_group, num_features, model_name):
    num_graph = len(unique_group)                       # number of graph (155)
    data_x_list = []                                    # node별 feature list (17)
    data_y_list = []                                    # node별 y (바이러스를 전파시킬 인원수)
    data_edge_list = []                                 # edge list
    data_direct_list = []                               # 선행확진자 (0, 1)
    data_order_list = []                                # 차수 # 1,2,3,4
    for itr in range(num_graph):
        temp_data = group_df.iloc[itr]
        num_person = len(temp_data['id_inch'])
        # Birth: 0 if birth<1982 1 else
        # Gender: 0 if male 1 else 

        ##### Edge ----------------------------------------------------------------------------|
        ## Assign indices per patient on each graph
        cnt = 0
        p2i_dict = {}
        for person_itr in range(num_person):
            p2i_dict[temp_data['id_inch'][person_itr]] = cnt
            cnt += 1

        ## Assign edge on each graph
        edge1_list = []
        edge2_list = []
        for person_itr in range(num_person):
            pre_patient = temp_data['index_id_cdc'][person_itr]   # pre_patient : 지금 환자에게 전파한 환자 (선행확진자)

            # graph 내에 선행확진자(pre_patient)가 존재하면 edge 설정
            if pre_patient in p2i_dict:
                post_patient = temp_data['id_inch'][person_itr]

                ## edge1 -> edge2 로 나타나는 edge (i.e. edge1_list[0] -> edge2_list[0])
                edge1_list.append(p2i_dict[pre_patient])
                edge2_list.append(p2i_dict[post_patient])
                if model_name not in ['MagNet']:
                    edge1_list.append(p2i_dict[post_patient])
                    edge2_list.append(p2i_dict[pre_patient])
        data_edge = [edge1_list, edge2_list]
        data_edge_list.append(data_edge)
        #-----------------------------------------------------------------------------------|

        ##### order ------------------------------------------------------------------------|
        ## Assign order of nodes on each graph
        data_order = np.zeros((num_person,))
        for person_itr in range(num_person):
            order = int(temp_data['contact_n'][person_itr][0])
            data_order[person_itr] = order
        data_order_list.append(data_order)
        #-----------------------------------------------------------------------------------|

        ##### y ----------------------------------------------------------------------------|
        ## Assign y label of nodes on each graph
        data_y = np.zeros((num_person,))
        for person_itr in range(num_person):
            y = temp_data['index_id_cdc_num'][person_itr]
            data_y[person_itr] = y
        data_y_list.append(data_y)
        #-----------------------------------------------------------------------------------|

        ##### NODE -------------------------------------------------------------------------|
        ## Assign x features of nodes on each graph
        data_x = np.zeros((num_person, num_features))
        for person_itr in range(num_person):
            # Birth
            if temp_data['birthyear'][person_itr] >= 2000:
                data_x[person_itr,0] = 0
            elif 1990 <= temp_data['birthyear'][person_itr] < 2000:
                data_x[person_itr,0] = 1
            elif 1980 <= temp_data['birthyear'][person_itr] < 1990:
                data_x[person_itr,0] = 2
            elif 1970 <= temp_data['birthyear'][person_itr] < 1980:
                data_x[person_itr,0] = 3
            elif 1960 <= temp_data['birthyear'][person_itr] < 1970:
                data_x[person_itr,0] = 4
            elif 1950 <= temp_data['birthyear'][person_itr] < 1960:
                data_x[person_itr,0] = 5
            elif 1940 <= temp_data['birthyear'][person_itr] < 1950:
                data_x[person_itr,0] = 6
            elif 1930 <= temp_data['birthyear'][person_itr] < 1940:
                data_x[person_itr,0] = 7
            elif 1920 <= temp_data['birthyear'][person_itr] < 1930:
                data_x[person_itr,0] = 8
            else:
                data_x[person_itr,0] = 9

        
            # Gender
            if temp_data['sex'][person_itr] == '남':
                data_x[person_itr,1] = 0
            else:
                data_x[person_itr,1] = 1
        
            # a property
            for a_itr in range(1,16):
                data_x[person_itr,a_itr+1] = temp_data['a'+str(a_itr)][person_itr]
        data_x_list.append(data_x)
        #-----------------------------------------------------------------------------------|

    assert len(data_x_list) == len(data_edge_list) == len(data_y_list) == len(data_order_list)
    print(f"Exist {len(data_x_list)} graphs after first step....(Split graphs)")
    return data_x_list, data_edge_list, data_y_list, data_order_list, num_graph



def masking(data_x_list, data_edge_list, data_y_list, data_order_list, target_order, train_ratio=0.2):
    te_val_ratio = 1 - train_ratio

    new_data_x_list = []
    new_data_edge_list = []
    new_data_y_list = []
    new_data_order_list = []
    new_data_tr_mask_list = []
    new_data_val_mask_list = []
    new_data_te_mask_list = []

    for data_x, data_edge, data_y, data_order in zip(data_x_list, data_edge_list, data_y_list, data_order_list):
        mask_node_idx_list = np.argwhere(np.asarray(data_order) > target_order) # one-hop 보장
        
        # data node / label / order remove
        if not len(mask_node_idx_list) == 0:
            data_x = list(np.delete(list(data_x), mask_node_idx_list, axis=0))
            data_y = list(np.delete(data_y, mask_node_idx_list, axis=0))
            data_order = list(np.delete(data_order, mask_node_idx_list, axis=0))

        # edge remove
        edge_node_idx = []
        for mask_node_idx in mask_node_idx_list:
            match_value1 = np.argwhere(np.asarray(data_edge[0]) == mask_node_idx)
            match_value2 = np.argwhere(np.asarray(data_edge[1]) == mask_node_idx)
            edge_node_idx.extend(match_value1)
            edge_node_idx.extend(match_value2)
        edge_node_idx = np.unique(edge_node_idx)
        if not len(edge_node_idx) == 0:
            data_edge = [list(np.delete(data_edge[0], edge_node_idx)), list(np.delete(data_edge[1], edge_node_idx))]

        unique_node = list(set(data_edge[0] + data_edge[1]))
        unique_node = sorted(unique_node)
        old_new_dict = {}
        cnt = 0
        for node in unique_node:
            old_new_dict[node] = cnt
            cnt += 1
        
        l1 = []
        l2 = []
        for l1_ele, l2_ele in zip(data_edge[0], data_edge[1]):
            l1.append(old_new_dict[l1_ele])
            l2.append(old_new_dict[l2_ele])
        data_edge = [l1,l2]

        tr_data_mask = np.ones((len(data_x),)) # train mask 
        val_data_mask = np.zeros((len(data_x),)) # validation mask
        te_data_mask = np.zeros((len(data_x),)) # test mask
        mask_node_idx_list = np.argwhere(np.asarray(data_order) == target_order)

        if len(mask_node_idx_list) == 1:
            mask_node_idx_list = [int(mask_node_idx_list[0])]
        else:
            mask_node_idx_list = mask_node_idx_list.squeeze()


        if not len(mask_node_idx_list) == 0:
            # train에 사용하는 mask
            tr_mask_node_idx_list = np.random.choice(np.asarray(mask_node_idx_list),
                                            size=int(len(mask_node_idx_list)*te_val_ratio),
                                            replace=False)
            tr_data_mask[tr_mask_node_idx_list] = 0
            
            # train 하고 남은 것 중, 절반은 val, 절반은 test
            val_mask_node_idx_list = np.random.choice(np.asarray(tr_mask_node_idx_list),
                                            size=int(len(tr_mask_node_idx_list)*0.5),
                                            replace=False)
            if len(val_mask_node_idx_list) > 0:
                val_data_mask[val_mask_node_idx_list] = 1
            te_mask_node_idx_list = np.asarray(list(set(tr_mask_node_idx_list) - set(val_mask_node_idx_list)))
            if len(te_mask_node_idx_list) > 0:
                te_data_mask[te_mask_node_idx_list] = 1

            assert len(data_x) == np.sum(tr_data_mask) + np.sum(val_data_mask) + np.sum(te_data_mask)

        new_data_x_list.append(data_x)
        new_data_edge_list.append(data_edge)
        new_data_y_list.append(data_y)
        new_data_order_list.append(data_order)
        new_data_tr_mask_list.append(tr_data_mask)
        new_data_val_mask_list.append(val_data_mask)
        new_data_te_mask_list.append(te_data_mask)

    assert len(new_data_x_list) == len(new_data_edge_list) == len(new_data_y_list) == len(new_data_order_list) == len(new_data_tr_mask_list)
    print(f"Exist {len(new_data_x_list)} graphs after second step....(Masking)")

    return new_data_x_list, new_data_edge_list, new_data_y_list, new_data_order_list, new_data_tr_mask_list, new_data_val_mask_list, new_data_te_mask_list



def del_zero_edge(new_data_x_list, new_data_edge_list, new_data_y_list, new_data_order_list,
                new_data_tr_mask_list, new_data_val_mask_list, new_data_te_mask_list,
                num_graph):
    final_data_x_list = []
    final_data_edge_list = []
    final_data_y_list = []
    final_data_order_list = []
    final_data_tr_mask_list = []
    final_data_val_mask_list = []
    final_data_te_mask_list = []
    for itr in range(num_graph):
        if len(new_data_edge_list[itr][0]) >= 2:
            final_data_x_list.append(new_data_x_list[itr])
            final_data_edge_list.append(new_data_edge_list[itr])
            final_data_y_list.append(new_data_y_list[itr])
            final_data_order_list.append(new_data_order_list[itr])
            final_data_tr_mask_list.append(new_data_tr_mask_list[itr])
            final_data_val_mask_list.append(new_data_val_mask_list[itr])
            final_data_te_mask_list.append(new_data_te_mask_list[itr])

    
    assert len(final_data_x_list)==len(final_data_edge_list)==len(final_data_y_list)==len(final_data_order_list)==len(final_data_tr_mask_list)
    print(f"Exist {len(final_data_x_list)} graphs after third step....(Deleting zero edge graphs)")
    return final_data_x_list, final_data_edge_list, final_data_y_list, final_data_order_list, final_data_tr_mask_list, final_data_val_mask_list, final_data_te_mask_list



def check_graphs(final_data_x_list, final_data_edge_list, final_data_y_list, final_data_order_list,
                final_data_tr_mask_list, final_data_val_mask_list, final_data_te_mask_list):
    for itr in range(len(final_data_x_list)):
        num_node = len(final_data_x_list[itr])
        num_uniqe_node_edge = len(set(final_data_edge_list[itr][0]))
        num_y = len(final_data_y_list[itr])
        num_order = len(final_data_order_list[itr])
        num_tr_mask = len(final_data_tr_mask_list[itr])
        num_val_mask = len(final_data_val_mask_list[itr])
        num_te_mask = len(final_data_te_mask_list[itr])

        assert num_node == num_order == num_tr_mask == num_val_mask == num_te_mask == num_y
        assert num_uniqe_node_edge >= 1



def to_tensor(final_data_x_list, final_data_edge_list, final_data_y_list, final_data_order_list,
                        final_data_tr_mask_list, final_data_val_mask_list, final_data_te_mask_list,
                        device):
    final_data_list = []
    for data in zip(final_data_x_list, final_data_edge_list, final_data_y_list, 
                    final_data_tr_mask_list, final_data_val_mask_list, final_data_te_mask_list,
                    final_data_order_list):
        data_x = torch.FloatTensor(data[0]).to(device)
        data_edge = torch.LongTensor(data[1]).to(device)
        data_y = torch.FloatTensor(data[2]).squeeze().to(device)
        # data_y = torch.FloatTensor(data[2]).to(device)

        data_tr_mask = torch.LongTensor(data[3]).to(device)
        data_tr_mask = data_tr_mask.type(torch.bool)

        data_val_mask = torch.LongTensor(data[4]).to(device)
        data_val_mask = data_val_mask.type(torch.bool)

        data_te_mask = torch.LongTensor(data[5]).to(device)
        data_te_mask = data_te_mask.type(torch.bool)

        data_order = data[-1]
        
        final_data_list.append([data_x, data_edge, data_y, 
                        data_tr_mask, data_val_mask, data_te_mask, data_order])
    
    print(f"Finally exist {len(final_data_list)} graphs after all steps!!")

    return final_data_list
# ---------------------------------------------------------------------------------------------



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
def train(data, model, optimizer, criterion):
    model.train()
    binary_X, cont_X, cat_X, y = data
    data_x = torch.cat((binary_X, cont_X, cat_X), dim=1).cuda()
    y=y.cuda()
    optimizer.zero_grad()
    batch_num = data_x.shape[0]
    out = model(data_x).squeeze()

    loss = criterion(out, y)
    if not torch.isnan(loss):
      loss.backward()
      optimizer.step()
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
    out = model(data_x).squeeze()

    loss = eval_criterion(out, y)
    if not torch.isnan(loss):
        return loss.item(),batch_num, out, y
    else:
        return 0,batch_num, out, y
    

def cal_group_loss(ground_truth_list, predicted_list, eval_criterion="MAE"):
    group_loss_dict = {}
    mean_group_loss = 0.
    for i in np.unique(ground_truth_list):
        mask = (ground_truth_list == i)

        if eval_criterion == "MAE":
            val_group_loss = mean_absolute_error(ground_truth_list[mask], predicted_list[mask])
        elif eval_criterion == "RMSE":
            val_group_loss = mean_squared_error(ground_truth_list[mask], predicted_list[mask])
            val_group_loss = np.sqrt(val_group_loss)

        mean_group_loss += val_group_loss
        
        group_loss_dict[i] = val_group_loss

    mean_group_loss /= len(group_loss_dict)

    return group_loss_dict, mean_group_loss


## Test ----------------------------------------------------------------------------------------
@torch.no_grad()
def test(data, model, eval_criterion):    
    model.eval()
    binary_X, cont_X, cat_X, y = data
    y=y.cuda()
    data_x = torch.cat((binary_X, cont_X, cat_X), dim=1).cuda()

    batch_num = data_x.shape[0]
    out = model(data_x).squeeze()

    loss = eval_criterion(out, y)
    if not torch.isnan(loss):
        return loss.item(),batch_num, out, y
    else:
        return 0,batch_num, out, y

