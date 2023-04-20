import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import pdb
    

class MLPRegressor(nn.Module):
    '''
    embedding 태워서 input으로 넣도록 수정 필요
    '''
    def __init__(self, input_size=128, hidden_size=64, output_size=2, drop_out=0.0):
        super().__init__()
        self.cont_NN = nn.Sequential(nn.Linear(5, 64),
                                     nn.ReLU())
        emb_hidden_dim = hidden_size
        self.lookup_gender  = nn.Embedding(2, emb_hidden_dim).to('cuda:0')
        self.lookup_korean  = nn.Embedding(2, emb_hidden_dim).to('cuda:0')
        self.lookup_primary  = nn.Embedding(2, emb_hidden_dim).to('cuda:0')
        self.lookup_job  = nn.Embedding(11, emb_hidden_dim).to('cuda:0')
        self.lookup_place  = nn.Embedding(19, emb_hidden_dim).to('cuda:0')
        self.lookup_add  = nn.Embedding(31, emb_hidden_dim).to('cuda:0')
        self.lookup_rep  = nn.Embedding(34, emb_hidden_dim).to('cuda:0')
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        # self.fc3 = nn.Linear(hidden_size, output_size, bias=True)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, cont_x, cat_x, len):
        a1_embs = self.lookup_gender(cat_x[:,:,0].to(torch.int))
        a2_embs = self.lookup_korean(cat_x[:,:,1].to(torch.int))
        a3_embs = self.lookup_primary(cat_x[:,:,2].to(torch.int))
        a4_embs = self.lookup_job(cat_x[:,:,3].to(torch.int))
        a5_embs = self.lookup_place(cat_x[:,:,4].to(torch.int))
        a6_embs = self.lookup_add(cat_x[:,:,5].to(torch.int))
        a7_embs = self.lookup_rep(cat_x[:,:,6].to(torch.int))
        # categorical datas embedding 평균
        cat_embs = torch.mean(torch.stack([a1_embs, a2_embs, a3_embs, a4_embs, a5_embs,
                                              a6_embs, a7_embs]), axis=0)
        
        cont_x = self.cont_NN(cont_x)
        x = torch.cat((cat_embs, cont_x), dim=2)
        # cont, category data embedding 합쳐서 datalen 길이만큼 자른다음 각각에 대해서 평균내서 크기 맞춘다음 다시 합치기
        sliced_tensors = []
        for i in range(a1_embs.shape[0]):
            m = len[i].item()
            sliced_tensor = x[i, :m, :]  
            sliced_tensor = torch.mean(sliced_tensor, dim=0)
            sliced_tensors.append(sliced_tensor)
        x = torch.stack(sliced_tensors, dim=0)
        
        x = F.relu(self.fc1(x))
        x = self.drop_out(F.relu(self.fc2(x)))
        # x = F.relu(self.fc3(x))
        return x


class LinearRegression(torch.nn.Module):
    def __init__(self, input_size=24, out_channels=2):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, out_channels)

    def forward(self, x):
        x = self.linear1(x)
        return x


class TSTransformer(torch.nn.Module):
    def __init__(self, hidden_size=16, output_size=16):
        super().__init__()
        self.linear_cont_P = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.embedding_p = nn.Embedding(5, output_size)
        self.linear_cont_C = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.embedding_c = nn.Embedding(2, output_size)
        # self.transformer = PETransformer(nhead=16, num_encoder_layers=12)
        
    def forward(self, cont_P, disc_P, cont_C, disc_C):
        cont_P_out = self.linear_cont_P(cont_P)
        disc_P_out = self.embedding_p(disc_P)
        p_out = torch.cat((cont_P_out.unsqueeze(1), disc_P_out.unsqueeze(1)), dim=1)
        p_out = torch.mean(p_out, dim=1)
        
        cont_C_out = self.linear_cont_C(cont_C)
        disc_C_out = self.embedding_c(disc_C)
        c_out = torch.cat((cont_C_out.unsqueeze(1), disc_C_out.unsqueeze(1)), dim=1)
        c_out = torch.mean(c_out, dim=1)
        
        out = torch.add(p_out, c_out)
        
        return out

class PETransformer(nn.Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Modified Positional Encoding
        self.pos_encoder = nn.Embedding(d_model, d_model)

    def forward(self, src, tgt):
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

class SVR(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass