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
    def __init__(self, input_size=128, hidden_size=64, output_size=2, drop_out=0.0, apply_embedding=True):
        super().__init__()
        if not apply_embedding:
            input_size = 12
        self.embedding = TableEmbedding(128, apply_embedding = apply_embedding)
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        # self.fc3 = nn.Linear(hidden_size, output_size, bias=True)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, cont_p, cont_c, cat_p, cat_c, len):
        x = self.embedding(cont_p, cont_c, cat_p, cat_c, len)
        x = F.relu(self.fc1(x))
        x = self.drop_out(F.relu(self.fc2(x)))
        # x = F.relu(self.fc3(x))
        return x

class LinearRegression(torch.nn.Module):
    def __init__(self, input_size=128, out_channels=2, apply_embedding=True):
        super().__init__()
        if not apply_embedding:
            input_size = 12
        self.embedding = TableEmbedding(128, apply_embedding = apply_embedding)
        self.linear1 = torch.nn.Linear(input_size, out_channels)

    def forward(self, cont_p, cont_c, cat_p, cat_c, len):
        x = self.embedding(cont_p, cont_c, cat_p, cat_c, len)
        x = self.linear1(x)
        return x

class TableEmbedding(torch.nn.Module):
    def __init__(self, output_size=128, apply_embedding=True):
        super().__init__()
        self.apply_embedding = apply_embedding
        if apply_embedding:
            print("Embedding applied to data")
            emb_hidden_dim = output_size//4
            self.cont_p_NN = nn.Sequential(nn.Linear(3, emb_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(emb_hidden_dim, emb_hidden_dim))
            self.cont_c_NN = nn.Sequential(nn.Linear(2, emb_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(emb_hidden_dim, emb_hidden_dim))
        self.lookup_gender  = nn.Embedding(2, emb_hidden_dim).to('cuda:0')
        self.lookup_korean  = nn.Embedding(2, emb_hidden_dim).to('cuda:0')
        self.lookup_primary  = nn.Embedding(2, emb_hidden_dim).to('cuda:0')
        self.lookup_job  = nn.Embedding(11, emb_hidden_dim).to('cuda:0')
        self.lookup_rep  = nn.Embedding(34, emb_hidden_dim).to('cuda:0')
        self.lookup_place  = nn.Embedding(19, emb_hidden_dim).to('cuda:0')
        self.lookup_add  = nn.Embedding(31, emb_hidden_dim).to('cuda:0')

    def forward(self, cont_p, cont_c, cat_p, cat_c, len):
        if self.apply_embedding:
            cont_p = self.cont_p_NN(cont_p)
            cont_c = self.cont_c_NN(cont_c)
        a1_embs = self.lookup_gender(cat_p[:,:,0].to(torch.int))
        a2_embs = self.lookup_korean(cat_p[:,:,1].to(torch.int))
        a3_embs = self.lookup_primary(cat_p[:,:,2].to(torch.int))
        a4_embs = self.lookup_job(cat_p[:,:,3].to(torch.int))
        a7_embs = self.lookup_rep(cat_p[:,:,4].to(torch.int))
        a5_embs = self.lookup_place(cat_c[:,:,0].to(torch.int))
        a6_embs = self.lookup_add(cat_c[:,:,1].to(torch.int))
        # categorical datas embedding 평균
        cat_p = torch.mean(torch.stack([a1_embs, a2_embs, a3_embs, a4_embs, a5_embs]), axis=0)
        cat_c = torch.mean(torch.stack([a6_embs, a7_embs]), axis=0)
        x = torch.cat((cat_p, cat_c, cont_p, cont_c), dim=2)
        sliced_tensors = []
        for i in range(x.shape[0]):
            m = len[i].item()
            sliced_tensor = x[i, :m, :]  
            sliced_tensor = torch.mean(sliced_tensor, dim=0)
            sliced_tensors.append(sliced_tensor)
        x = torch.stack(sliced_tensors, dim=0)
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