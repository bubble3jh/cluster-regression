import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from utils import reduction_cluster
import pdb
    

class MLPRegressor(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, num_layers=3, output_size=2, drop_out=0.0, disable_embedding=False):
        super().__init__()
        self.num_layers = num_layers
        if disable_embedding:
            input_size = 12
        self.embedding = TableEmbedding(input_size, disable_embedding = disable_embedding)
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size, bias=True)])
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size, bias=True))
        self.layers.append(nn.Linear(hidden_size, output_size, bias=True))
        self.dropout = nn.Dropout(drop_out)
        
    def forward(self, cont_p, cont_c, cat_p, cat_c, len, diff_days):
        x = self.embedding(cont_p, cont_c, cat_p, cat_c, len, diff_days)
        for i, layer in enumerate(self.layers):
            if i == self.num_layers - 1:
                x = layer(x)  # No ReLU for the last layer
            else:
                x = self.dropout(F.relu(layer(x)))
        return x

class LinearRegression(torch.nn.Module):
    def __init__(self, input_size=128, out_channels=2, disable_embedding=False):
        super().__init__()
        if disable_embedding:
            input_size = 12
        self.embedding = TableEmbedding(input_size, disable_embedding = disable_embedding)
        self.linear1 = torch.nn.Linear(input_size, out_channels)

    def forward(self, cont_p, cont_c, cat_p, cat_c, len, diff_days):
        x = self.embedding(cont_p, cont_c, cat_p, cat_c, len, diff_days)
        x = self.linear1(x)
        return x

class Transformer(nn.Module):
    '''
        input_size : TableEmbedding 크기
        hidden_size : Transformer Encoder 크기
        output_size : y, d (2)
        num_layers : Transformer Encoder Layer 개수
        num_heads : Multi Head Attention Head 개수
        drop_out : DropOut 정도
        disable_embedding : 연속 데이터 embedding 여부
    '''
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, drop_out, disable_embedding):
        super(Transformer, self).__init__()
        
        self.embedding = TableEmbedding(output_size=input_size, disable_embedding = disable_embedding, disable_pe=False, reduction="date")
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_size))
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size, 
            dropout=drop_out,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)  

        self.init_weights()

    ## Transformer Weight 초기화 방법 ##
    def init_weights(self) -> None:
        initrange = 0.1
        for module in self.embedding.modules():
                if isinstance(module, nn.Linear) :
                    module.weight.data.uniform_(-initrange, initrange)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.Embedding):
                    module.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    def forward(self, cont_p, cont_c, cat_p, cat_c, val_len, diff_days):
        embedded = self.embedding(cont_p, cont_c, cat_p, cat_c, val_len, diff_days)
        cls_token = self.cls_token.expand(embedded.size(0), -1, -1) 
        input_with_cls = torch.cat([cls_token, embedded], dim=1)
        mask = (torch.arange(input_with_cls.size(1)).expand(input_with_cls.size(0), -1).cuda() < val_len.unsqueeze(1)).cuda()
        output = self.transformer_encoder(input_with_cls, src_key_padding_mask=mask.bool())  
        cls_output = output[:, 0, :]  
        regression_output = self.fc(cls_output) 
        
        return regression_output
    
class TableEmbedding(torch.nn.Module):
    '''
        output_size : embedding output의 크기
        disable_embedding : 연속 데이터의 임베딩 유무
        disable_pe : transformer의 sequance 기준 positional encoding add 유무
        reduction : "mean" : cluster 별 평균으로 reduction
                    "date" : cluster 내 date 평균으로 reduction
    '''
    def __init__(self, output_size=128, disable_embedding=False, disable_pe=True, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.disable_embedding = disable_embedding
        self.disable_pe = disable_pe
        if not disable_embedding:
            print("Embedding applied to data")
            nn_dim = emb_hidden_dim = emb_dim_c = emb_dim_p = output_size//4
            self.cont_p_NN = nn.Sequential(nn.Linear(3, emb_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(emb_hidden_dim, nn_dim))
            self.cont_c_NN = nn.Sequential(nn.Linear(2, emb_hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(emb_hidden_dim, nn_dim))
        else:
            emb_dim_p = 5
            emb_dim_c = 2
        self.lookup_gender  = nn.Embedding(2, emb_dim_p)
        self.lookup_korean  = nn.Embedding(2, emb_dim_p)
        self.lookup_primary  = nn.Embedding(2, emb_dim_p)
        self.lookup_job  = nn.Embedding(11, emb_dim_p)
        self.lookup_rep  = nn.Embedding(34, emb_dim_p)
        self.lookup_place  = nn.Embedding(19, emb_dim_c)
        self.lookup_add  = nn.Embedding(31, emb_dim_c)
        self.positional_embedding  = nn.Embedding(5, nn_dim + emb_hidden_dim + emb_dim_c + emb_dim_p)

    def forward(self, cont_p, cont_c, cat_p, cat_c, val_len, diff_days):
        if not self.disable_embedding:
            cont_p_emb = self.cont_p_NN(cont_p)
            cont_c_emb = self.cont_c_NN(cont_c)
        a1_embs = self.lookup_gender(cat_p[:,:,0].to(torch.int))
        a2_embs = self.lookup_korean(cat_p[:,:,1].to(torch.int))
        a3_embs = self.lookup_primary(cat_p[:,:,2].to(torch.int))
        a4_embs = self.lookup_job(cat_p[:,:,3].to(torch.int))
        a5_embs = self.lookup_rep(cat_p[:,:,4].to(torch.int))
        a6_embs = self.lookup_place(cat_c[:,:,0].to(torch.int))
        a7_embs = self.lookup_add(cat_c[:,:,1].to(torch.int))
        
        cat_p_emb = torch.mean(torch.stack([a1_embs, a2_embs, a3_embs, a4_embs, a5_embs]), axis=0)
        cat_c_emb = torch.mean(torch.stack([a6_embs, a7_embs]), axis=0)

        if not self.disable_embedding:
            x = torch.cat((cat_p_emb, cat_c_emb, cont_p_emb, cont_c_emb), dim=2)
        else:
            x = torch.cat((cat_p_emb, cat_c_emb, cont_p, cont_c), dim=2)
            
        if not self.disable_pe:
            x = x + self.positional_embedding(diff_days.int().squeeze(2))
        return reduction_cluster(x, diff_days, val_len, self.reduction)