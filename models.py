import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from utils import reduction_cluster, reparametrize
import pdb
import warnings

warnings.filterwarnings("ignore", "Converting mask without torch.bool dtype to bool")

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
                x = layer(x)  
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
        if not disable_pe:
            self.positional_embedding  = nn.Embedding(5, output_size)

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
    

class CEVAEEmbedding(torch.nn.Module):
    '''
        output_size : embedding output의 크기
        disable_embedding : 연속 데이터의 임베딩 유무
        disable_pe : transformer의 sequance 기준 positional encoding add 유무
        reduction : "mean" : cluster 별 평균으로 reduction
                    "date" : cluster 내 date 평균으로 reduction
    '''
    def __init__(self, output_size=128, disable_embedding=False, disable_pe=True, reduction="date"):
        super().__init__()
        self.reduction = reduction
        self.disable_embedding = disable_embedding
        self.disable_pe = disable_pe
        activation = nn.ELU()
        if not disable_embedding:
            print("Embedding applied to data")
            nn_dim = emb_hidden_dim = emb_dim = output_size//4
            self.cont_p_NN = nn.Sequential(nn.Linear(3, emb_hidden_dim),
                                        activation,
                                        nn.Linear(emb_hidden_dim, nn_dim))
            self.cont_c_NN = nn.Sequential(nn.Linear(1, emb_hidden_dim),
                                        activation,
                                        nn.Linear(emb_hidden_dim, nn_dim))
        else:
            emb_dim_p = 5
            emb_dim_c = 2
        self.lookup_gender  = nn.Embedding(2, emb_dim)
        self.lookup_korean  = nn.Embedding(2, emb_dim)
        self.lookup_primary  = nn.Embedding(2, emb_dim)
        self.lookup_job  = nn.Embedding(11, emb_dim)
        self.lookup_rep  = nn.Embedding(34, emb_dim)
        self.lookup_place  = nn.Embedding(19, emb_dim)
        self.lookup_add  = nn.Embedding(31, emb_dim)
        if not disable_pe:
            self.positional_embedding  = nn.Embedding(5, output_size)

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
    

class CEVAETransformer(nn.Module):
    '''
        input_size : TableEmbedding 크기
        hidden_size : Transformer Encoder 크기
        output_size : y, d (2)
        num_layers : Transformer Encoder Layer 개수
        num_heads : Multi Head Attention Head 개수
        drop_out : DropOut 정도
        disable_embedding : 연속 데이터 embedding 여부
    '''
    def __init__(self, input_size, hidden_size, num_layers, num_heads, drop_out):
        super(CEVAETransformer, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size,
            nhead=num_heads,
            dim_feedforward=hidden_size, 
            dropout=drop_out,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_size))

    def forward(self, x, val_len):
        cls_token = self.cls_token.expand(x.size(0), -1, -1) 
        input_with_cls = torch.cat([cls_token, x], dim=1)
        mask = (torch.arange(input_with_cls.size(1)).expand(input_with_cls.size(0), -1).cuda() < val_len.unsqueeze(1)).cuda()
        output = self.transformer_encoder(input_with_cls, src_key_padding_mask=mask.bool())  
        cls_emb = output[:, 0, :]  
        return cls_emb
    
# ------------------------ for yd only --embedding_dim 64 --shared_layers 1 --pred_layers 1 -n 300
# class CEVAE_Encoder(nn.Module):
#     def __init__(self, input_dim, latent_dim, hidden_dim=128, shared_layers=3, pred_layers=3):
#         super(CEVAE_Encoder, self).__init__()
                
#         # Shared layers
#         layers = []
#         for _ in range(shared_layers):
#             layers.append(nn.Linear(input_dim if len(layers) == 0 else hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#         self.fc_shared = nn.Sequential(*layers)
        
#         # Latent variable z distribution parameters (mean and log variance)
#         self.fc_mu = nn.Linear(hidden_dim, latent_dim)
#         self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
#         # Predict y, d with MLP
#         yd_layers = []
#         for _ in range(pred_layers):
#             yd_layers.append(nn.Linear(hidden_dim if len(yd_layers) == 0 else hidden_dim, hidden_dim))
#             yd_layers.append(nn.ReLU())
#         yd_layers.append(nn.Linear(hidden_dim, 2))
#         self.fc_yd = nn.Sequential(*yd_layers)
        
#         # Predict t with MLP
#         t_layers = []
#         for _ in range(pred_layers):
#             t_layers.append(nn.Linear(hidden_dim if len(t_layers) == 0 else hidden_dim, hidden_dim))
#             t_layers.append(nn.ReLU())
#         t_layers.append(nn.Linear(hidden_dim, 7))
#         self.fc_t = nn.Sequential(*t_layers)
    
#     def forward(self, x):
#         h_shared = self.fc_shared(x)
        
#         mu = self.fc_mu(h_shared)
#         logvar = self.fc_logvar(h_shared)
        
#         yd_pred = self.fc_yd(h_shared)
#         t_pred = self.fc_t(h_shared)
        
#         return mu, logvar, yd_pred, t_pred
    
# class CEVAE_Encoder(nn.Module): # -- [train all, divided by t]
#     def __init__(self, input_dim, latent_dim, hidden_dim=128, shared_layers=3, pred_layers=3, t_classes=7):
#         super(CEVAE_Encoder, self).__init__()
#         # Warm up layer
#         self.warm_up = nn.Linear(input_dim, 2) # predict only y and d

#         # Shared layers
#         layers = []
#         for _ in range(shared_layers):
#             layers.append(nn.Linear(input_dim if len(layers) == 0 else hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())
#         self.fc_shared = nn.Sequential(*layers)
        
#         # Predict t with MLP
#         t_layers = []
#         for _ in range(pred_layers):
#             t_layers.append(nn.Linear(hidden_dim if len(t_layers) == 0 else hidden_dim, hidden_dim))
#             t_layers.append(nn.ReLU())
#         t_layers.append(nn.Linear(hidden_dim, t_classes))
#         self.fc_t = nn.Sequential(*t_layers)
        
#         # Predict yd based on t
#         self.yd_nns = nn.ModuleList([
#             self._build_yd_predictor(hidden_dim, pred_layers) for _ in range(t_classes)
#         ])
        
#         # Calculate z (mu and logvar) based on t, yd, and x
#         self.z_nns = nn.ModuleList([
#             self._build_z_predictor(hidden_dim + 2, latent_dim, hidden_dim, pred_layers) for _ in range(t_classes)
#         ])

#     def _build_yd_predictor(self, hidden_dim, pred_layers):
#         yd_layers = []
#         for _ in range(pred_layers):
#             yd_layers.append(nn.Linear(hidden_dim, hidden_dim))
#             yd_layers.append(nn.ReLU())
#         yd_layers.append(nn.Linear(hidden_dim, 2))
#         return nn.Sequential(*yd_layers)
    
#     def _build_z_predictor(self, input_dim, latent_dim, hidden_dim, pred_layers):
#         z_layers = []
#         for _ in range(pred_layers):
#             z_layers.append(nn.Linear(input_dim if len(z_layers) == 0 else hidden_dim, hidden_dim))
#             z_layers.append(nn.ReLU())
#         z_layers.extend([nn.Linear(hidden_dim, latent_dim), nn.Linear(hidden_dim, latent_dim)])
#         return nn.ModuleDict({
#             'mu': nn.Sequential(*z_layers[:-1]),      # pred layers hidden 공유, final layer 분리
#             'logvar': nn.Sequential(*(z_layers[:-2] + [z_layers[-1]]))  
#         })

#     def forward(self, x, t_gt=None):
#         h_shared = self.fc_shared(x)
#         warm_yd = self.warm_up(x)
#         if t_gt==None:
#             t_pred = self.fc_t(h_shared)
#             t_class = t_pred.argmax(dim=1)
#         else:
#             t_class = t_gt
#             t_pred = None
#         yd_preds = [yd_nn(h_shared) for yd_nn in self.yd_nns]
#         yd_pred = torch.stack([yd_preds[i][idx] for idx, i in enumerate(t_class)], dim=0)

#         # Calculate mu and logvar based on t, yd, and x
#         h_combined = torch.cat([h_shared, yd_pred], dim=1)
#         z_preds = [z_nn['mu'](h_combined) for z_nn in self.z_nns]
#         mu = torch.stack([z_preds[i][idx] for idx, i in enumerate(t_class)], dim=0)
        
#         z_preds_logvar = [z_nn['logvar'](h_combined) for z_nn in self.z_nns]
#         logvar = torch.stack([z_preds_logvar[i][idx] for idx, i in enumerate(t_class)], dim=0)

#         return mu, logvar, yd_pred, t_pred, warm_yd


class CEVAE_Encoder(nn.Module): # -- [train all, conditioned by t]
    def __init__(self, input_dim, latent_dim, hidden_dim=128, shared_layers=3, pred_layers=3, t_pred_layers=3, t_embed_dim=8, yd_embed_dim=8, drop_out=0, t_classes=None, skip_hidden=False):
        super(CEVAE_Encoder, self).__init__()
        # Embedding for continuous t
        # Warm up layer
        self.warm_up = nn.Linear(input_dim, 2) # predict only y and d
        self.t_embedding = nn.Linear(1, t_embed_dim)
        self.yd_embedding = nn.Linear(2, yd_embed_dim)
        activation = nn.ELU()
        self.skip_hidden = skip_hidden
        # Predict t with MLP
        t_layers = []
        for _ in range(t_pred_layers):
            t_layers.append(nn.Linear(hidden_dim if len(t_layers) == 0 else hidden_dim, hidden_dim))
            t_layers.append(activation)
            t_layers.append(nn.Dropout(drop_out))
        t_layers.append(nn.Linear(hidden_dim, 1))
        self.fc_t = nn.Sequential(*t_layers)

        if not skip_hidden:
            # Shared layers
            layers = []
            for _ in range(shared_layers):
                layers.append(nn.Linear(input_dim + t_embed_dim if len(layers) == 0 else hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(nn.Dropout(drop_out))
            self.fc_shared = nn.Sequential(*layers)

        # Latent variable z distribution parameters (mean and log variance)
        self.fc_mu = nn.Linear(hidden_dim + t_embed_dim + yd_embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim + t_embed_dim + yd_embed_dim, latent_dim)
        
        # Predict y, d with MLP (now conditioned on t)
        yd_layers = []
        for _ in range(pred_layers):
            yd_layers.append(nn.Linear(hidden_dim + t_embed_dim if len(yd_layers) == 0 else hidden_dim, hidden_dim))
            yd_layers.append(activation)
            yd_layers.append(nn.Dropout(drop_out))
        yd_layers.append(nn.Linear(hidden_dim, 2))
        self.fc_yd = nn.Sequential(*yd_layers)
    
    def forward(self, x, t_gt=None):
        warm_yd = self.warm_up(x)

        # Embed the continuous t
        t_pred = self.fc_t(x) if t_gt == None else t_gt.float().unsqueeze(1)
        t_embed = self.t_embedding(t_pred)

        # Concatenate input x and embedded t
        h_shared = x if self.skip_hidden else self.fc_shared(torch.cat([x, t_embed], dim=1))

        # Predict y, d conditioned on t
        yd_pred = self.fc_yd(torch.cat([h_shared, t_embed], dim=1))
        yd_embed = self.yd_embedding(yd_pred)

        # Concatenate shared features and embedded t for mu and logvar
        h_tyd = torch.cat([h_shared, t_embed, yd_embed], dim=1)

        # Pred mu, logvar of Z
        mu = self.fc_mu(h_tyd)
        logvar = self.fc_logvar(h_tyd)
        
        return mu, logvar, yd_pred, t_pred.squeeze(), warm_yd

# class CEVAE_Decoder(nn.Module): [train seperated yd]
#     def __init__(self, latent_dim, output_dim, hidden_dim=128, num_layers=2, t_classes=7):
#         super(CEVAE_Decoder, self).__init__()
        
#         # Predict t from z
#         t_layers = []
#         for _ in range(num_layers):
#             t_layers.append(nn.Linear(latent_dim if len(t_layers) == 0 else hidden_dim, hidden_dim))
#             t_layers.append(nn.ReLU())
#         t_layers.append(nn.Linear(hidden_dim, t_classes))
#         self.fc_t = nn.Sequential(*t_layers)
        
#         # Predict y,d based on z and t
#         self.yd_nns = nn.ModuleList([
#             self._build_yd_predictor(latent_dim, hidden_dim, num_layers) for _ in range(t_classes)
#         ])
        
#         # Directly predict x from z
#         x_layers = []
#         for _ in range(num_layers):
#             x_layers.append(nn.Linear(latent_dim if len(x_layers) == 0 else hidden_dim, hidden_dim))
#             x_layers.append(nn.ReLU())
#         x_layers.append(nn.Linear(hidden_dim, output_dim))
#         self.fc_x = nn.Sequential(*x_layers)
    
#     def _build_yd_predictor(self, latent_dim, hidden_dim, num_layers):
#         yd_layers = []
#         for _ in range(num_layers):
#             yd_layers.append(nn.Linear(latent_dim if len(yd_layers) == 0 else hidden_dim, hidden_dim))
#             yd_layers.append(nn.ReLU())
#         yd_layers.append(nn.Linear(hidden_dim, 2))  # Assuming y,d output is of size 2
#         return nn.Sequential(*yd_layers)
    
#     def forward(self, z, t_gt=None):
#         # Predict t from z
#         if t_gt==None:
#             t_pred = self.fc_t(z)
#             t_class = t_pred.argmax(dim=1)
#         else:
#             t_class = t_gt
#             t_pred = None
#         yd_preds = [yd_nn(z) for yd_nn in self.yd_nns]
#         yd_pred = torch.stack([yd_preds[i][idx] for idx, i in enumerate(t_class)], dim=0)
        
#         # Directly predict x from z
#         x_pred = self.fc_x(z)
        
#         return t_pred, yd_pred, x_pred

class CEVAE_Decoder(nn.Module): #  [conditioned t, train overall yd]
    def __init__(self, latent_dim, output_dim, hidden_dim=128, t_pred_layers=2, shared_layers=2, t_embed_dim=16, drop_out=0, t_classes=7, skip_hidden=False):
        super(CEVAE_Decoder, self).__init__()
        self.skip_hidden = skip_hidden
        self.t_embedding = nn.Linear(1, t_embed_dim)
        activation = nn.ELU()
        # Predict t from z
        t_layers = []
        for _ in range(t_pred_layers):
            t_layers.append(nn.Linear(latent_dim if len(t_layers) == 0 else hidden_dim, hidden_dim))
            t_layers.append(activation)
            t_layers.append(nn.Dropout(drop_out))
        t_layers.append(nn.Linear(hidden_dim, 1))
        self.fc_t = nn.Sequential(*t_layers)
        
        if not skip_hidden:
            # Shared layers
            layers = []
            for _ in range(shared_layers):
                layers.append(nn.Linear(latent_dim + t_embed_dim if len(layers) == 0 else hidden_dim, hidden_dim))
                layers.append(activation)
                layers.append(nn.Dropout(drop_out))
            self.fc_shared = nn.Sequential(*layers)

        self.x_head = nn.Linear(latent_dim + t_embed_dim if skip_hidden else hidden_dim + t_embed_dim, output_dim)
        self.yd_head = nn.Linear(latent_dim + t_embed_dim if skip_hidden else hidden_dim + t_embed_dim, 2)
    
    def forward(self, z, t_gt=None):
        # Predict t from z
        t_pred = self.fc_t(z) if t_gt == None else t_gt.float().unsqueeze(1)
        
        t_embed = self.t_embedding(t_pred)
        h = z if self.skip_hidden else self.fc_shared(torch.cat([z, t_embed], dim=1))

        # Directly predict x from z
        x_pred = self.x_head(torch.cat([h, t_embed], dim=1))
        yd_pred = self.yd_head(torch.cat([h, t_embed], dim=1))
        
        return t_pred.squeeze(), yd_pred, x_pred

class CEVAE_det(nn.Module):
    def __init__(self, embedding_dim, latent_dim=64, encoder_hidden_dim=128, encoder_shared_layers=3, encoder_pred_layers=1, transformer_layers=3, drop_out=0.0, num_heads=2, t_pred_layers=3, t_classes=7, t_embed_dim=16, yd_embed_dim=16, skip_hidden=False):
        super(CEVAE_det, self).__init__()
        self.x_emb = CEVAEEmbedding(output_size=embedding_dim)
        self.transformer_encoder = CEVAETransformer(input_size=embedding_dim, hidden_size=latent_dim, num_layers=transformer_layers, num_heads=num_heads, drop_out=drop_out)
        self.encoder = CEVAE_Encoder(input_dim=embedding_dim, latent_dim=latent_dim, hidden_dim=encoder_hidden_dim, shared_layers=encoder_shared_layers, t_pred_layers=t_pred_layers , pred_layers=encoder_pred_layers, t_classes=t_classes, t_embed_dim=t_embed_dim, yd_embed_dim=yd_embed_dim, skip_hidden=skip_hidden)
        self.decoder = CEVAE_Decoder(latent_dim=latent_dim, output_dim=embedding_dim, hidden_dim=encoder_hidden_dim, t_pred_layers=t_pred_layers, shared_layers=encoder_shared_layers, t_classes=t_classes, t_embed_dim=t_embed_dim, skip_hidden=skip_hidden)

    def forward(self, cont_p, cont_c, cat_p, cat_c, _len, diff, t_gt=None):
        x = self.x_emb(cont_p, cont_c, cat_p, cat_c, _len, diff)
        x_transformed = self.transformer_encoder(x, _len)
        z_mu, z_logvar, enc_yd_pred, enc_t_pred, warm_yd = self.encoder(x_transformed, t_gt)
        
        # Sample z using reparametrization trick
        z = reparametrize(z_mu, z_logvar)
        
        # Decode z to get the reconstruction of x
        dec_t_pred, dec_yd_pred, x_reconstructed = self.decoder(z, t_gt)
        
        return x_transformed, x_reconstructed, z_mu, z_logvar, (enc_yd_pred, enc_t_pred), (dec_yd_pred, dec_t_pred), warm_yd
    

class CEVAE_debug(nn.Module):
    def __init__(self, embedding_dim, latent_dim=64, encoder_hidden_dim=128, encoder_shared_layers=3, encoder_pred_layers=1, transformer_layers=3, drop_out=0.0, num_heads=2, t_pred_layers=3, t_classes=7, t_embed_dim=16, yd_embed_dim=16, skip_hidden=False):
        super(CEVAE_debug, self).__init__()
        activation=nn.ELU()
        self.x_emb = CEVAEEmbedding(output_size=embedding_dim)
        self.transformer_encoder = CEVAETransformer(input_size=embedding_dim, hidden_size=latent_dim, num_layers=transformer_layers, num_heads=num_heads, drop_out=drop_out)
        self.fc = nn.Linear(embedding_dim,2)
        
        fc_layers = []
        for _ in range(3):
            fc_layers.append(nn.Linear(embedding_dim if len(fc_layers) == 0 else latent_dim, latent_dim))
            fc_layers.append(activation)
            fc_layers.append(nn.Dropout(drop_out))
        fc_layers.append(nn.Linear(latent_dim,2))
        self.fc_mlp = nn.Sequential(*fc_layers)
        
        fc_layers_d = []
        for _ in range(3):
            fc_layers_d.append(nn.Linear(embedding_dim if len(fc_layers_d) == 0 else latent_dim, latent_dim))
            fc_layers_d.append(activation)
            fc_layers_d.append(nn.Dropout(drop_out))
        fc_layers_d.append(nn.Linear(latent_dim,1))
        self.fc_mlp_d = nn.Sequential(*fc_layers_d)

    def forward(self, cont_p, cont_c, cat_p, cat_c, _len, diff, t_gt=None):
        x = self.x_emb(cont_p, cont_c, cat_p, cat_c, _len, diff)
        x_transformed = self.transformer_encoder(x, _len)
        # yd = self.fc(x_transformed)
        yd = self.fc_mlp(x_transformed)
        # d = self.fc_mlp_d(x_transformed)
        return yd
        # return torch.stack([y,d],dim=1).squeeze()

class CETransformer(nn.Module):
    def __init__(self, embedding_dim, latent_dim=64, mlp_hidden_dim=64, mlp_layers=3, transformer_layers=3, drop_out=0.0, num_heads=2, t_embed_dim=16, yd_embed_dim=16, use_raw_ydt=False, use_cls=True):
        super(CETransformer, self).__init__()
        self.use_raw_ydt = use_raw_ydt
        self.use_cls = use_cls

        self.x_emb = CEVAEEmbedding(output_size=embedding_dim)
        
        self.enc_t_embedding = nn.Linear(1, embedding_dim)
        self.enc_yd_embedding = nn.Linear(2, embedding_dim)
        
        self.dec_t_embedding = nn.Linear(1, embedding_dim)
        self.dec_yd_embedding = nn.Linear(2, embedding_dim)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim, 
            dropout=drop_out,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, transformer_layers)
        
        self.decoder_layer = TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim,
            dropout=drop_out,
            batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(self.decoder_layer, transformer_layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        
        self.enc_t_fc = MLP(input_dim=embedding_dim, hidden_dim=mlp_hidden_dim, output_dim=1, num_layers=mlp_layers, dropout_rate=drop_out)
        self.enc_yd_fc = MLP(input_dim=embedding_dim + 1 if use_raw_ydt else embedding_dim + t_embed_dim, hidden_dim=mlp_hidden_dim, output_dim=2, num_layers=mlp_layers, dropout_rate=drop_out)
        
        self.dec_t_fc = MLP(input_dim=embedding_dim, hidden_dim=mlp_hidden_dim, output_dim=1, num_layers=mlp_layers, dropout_rate=drop_out)
        self.dec_yd_fc = MLP(input_dim=embedding_dim + 1 if use_raw_ydt else embedding_dim + t_embed_dim, hidden_dim=mlp_hidden_dim, output_dim=2, num_layers=mlp_layers, dropout_rate=drop_out)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, cont_p, cont_c, cat_p, cat_c, val_len, diff, t_gt=None):
        # TODO: use cls must be true in this code
        x = self.x_emb(cont_p, cont_c, cat_p, cat_c, val_len, diff)

        ori_x = cls_token = self.cls_token.expand(x.size(0), -1, -1) 
        input_with_cls = torch.cat([cls_token, x], dim=1)
        if not self.use_cls:
            ori_x = input_with_cls
        
        enc_t_pred = self.enc_t_fc(ori_x)
        enc_t_emb = self.enc_t_embedding(enc_t_pred)
        enc_yd_pred = self.enc_yd_fc(ori_x + enc_t_emb)

        src_mask = (torch.arange(input_with_cls.size(1)).expand(input_with_cls.size(0), -1).cuda() < val_len.unsqueeze(1)).cuda()

        encoder_output = self.transformer_encoder(input_with_cls, src_key_padding_mask=src_mask)

        # use cls embedding as Z  
        z = encoder_output[:, 0, :] 
        if not self.use_cls:
            z = encoder_output

        tgt_length = input_with_cls.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_length).cuda()

        # use decoder output as x_hat
        decoder_output = self.transformer_decoder(input_with_cls, encoder_output, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask, tgt_key_padding_mask=src_mask)
        recon_x = decoder_output[:, 0, :]
        if not self.use_cls:
            recon_x = decoder_output

        return z, recon_x, (enc_yd_pred, enc_t_pred), (dec_yd_pred, dec_t_pred), warm_yd
    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_rate=0.5):
        super(MLP, self).__init__()
        layers = []

        if num_layers == 1:
            # 단일 층인 경우
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            # 입력층
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

            # 은닉층
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

            # 출력층
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)