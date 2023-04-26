import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pdb
    

class MLPRegressor(nn.Module):
    def __init__(self, input_size=128, hidden_size=64, output_size=2, drop_out=0.0, disable_embedding=False):
        super().__init__()
        if disable_embedding:
            input_size = 12
        self.embedding = TableEmbedding(128, disable_embedding = disable_embedding)
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
    def __init__(self, input_size=128, out_channels=2, disable_embedding=False):
        super().__init__()
        if disable_embedding:
            input_size = 12
        self.embedding = TableEmbedding(128, disable_embedding = disable_embedding)
        self.linear1 = torch.nn.Linear(input_size, out_channels)

    def forward(self, cont_p, cont_c, cat_p, cat_c, len):
        x = self.embedding(cont_p, cont_c, cat_p, cat_c, len)
        x = self.linear1(x)
        return x

class TableEmbedding(torch.nn.Module):
    def __init__(self, output_size=128, disable_embedding=False):
        super().__init__()
        self.disable_embedding = disable_embedding
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
        self.lookup_gender  = nn.Embedding(2, emb_dim_p).to('cuda:0')
        self.lookup_korean  = nn.Embedding(2, emb_dim_p).to('cuda:0')
        self.lookup_primary  = nn.Embedding(2, emb_dim_p).to('cuda:0')
        self.lookup_job  = nn.Embedding(11, emb_dim_p).to('cuda:0')
        self.lookup_rep  = nn.Embedding(34, emb_dim_p).to('cuda:0')
        self.lookup_place  = nn.Embedding(19, emb_dim_c).to('cuda:0')
        self.lookup_add  = nn.Embedding(31, emb_dim_c).to('cuda:0')

    def forward(self, cont_p, cont_c, cat_p, cat_c, len):
        if not self.disable_embedding:
            cont_p = self.cont_p_NN(cont_p)
            cont_c = self.cont_c_NN(cont_c)
        a1_embs = self.lookup_gender(cat_p[:,:,0].to(torch.int))
        a2_embs = self.lookup_korean(cat_p[:,:,1].to(torch.int))
        a3_embs = self.lookup_primary(cat_p[:,:,2].to(torch.int))
        a4_embs = self.lookup_job(cat_p[:,:,3].to(torch.int))
        a5_embs = self.lookup_rep(cat_p[:,:,4].to(torch.int))
        a6_embs = self.lookup_place(cat_c[:,:,0].to(torch.int))
        a7_embs = self.lookup_add(cat_c[:,:,1].to(torch.int))
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

class Transformer(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output

def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generates an upper-triangular matrix of ``-inf``, with zeros on ``diag``."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)