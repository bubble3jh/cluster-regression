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
    def __init__(self, input_size=24, hidden_size=64, output_size=2, drop_out=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        # self.fc3 = nn.Linear(hidden_size, output_size, bias=True)
        self.drop_out = nn.Dropout(drop_out)

    def forward(self, x):
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
        self.output_size = output_size
        self.linear_cont_P = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3 * output_size)
        )
        self.embedding_p = nn.Embedding(5, output_size)
        self.linear_cont_C = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2 * output_size)
        )
        self.embedding_c = nn.Embedding(2, output_size)
        # self.transformer = PETransformer(nhead=16, num_encoder_layers=12)
    
    def forward(self, cont_P, disc_P, cont_C, disc_C):
        cont_P_out = self.linear_cont_P(cont_P)
        cont_P_out = cont_P_out.view(-1, 3, self.output_size)
        disc_P_out = self.embedding_p(disc_P)
        
        p_out = torch.cat((cont_P_out, disc_P_out), dim=1)
        p_out = torch.mean(p_out, dim=1)
        
        # p_out = positional encoding PE + p_out

        cont_C_out = self.linear_cont_C(cont_C)
        cont_C_out = cont_C_out.view(-1, 2, self.output_size)
        disc_C_out = self.embedding_c(disc_C)
        c_out = torch.cat((cont_C_out, disc_C_out), dim=1)
        c_out = torch.mean(c_out, dim=1)
        
        out = torch.add(p_out, c_out) # [batch x 16 self.output_size]

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