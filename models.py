import torch
import torch.nn as nn
# import torch_geometric
import torch.nn.functional as F

from typing import Optional
# from torch_geometric.nn import GATv2Conv, GCNConv
# from torch_geometric_signed_directed.nn.directed import MagNetConv, complex_relu

class GAT_Net(nn.Module):
    def __init__(self, in_channels, hidden_dim = 8, heads = 2, out_channels=1, drop_out=0.6):
        super().__init__()

        # self.conv1 = GATv2Conv(in_channels, 8, heads=8, dropout=0.6)
        self.conv1 = GATv2Conv(in_channels, hidden_dim, heads)
        # On the Pubmed dataset, use heads=8 in conv2.
        # self.conv2 = GATv2Conv(8 * 8, out_channels, heads=1, concat=False, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_dim * heads, hidden_dim * heads, heads=1, concat=False)
        self.linear = nn.Linear(hidden_dim * heads, 1)
        self.drop_out = drop_out

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(x)
        # return F.log_softmax(x, dim=-1)
        return x



class GCN_Net(nn.Module):
    def __init__(self, in_channels, hidden_dim = 16, out_channels=1, drop_out=0.6):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_dim) # GCNConv 
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, 1)

        self.drop_out = drop_out

    def forward(self, x, edge_index):
        # pdb.set_trace()
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.drop_out, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.linear(x)
        # return F.log_softmax(x, dim=-1)
        return x
    

class MagNet(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 4, q: float = 0.25, K: int = 2, 
                 activation: bool = False, trainable_q: bool = False, layer: int = 2, 
                 drop_out: float = False, normalization: str = 'sym', cached: bool = False):
        super(MagNet, self).__init__()
        chebs = nn.ModuleList()
        chebs.append(MagNetConv(in_channels=in_channels, out_channels=hidden, K=K,
                                q=q, trainable_q=trainable_q, normalization=normalization, cached=cached))
        self.normalization = normalization
        self.activation = activation
        if self.activation:
            self.complex_relu = complex_relu.complex_relu_layer()

        for _ in range(1, layer):
            chebs.append(MagNetConv(in_channels=hidden, out_channels=hidden, K=K,
                                    q=q, trainable_q=trainable_q, normalization=normalization, cached=cached))

        self.Chebs = chebs

        self.Conv = nn.Conv1d(2*hidden, 2*hidden, kernel_size=1)
        self.linear = nn.Linear(2 * hidden, 1)
        self.dropout = drop_out

    def forward(self, x, edge_index: torch.LongTensor,
                edge_weight: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        real = x
        imag = x
        for cheb in self.Chebs:
            real, imag = cheb(real, imag, edge_index, edge_weight)
            if self.activation:
                real, imag = self.complex_relu(real, imag)

        x = torch.cat((real, imag), dim=-1)

        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        x = x.unsqueeze(0)
        x = x.permute((0, 2, 1))
        x = self.Conv(x)
        # x = F.log_softmax(x, dim=1)
        return x.squeeze()[0]
    

class MLPRegressor(nn.Module):
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

# class MLP(nn.Module):
#     """Just  an MLP"""
#     def __init__(self, n_inputs, n_outputs, hparams):
#         super(MLP, self).__init__()
#         self.input = nn.Linear(n_inputs, hparams['mlp_width'])
#         self.dropout = nn.Dropout(hparams['mlp_dropout'])
#         self.hiddens = nn.ModuleList([
#             nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
#             for _ in range(hparams['mlp_depth']-2)])
#         self.output = nn.Linear(hparams['mlp_width'], n_outputs)
#         self.n_outputs = n_outputs

#     def forward(self, x):
#         x = self.input(x)
#         x = self.dropout(x)
#         x = F.relu(x)
#         for hidden in self.hiddens:
#             x = hidden(x)
#             x = self.dropout(x)
#             x = F.relu(x)
#         x = self.output(x)
#         return x

class LinearRegression(torch.nn.Module):
    def __init__(self, input_size=17, out_channels=1):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, out_channels)

    def forward(self, x):
        x = self.linear1(x)
        return x



class SVM(torch.nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass