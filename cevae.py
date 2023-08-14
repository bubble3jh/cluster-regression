# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
This example demonstrates how to use the Causal Effect Variational Autoencoder
[1] implemented in pyro.contrib.cevae.CEVAE, documented at
http://docs.pyro.ai/en/latest/contrib.cevae.html

**References**

[1] C. Louizos, U. Shalit, J. Mooij, D. Sontag, R. Zemel, M. Welling (2017).
    Causal Effect Inference with Deep Latent-Variable Models.
    http://papers.nips.cc/paper/7223-causal-effect-inference-with-deep-latent-variable-models.pdf
    https://github.com/AMLab-Amsterdam/CEVAE
"""
import argparse
import logging

import torch

import pyro
import pyro.distributions as dist

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from IPython.display import display

import os, time
import math

import argparse
import tabulate

import utils, models, ml_algorithm
import wandb
from torch.utils.data import DataLoader, random_split, ConcatDataset, TensorDataset
from tqdm import tqdm

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)


def main(args):
    if args.use_default:
        from pyro.contrib.cevae import CEVAE, PreWhitener
        t_type="binary"
        args.device = "cpu"
    else:
        from causal.cevae__init__ import CEVAE, PreWhitener
        t_type="multi"

    utils.set_seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available(): 
        device = 'cuda'
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        device = 'cpu'
    torch.device(device)
    print(f'device : {device}')

    ## Load Data --------------------------------------------------------------------------------
    args.data_path='./data/'
    args.scaling='minmax'
    data = pd.read_csv(args.data_path+f"data_cut_{0}.csv")
    dataset = utils.CEVAEdataset(data, args.scaling, t_type)
    x, y, t = dataset.get_data()
    x = x.to(device); y = y.to(device).float(); t = t.to(device).float()
    dataset_size = x.size(0)

    indices = torch.randperm(dataset_size)
    train_ratio = 0.8
    train_size = int(train_ratio * dataset_size)

    x_train = x[indices[:train_size]]
    y_train = y[indices[:train_size]][:,0]
    d_train = y[indices[:train_size]][:,1]
    t_train = t[indices[:train_size]]

    x_test = x[indices[train_size:]]
    y_test = y[indices[train_size:]][:,0]
    d_test = y[indices[train_size:]][:,1]
    t_test = t[indices[train_size:]]
    print("Successfully load data!")

    #-------------------------------------------------------------------------------------
    # Train.

    pyro.clear_param_store()
    cevae = CEVAE(
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_samples=10,
        outcome_dist='normal'
    )

    kwargs = {}
    if not args.use_default:
        kwargs['d'] = d_train

    cevae.fit(
        x = x_train,
        t = t_train,
        y = y_train,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        weight_decay=args.weight_decay,
        **kwargs
    )
    print("Successfully trained model!")

    #-------------------------------------------------------------------------------------
    # Evaluate.

    y_diffs = []
    d_diffs = []
    testdataset = TensorDataset(x_test, t_test, y_test, d_test)
    dl = DataLoader(testdataset, batch_size=16)
    whiten = PreWhitener(x_test)
    with tqdm(initial = 0, total = len(dl)) as pbar:
        pbar.set_description('calculating loss')
        for x, t, y, d in dl:
            x = whiten(x)
            y_hat = cevae.model.y_mean(x, t)
            d_hat = cevae.model.d_mean(x, t) 
            
            y_diffs.append(y - y_hat)
            d_diffs.append(d - d_hat)
            pbar.update(1)
    
    y_diffs_tensor = torch.cat(y_diffs)
    d_diffs_tensor = torch.cat(d_diffs)

    y_mae = torch.mean(torch.abs(y_diffs_tensor))
    y_rmse = torch.sqrt(torch.mean(y_diffs_tensor ** 2))

    d_mae = torch.mean(torch.abs(d_diffs_tensor))
    d_rmse = torch.sqrt(torch.mean(d_diffs_tensor ** 2))
    df = pd.DataFrame({
    'Metrics': ['MAE', 'RMSE'],
    'Y-values': [0, 0],
    'D-values': [0, 0]
    })
    df['Y-values'] = [y_mae.item(), y_rmse.item()]
    df['D-values'] = [d_mae.item(), d_rmse.item()]
    display(df)

    naive_ate_y = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
    naive_ate_d = d_test[t_test == 1].mean() - d_test[t_test == 0].mean()
    print("naive ATE y = {:0.3g}".format(naive_ate_y))
    print("naive ATE d = {:0.3g}".format(naive_ate_d))

    if args.jit:
        cevae = cevae.to_script_module()
    est_ite = cevae.ite(x_test) #TODO : y,d 수정 필요
    est_ate = est_ite.mean()
    print("estimated ATE = {:0.3g}".format(est_ate.item()))
    ## TODO : 그래서 잘 학습하고 난 다음, x,t 넣어주고 y,d값은 어떻게 뽑지? 뽑아야 loss를 구하는디

if __name__ == "__main__":
    assert pyro.__version__.startswith("1.8.6")
    parser = argparse.ArgumentParser(
        description="Causal Effect Variational Autoencoder"
    )
    parser.add_argument("--num-data", default=13061, type=int)
    parser.add_argument("--feature-dim", default=11, type=int)
    parser.add_argument("--latent-dim", default=20, type=int)
    parser.add_argument("--hidden-dim", default=200, type=int)
    parser.add_argument("--num-layers", default=3, type=int)
    parser.add_argument("-n", "--num-epochs", default=50, type=int)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--use_default", action='store_true',
        help = "Use default cevae file (Default : False)")
    args = parser.parse_args()
    main(args)