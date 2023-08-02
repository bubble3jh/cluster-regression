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
from pyro.contrib.cevae import CEVAE
# from causal.cevae__init__ import CEVAE

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import os, time
import math

import argparse
import tabulate

import utils, models, ml_algorithm
import wandb
from torch.utils.data import DataLoader, random_split, ConcatDataset

logging.getLogger("pyro").setLevel(logging.DEBUG)
logging.getLogger("pyro").handlers[0].setLevel(logging.DEBUG)


def generate_data(args):
    """
    This implements the generative process of [1], but using larger feature and
    latent spaces ([1] assumes ``feature_dim=1`` and ``latent_dim=5``).
    """
    z = dist.Bernoulli(0.5).sample([args.num_data])
    x = dist.Normal(z, 5 * z + 3 * (1 - z)).sample([args.feature_dim]).t()
    t = dist.Bernoulli(0.75 * z + 0.25 * (1 - z)).sample()
    y = dist.Bernoulli(logits=3 * (z + 2 * (2 * t - 2))).sample()

    # Compute true ite for evaluation (via Monte Carlo approximation).
    t0_t1 = torch.tensor([[0.0], [1.0]])
    y_t0, y_t1 = dist.Bernoulli(logits=3 * (z + 2 * (2 * t0_t1 - 2))).mean
    true_ite = y_t1 - y_t0
    return x, t, y, true_ite


def main(args):
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    ## Load Data --------------------------------------------------------------------------------
    ### ./data/data_mod.ipynb 에서 기본적인 데이터 전처리  ###
    args.data_path='./data/'
    args.scaling='minmax'
    data = pd.read_csv(args.data_path+f"data_cut_{0}.csv")
    dataset = utils.CEVAEdataset(data, args.scaling)
    x, y, t = dataset.get_data()
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
    # Generate synthetic data.
    
    # x_train, t_train, y_train, _ = generate_data(args)
    # Train.
    pyro.clear_param_store()
    cevae = CEVAE(
        feature_dim=args.feature_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_samples=10,
        outcome_dist='normal'
    ).to(torch.float64)
    
    cevae.fit(
        x_train,
        t_train,
        y_train,
        d_train,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        learning_rate_decay=args.learning_rate_decay,
        weight_decay=args.weight_decay,
    )

    # Evaluate.
    # x_test, t_test, y_test, true_ite = generate_data(args)
    # true_ate = true_ite.mean()
    # print("true ATE = {:0.3g}".format(true_ate.item()))
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
    parser.add_argument("-b", "--batch-size", default=100, type=int)
    parser.add_argument("-lr", "--learning-rate", default=1e-3, type=float)
    parser.add_argument("-lrd", "--learning-rate-decay", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=1234567890, type=int)
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    main(args)