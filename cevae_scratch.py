# Copyright (c) 2017-2019 Uber Technologies, Inc.
# SPDX-License-Identifier: Apache-2.0

import argparse
import logging

import torch


import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from IPython.display import display

import math

import argparse
from prettytable import PrettyTable
from models import CEVAE_det, Transformer
import utils, models, ml_algorithm
import wandb
from torch.utils.data import DataLoader, random_split, ConcatDataset, TensorDataset
from tqdm import tqdm


def main(args):
    args.tukey = False # TODO : Hard Coding
    utils.set_seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available(): 
        device = 'cuda'
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")
        # generator = torch.Generator(device=device)
    else:
        device = 'cpu'
        # generator = torch.Generator(device=device)
    torch.device(device)
    if not args.ignore_wandb:
            wandb.init(entity="mlai_medical_ai" ,project="causal-effect-vae", config=args, group="scratch") #group=args.sweep_group)
            wandb.run.name=f"cevae scratch_first_trial"
    ## Criterion ------------------------------------------------------------------------------
    # Train Criterion
    if args.criterion in ['MSE', 'RMSE']:
        criterion = nn.MSELoss(reduction="sum") 

    # Validation Criterion
    if args.eval_criterion == 'MAE':
        eval_criterion = nn.L1Loss(reduction="sum")

    elif args.eval_criterion == "RMSE":
        eval_criterion = nn.MSELoss(reduction="sum")
        
    ## Load Data --------------------------------------------------------------------------------
    args.data_path='./data/'
    args.scaling='minmax'
    data = pd.read_csv(args.data_path+f"data_cut_{0}.csv")
    t_classes=2 if args.binary_t else 7
    # dataset = utils.CEVAEdataset(data, args.scaling, t_type)
    dataset = utils.Tabledata(args, data, scale="minmax", use_treatment=True, binary_t=args.binary_t)
    train_dataset, val_dataset, test_dataset = random_split(dataset, utils.data_split_num(dataset))
    tr_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print("Successfully load data!")

    # Load Model --------------------------------------------------------------------------------
    model = CEVAE_det(embedding_dim=args.embedding_dim, latent_dim=args.latent_dim, encoder_hidden_dim=args.hidden_dim, 
                    encoder_shared_layers=args.shared_layers, encoder_pred_layers=args.pred_layers, transformer_layers=args.num_layers, 
                    drop_out=args.drop_out, t_classes=t_classes, t_pred_layers=args.t_pred_layers, skip_hidden=args.skip_hidden,
                    t_embed_dim=args.t_embed_dim, yd_embed_dim=args.yd_embed_dim).to(device)
    print(model)
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss(); aux_criterion = torch.nn.CrossEntropyLoss()
    print("Successfully load model!")
    #-------------------------------------------------------------------------------------
    # Train.
    best_epochs=[0] 
    best_val_loss_d = 9999 ; best_val_loss_y = 9999 
    best_test_losses = [9999 for j in range(4)]
    log_dict = {}

    pbar = tqdm(range(args.num_epochs), position=0, leave=True)
    for epoch in pbar:
        tr_epoch_loss_d = 0; tr_epoch_loss_y = 0; tr_epoch_eval_loss_d = 0; tr_epoch_eval_loss_y = 0; val_epoch_loss_d = 0; val_epoch_loss_y = 0; te_mae_epoch_loss_d = 0; te_mae_epoch_loss_y = 0; te_mse_epoch_loss_d = 0; te_mse_epoch_loss_y = 0

        concat_tr_num_data = 0; concat_val_num_data = 0; concat_te_num_data = 0

        tr_gt_y_list = []; val_gt_y_list = []; te_gt_y_list = []
        tr_pred_y_list = []; val_pred_y_list = []; te_pred_y_list = []
        
        tr_gt_d_list = []; val_gt_d_list = []; te_gt_d_list = []
        tr_pred_d_list = []; val_pred_d_list = []; te_pred_d_list = []
        for itr, data in enumerate(tr_dataloader):
            tr_batch_loss_d, tr_batch_loss_y, tr_num_data, tr_predicted, tr_ground_truth, *tr_eval_losses = utils.train(data, model, optimizer, criterion, epoch, warmup_iter=args.warmup_iter, aux_criterion=aux_criterion, eval_criterion=eval_criterion, use_treatment=True,
                                                                                                                        a_y=train_dataset.dataset.a_y, a_d=train_dataset.dataset.a_d, b_y=train_dataset.dataset.b_y, b_d=train_dataset.dataset.b_d, pred_model=args.pred_model, binary_t=args.binary_t)
            tr_epoch_loss_d += tr_batch_loss_d
            tr_epoch_loss_y += tr_batch_loss_y
            concat_tr_num_data += tr_num_data

            tr_pred_y_list += list(tr_predicted[:,0].cpu().detach().numpy())
            tr_gt_y_list += list(tr_ground_truth[:,0].cpu().detach().numpy())
            tr_pred_d_list += list(tr_predicted[:,1].cpu().detach().numpy())
            tr_gt_d_list += list(tr_ground_truth[:,1].cpu().detach().numpy())
        # Calculate Epoch loss
        tr_loss_d = tr_epoch_loss_d / concat_tr_num_data
        tr_loss_y = tr_epoch_loss_y / concat_tr_num_data
        tr_eval_loss_d = tr_eval_losses[1] / concat_tr_num_data
        tr_eval_loss_y = tr_eval_losses[0] / concat_tr_num_data
        log_dict.update({
            "tr_loss_d": tr_loss_d,
            "tr_loss_y": tr_loss_y,
            "tr_eval_loss_d": tr_eval_loss_d,
            "tr_eval_loss_y": tr_eval_loss_y
        })
        if args.criterion == "RMSE":
            tr_loss_d = math.sqrt(tr_loss_d)
            tr_loss_y = math.sqrt(tr_loss_y)
        # ---------------------------------------------------------------------------------------
    
        val_output=[]; test_output=[]
        val_loss_d_list = []; val_loss_y_list = []
        test_mae_d_list = []; test_mae_y_list = [] ;test_rmse_d_list = []; test_rmse_y_list = []
        ## Validation Phase ----------------------------------------------------------------------
        for data in val_dataloader:
            val_batch_loss_d, val_batch_loss_y, val_num_data, val_predicted, val_ground_truth = utils.valid(data, model, eval_criterion,
                                                                                args.scaling, val_dataset.dataset.a_y, val_dataset.dataset.b_y,
                                                                                val_dataset.dataset.a_d, val_dataset.dataset.b_d, use_treatment=True)
            val_epoch_loss_d += val_batch_loss_d
            val_epoch_loss_y += val_batch_loss_y
            concat_val_num_data += val_num_data

            val_pred_y_list += list(val_predicted[:,0].cpu().detach().numpy())
            val_gt_y_list += list(val_ground_truth[:,0].cpu().detach().numpy())
            val_pred_d_list += list(val_predicted[:,1].cpu().detach().numpy())
            val_gt_d_list += list(val_ground_truth[:,1].cpu().detach().numpy())

        # Calculate Epoch loss
        val_loss_d = val_epoch_loss_d / concat_val_num_data
        val_loss_y = val_epoch_loss_y / concat_val_num_data
        if args.eval_criterion == "RMSE":
            val_loss_d = math.sqrt(val_loss_d)
            val_loss_y = math.sqrt(val_loss_y)
        log_dict.update({
            "val_loss_d": val_loss_d,
            "val_loss_y": val_loss_y
        })
        # save list for all cut-off dates
        val_loss_d_list.append(val_loss_d)
        val_loss_y_list.append(val_loss_y)
        # ---------------------------------------------------------------------------------------

        ## Test Phase ----------------------------------------------------------------------
        for data in test_dataloader:
            te_mae_batch_loss_d, te_mae_batch_loss_y, te_mse_batch_loss_d, te_mse_batch_loss_y, te_num_data, te_predicted, te_ground_truth = utils.test(data, model,
                                                                                args.scaling, test_dataset.dataset.a_y, test_dataset.dataset.b_y,
                                                                                test_dataset.dataset.a_d, test_dataset.dataset.b_d, use_treatment=True)
            te_mae_epoch_loss_d += te_mae_batch_loss_d
            te_mae_epoch_loss_y += te_mae_batch_loss_y
            te_mse_epoch_loss_d += te_mse_batch_loss_d
            te_mse_epoch_loss_y += te_mse_batch_loss_y
            concat_te_num_data += te_num_data

            # Restore Prediction and Ground Truth
            te_pred_y, te_pred_d, te_gt_y, te_gt_d= utils.reverse_scaling(args.scaling, te_predicted, te_ground_truth, test_dataset.dataset.a_y, test_dataset.dataset.b_y, test_dataset.dataset.a_d, test_dataset.dataset.b_d)

            te_pred_y_list += list(te_pred_y.cpu().detach().numpy())
            te_gt_y_list += list(te_gt_y.cpu().detach().numpy())
            te_pred_d_list += list(te_pred_d.cpu().detach().numpy())
            te_gt_d_list += list(te_gt_d.cpu().detach().numpy())

        # Calculate Epoch loss
        te_mae_loss_d = te_mae_epoch_loss_d / concat_te_num_data
        te_mae_loss_y = te_mae_epoch_loss_y / concat_te_num_data
        te_rmse_loss_d = math.sqrt(te_mse_epoch_loss_d / concat_te_num_data)
        te_rmse_loss_y = math.sqrt(te_mse_epoch_loss_y / concat_te_num_data)

        # save list for all cut-off dates
        test_mae_d_list.append(te_mae_loss_d);test_mae_y_list.append(te_mae_loss_y)
        test_rmse_d_list.append(te_rmse_loss_d); test_rmse_y_list.append(te_rmse_loss_y)
        log_dict.update({
            "te_mae_loss_d": te_mae_loss_d,
            "te_mae_loss_y": te_mae_loss_y,
            "te_rmse_loss_d": te_rmse_loss_d,
            "te_rmse_loss_y": te_rmse_loss_y
        })
        # ---------------------------------------------------------------------------------------
        if not args.ignore_wandb:
            wandb.log(log_dict)
        # Save Best Model (Early Stopping)
        if val_loss_d + val_loss_y < best_val_loss_d + best_val_loss_y:
            best_epochs = epoch
            best_val_loss_d = val_loss_d
            best_val_loss_y = val_loss_y

            best_test_losses[0] = te_mae_loss_d
            best_test_losses[1] = te_mae_loss_y
            best_test_losses[2] = te_rmse_loss_d
            best_test_losses[3] = te_rmse_loss_y
            if not args.ignore_wandb:
                wandb.run.summary["best_epoch"] = best_epochs
                wandb.run.summary["best_val_loss_d"] = best_val_loss_d
                wandb.run.summary["best_val_loss_y"] = best_val_loss_y
                wandb.run.summary["best_test_mae_loss_d"] = te_mae_loss_d
                wandb.run.summary["best_test_mae_loss_y"] = te_mae_loss_y
                wandb.run.summary["best_test_rmse_loss_d"] = te_rmse_loss_d
                wandb.run.summary["best_test_rmse_loss_y"] = te_rmse_loss_y
        desc = f"Epoch: {epoch+1}, tr_loss_y: {tr_eval_loss_y:.4f}, tr_loss_d: {tr_eval_loss_d:.4f}, val_loss_y: {val_loss_y:.4f}, val_loss_d: {val_loss_d:.4f}, te_loss_y: {te_mae_loss_y:.4f}, te_loss_d: {te_mae_loss_d:.4f}"
        pbar.set_description(desc)
        pbar.update(1)
    print(f"Successfully trained model! | best d mae : {best_test_losses[0]:.4f} | best y mae : {best_test_losses[1]:.4f}")

    #-------------------------------------------------------------------------------------
    # Evaluate counter factual [only on train set]
    counterfactual_differences = utils.estimate_counterfactuals(model, tr_dataloader, a_y=train_dataset.dataset.a_y, a_d=train_dataset.dataset.a_d, b_y=train_dataset.dataset.b_y, b_d=train_dataset.dataset.b_d, use_treatment=True, binary_t=args.binary_t)
    organized_counterfactuals = utils.organize_counterfactuals(counterfactual_differences)
    avg_cf = utils.compute_average_differences(organized_counterfactuals)
    utils.print_average_differences(avg_cf)

def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameters Configuration")
    parser.add_argument("--embedding_dim", default=32, type=int)
    parser.add_argument("--latent_dim", default=16, type=int, help='z dimension')
    parser.add_argument("--hidden_dim", default=32, type=int, help='y,d,t layers dimension')
    parser.add_argument("--t_embed_dim", default=8, type=int, help='t emb dimension')
    parser.add_argument("--yd_embed_dim", default=8, type=int, help='y,d emb dimension')
    parser.add_argument("--num_layers", default=2, type=int, help='transformer layers')
    parser.add_argument("--pred_layers", default=1, type=int, help='y,d predictor head layers')
    parser.add_argument("--t_pred_layers", default=2, type=int, help='t predictor layers')
    parser.add_argument("--shared_layers", default=2, type=int, help='y,d predictor featurizer layers')
    parser.add_argument("-n", "--num_epochs", default=30, type=int)
    parser.add_argument("--warmup_iter", default=0, type=int)
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("-lr", "--learning_rate", default=1e-3, type=float)
    parser.add_argument("-lrd", "--learning_rate_decay", default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--drop_out", type=float, default=0.0)
    parser.add_argument("--pred_model", default="encoder", type=str, choices=["encoder", "decoder"])
    parser.add_argument("--binary_t", action='store_true',
        help = "Use t as binary class (Default : False)")
    parser.add_argument("--skip_hidden", action='store_true',
        help = "Skip hidden (Default : False)")
    parser.add_argument('--make_model_complicate', action='store_true')
    # parser.add_argument("--lambda1", default=1, type=float, help='additional q loss for t')
    # parser.add_argument("--lambda2", default=1, type=float, help='additional q loss for y')
    # parser.add_argument("--lambda3", default=1, type=float, help='additional q loss for d')
    # parser.add_argument("--elbo_lambda1", default=1, type=float, help='ELBO P loss for z')
    # parser.add_argument("--elbo_lambda2", default=1, type=float, help='ELBO P loss for x')
    # parser.add_argument("--elbo_lambda3", default=1, type=float, help='ELBO P loss for t')
    # parser.add_argument("--elbo_lambda4", default=1, type=float, help='ELBO P loss for y')
    # parser.add_argument("--elbo_lambda5", default=1, type=float, help='ELBO P loss for d')
    # parser.add_argument("--elbo_lambda6", default=1, type=float, help='ELBO Q loss for z')
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--sweep_group", default="default", type=str)
    # Criterion -----------------------------------------------------
    parser.add_argument(
        "--criterion",
        type=str, default='MSE', choices=["MSE", "RMSE"],
        help="Criterion for training (default : MSE)")
    parser.add_argument(
        "--eval_criterion",
        type=str, default='MAE', choices=["MAE", "RMSE"],
        help="Criterion for training (default : MAE)")
    parser.add_argument('--tukey', action='store_false', help='Use tukey transformation to get divergence')
    parser.add_argument('--beta', type=float, default=0.5, help='parameter for Tukey transformation (Default : 0.5)')
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")
    args = parser.parse_args()
    if args.make_model_complicate:
        args.embedding_dim = 128
        args.latent_dim = 64
        args.hidden_dim = 128
        args.t_embed_dim = 32
        args.yd_embed_dim = 32
        args.t_pred_layers = 3
        args.shared_layers = 3
    return args

def print_args_to_table(args):
    table = PrettyTable(['Hyperparameter', 'Value'])
    for arg, val in vars(args).items():
        table.add_row([arg, val])
    print(table)

if __name__ == "__main__":
    table = PrettyTable()
    args = parse_args()
    print_args_to_table(args)
    main(args)
    