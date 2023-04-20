import torch
import torch.nn as nn
import numpy as np
import pandas as pd

import os, time

import argparse
import tabulate

import utils, models
import wandb
from torch.utils.data import DataLoader

# import warnings
# warnings.filterwarnings('ignore')

## Argparse ----------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Cluster Medical-AI")

parser.add_argument("--seed", type=int, default=1000, help="random seed (default: 1000)")

parser.add_argument("--resume", type=str, default=None,
    help="path to load saved model to resume training (default: None)",)

parser.add_argument("--ignore_wandb", action='store_true',
        help = "Stop using wandb (Default : False)")

parser.add_argument("--save_pred", action='store_true',
        help = "Save ground truth and prediction as csv (Default : False)")

# Data ---------------------------------------------------------
parser.add_argument(
    "--data_path",
    type=str,
    default='./data/data_final_mod.csv',
    help="path to datasets location",)

# parser.add_argument("--tr_ratio", type=float, default=0.7,
#           help="Ratio of train data (Default : 0.2)")

# parser.add_argument("--val_ratio", type=float, default=0.1,
#           help="Ratio of validation data (Default : 0.1)")

# parser.add_argument("--te_ratio", type=float, default=0.2,
#           help="Ratio of test data (Default : 0.2)")

parser.add_argument(
    "--batch_size",
    type=int, default=32,
    help="Batch Size (default : 32)"
)

parser.add_argument(
    "--scaling",
    type=str,
    default='minmax',
    choices=['minmax', 'normalization']
)
#----------------------------------------------------------------


# Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='mlp',
    choices=["transformer", "linear", "ridge", "mlp", "svr"],
    help="model name (default : mlp)")

parser.add_argument("--save_path",
            type=str, default="/mlainas/medical-ai/cluster-regression/exp_result/",
            help="Path to save best model dict")

parser.add_argument(
    "--num_features",
    type=int, default=128,
    help="feature size (default : 128)"
)

parser.add_argument(
    "--hidden_dim",
    type=int, default=64,
    help="MLP model hidden size (default : 64)"
)

parser.add_argument(
    "--output_size",
    type=int, default=2,
    help="Output size (default : 2)"
)

parser.add_argument(
    "--drop_out",
    type=float, default=0.0,
    help="Dropout Rate (Default : 0)"
)

parser.add_argument(
    "--mask_ratio",
    type=float, default=0.5,
    help="Cluster Mask Ratio (Default : 0.5)"
)

#----------------------------------------------------------------

# Criterion -----------------------------------------------------
parser.add_argument(
    "--criterion",
    type=str, default='MSE', choices=["MSE", "RMSE"],
    help="Criterion for training (default : MSE)")

parser.add_argument(
    "--eval_criterion",
    type=str, default='MAE', choices=["MAE"],
    help="Criterion for training (default : MAE)")
#----------------------------------------------------------------

# Learning Hyperparameter --------------------------------------
parser.add_argument("--lr_init", type=float, default=0.005,
                help="learning rate (Default : 0.005)")

parser.add_argument("--optim", type=str, default="adam",
                    choices=["sgd", "adam"],
                    help="Optimization options")

parser.add_argument("--momentum", type=float, default=0.9,
                help="momentum (Default : 0.9)")

parser.add_argument("--nesterov", action='store_true',  help="Nesterov (Default : False)")

parser.add_argument("--epochs", type=int, default=300, metavar="N",
    help="number epochs to train (Default : 300)")

parser.add_argument("--wd", type=float, default=5e-4, help="weight decay (Default: 5e-4)")

parser.add_argument("--scheduler", type=str, default='constant', choices=['constant', "cos_anneal"])

parser.add_argument("--t_max", type=int, default=300,
                help="T_max for Cosine Annealing Learning Rate Scheduler (Default : 300)")
#----------------------------------------------------------------

parser.add_argument("--lamb", type=float, default=0.0,
                help="Penalty term for Ridge Regression (Default : 0)")


args = parser.parse_args()
## ----------------------------------------------------------------------------------------------------



## Set seed and device ----------------------------------------------------------------
utils.set_seed(args.seed)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")
#-------------------------------------------------------------------------------------

## Set wandb ---------------------------------------------------------------------------
if args.ignore_wandb == False:
    wandb.init(entity="mlai_medical_ai", project="cluster-regression")
    wandb.config.update(args)
    wandb.run.name = f"{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}"

## Load Data --------------------------------------------------------------------------------
data = pd.read_csv(args.data_path)
dataset = utils.Tabledata(data, args.scaling, args.mask_ratio)
# dataset = utils.Seqdata(data)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
print("Successfully load data!")
#-------------------------------------------------------------------------------------


## Model ------------------------------------------------------------------------------------
if args.model == 'transformer':
    model = models.TSTransformer(hidden_size=args.hidden_dim,
                    output_size=args.output_size).to(args.device)
   
elif args.model == "mlp":
    model = models.MLPRegressor(input_size=args.num_features,
                    hidden_size=args.hidden_dim,
                    output_size=args.output_size,
                    drop_out=args.drop_out).to(args.device)

elif args.model in ["linear", "ridge"]:
    model = models.LinearRegression(input_size=args.num_features,
                    out_channels=args.output_size).to(args.device)

elif args.model == "svr":
    raise RuntimeError("SVR 구현하면 이거 지우기")



print(f"Successfully prepare {args.model} model")
# ---------------------------------------------------------------------------------------------


## Criterion ------------------------------------------------------------------------------
# Train Criterion
if args.criterion == 'MSE':
    criterion = nn.MSELoss() 

elif args.criterion == "RMSE":
    criterion = utils.RMSELoss()


# Validation / Test Criterion
if args.eval_criterion == 'MAE':
    eval_criterion = nn.L1Loss()

elif args.eval_criterion == "RMSE":
    eval_criterion = utils.RMSELoss()
# ---------------------------------------------------------------------------------------------

## Optimizer and Scheduler --------------------------------------------------------------------
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.wd)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd, nesterov=args.nesterov)
else:
    raise NotImplementedError

if args.scheduler  == "cos_anneal":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)    
else:
    scheduler = None
# ---------------------------------------------------------------------------------------------


## Training Phase -----------------------------------------------------------------------------
columns = ["ep", "lr",
           f"tr_loss({args.criterion})",
           f"val_loss({args.eval_criterion})",
           f"te_loss({args.eval_criterion})",
           "time"]

best_val_loss = 9999

for epoch in range(1, args.epochs + 1):
    time_ep = time.time()
    lr = optimizer.param_groups[0]['lr']

    tr_epoch_loss = 0; val_epoch_loss = 0; te_epoch_loss = 0

    total_tr_num_data = 0; total_val_num_data = 0; total_te_num_data = 0

    tr_gt_y_list = []; val_gt_y_list = []; te_gt_y_list = []
    tr_pred_y_list = []; val_pred_y_list = []; te_pred_y_list = []
    
    tr_gt_d_list = []; val_gt_d_list = []; te_gt_d_list = []
    tr_pred_d_list = []; val_pred_d_list = []; te_pred_d_list = []

    for itr, data in enumerate(dataloader):
        tr_batch_loss, tr_num_data, tr_predicted, tr_ground_truth = utils.train(data, model, optimizer, criterion, args.lamb)
        val_batch_loss, val_num_data, val_predicted, val_ground_truth = utils.valid(data, model, eval_criterion,
                                                                            args.scaling, dataset.a_y, dataset.b_y,
                                                                            dataset.a_d, dataset.b_d)
        te_batch_loss, te_num_data, te_predicted, te_ground_truth = utils.test(data, model, eval_criterion,
                                                                            args.scaling, dataset.a_y, dataset.b_y,
                                                                            dataset.a_d, dataset.b_d)
        tr_epoch_loss += tr_batch_loss
        val_epoch_loss += val_batch_loss
        te_epoch_loss += te_batch_loss

        total_tr_num_data += tr_num_data
        total_val_num_data += val_num_data
        total_te_num_data += te_num_data

        # Restore Prediction and Ground Truth
        if args.scaling == "minmax":
            tr_pred_y = utils.restore_minmax(tr_predicted[:, 0], dataset.a_y, dataset.b_y)
            tr_gt_y = utils.restore_minmax(tr_ground_truth[:, 0], dataset.a_y, dataset.b_y)
            tr_pred_d = utils.restore_minmax(tr_predicted[:, 1], dataset.a_d, dataset.b_d)
            tr_gt_d = utils.restore_minmax(tr_ground_truth[:, 1], dataset.a_d, dataset.b_d)
                      
        elif args.scaling == "normalization":
            tr_pred_y = utils.restore_meanvar(tr_predicted[:, 0], dataset.a_y, dataset.b_y)
            tr_gt_y = utils.restore_meanvar(tr_ground_truth[:, 0], dataset.a_y, dataset.b_y)
            tr_pred_d = utils.restore_meanvar(tr_predicted[:, 1], dataset.a_d, dataset.b_d)
            tr_gt_d = utils.restore_meanvar(tr_ground_truth[:, 1], dataset.a_d, dataset.b_d)

        tr_pred_y_list += list(tr_pred_y.cpu().detach().numpy())
        tr_gt_y_list += list(tr_gt_y.cpu().detach().numpy())
        tr_pred_d_list += list(tr_pred_d.cpu().detach().numpy())
        tr_gt_d_list += list(tr_gt_d.cpu().detach().numpy())
        
        val_pred_y_list += list(val_predicted[:,0].cpu().detach().numpy())
        val_gt_y_list += list(val_ground_truth[:,0].cpu().detach().numpy())
        val_pred_d_list += list(val_predicted[:,1].cpu().detach().numpy())
        val_gt_d_list += list(val_ground_truth[:,1].cpu().detach().numpy())

        te_pred_y_list += list(te_predicted[:,0].cpu().detach().numpy())
        te_gt_y_list += list(te_ground_truth[:,0].cpu().detach().numpy())
        te_pred_d_list += list(te_predicted[:,1].cpu().detach().numpy())
        te_gt_d_list += list(te_ground_truth[:,1].cpu().detach().numpy())
        
    # Calculate Epoch Loss
    tr_loss = tr_epoch_loss / total_tr_num_data
    val_loss = val_epoch_loss / total_val_num_data
    te_loss = te_epoch_loss / total_te_num_data
    
    time_ep = time.time() - time_ep

    values = [epoch, lr, tr_loss, val_loss, te_loss, time_ep,]

    table = tabulate.tabulate([values], headers=columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 20 == 0 or epoch == 1:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)

    if args.scheduler == 'cos_anneal':
        scheduler.step()

    # Save Best Model (Early Stopping)
    if val_loss < best_val_loss:
        best_epoch = epoch
        best_val_loss = val_loss
        best_te_loss = te_loss
        
        # save state_dict
        os.makedirs(args.save_path, exist_ok=True)
        utils.save_checkpoint(file_path = f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}_best_val.pt",
                            epoch = epoch,
                            state_dict = model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            )
        if args.save_pred:
            # save prediction and ground truth as csv
            val_df = pd.DataFrame({'val_pred_y':val_pred_y_list,
                            'val_ground_truth_y':val_gt_y_list,
                            'val_pred_d' : val_pred_d_list,
                            'val_ground_truth_d' : val_gt_d_list})
            val_df.to_csv(f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}_best_val_pred.csv")

            te_df = pd.DataFrame({'te_pred_y':te_pred_y_list,
                            'te_ground_truth_y':te_gt_y_list,
                            'te_pred_d' : te_pred_d_list,
                            'te_ground_truth_d' : te_gt_d_list})                
            te_df.to_csv(f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}_best_te_pred.csv")
        
    
    if not args.ignore_wandb:
        wandb.log({"lr" : lr,
                "tr_loss" : tr_loss,
                "val_loss": val_loss,
                "te_loss" : te_loss,
                })
# ---------------------------------------------------------------------------------------------



## Print Best Model ---------------------------------------------------------------------------
print(f"Best {args.model} achieved {best_te_loss} on {best_epoch} epoch!!")
print(f"The model saved as '{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}_best_val.pt'!!")

if args.ignore_wandb == False:
    wandb.run.summary["best_epoch"]  = best_epoch
    wandb.run.summary["best_val_loss"] = best_val_loss
    wandb.run.summary["best_te_loss"] = best_te_loss
    wandb.run.summary["tr_dat_num"] = total_tr_num_data
    wandb.run.summary["val_dat_num"] : total_val_num_data
    wandb.run.summary["te_dat_num"] : total_te_num_data
# ---------------------------------------------------------------------------------------------