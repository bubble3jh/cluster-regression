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
parser.add_argument(
    "--table_idx",
    type=int, default=0, choices=[0, 1, 2, 3, 4, 5],
    help="Cluster Date print date (Default : 2) if 0, use concated dataset"
)
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
    choices=["transformer", "linear", "ridge", "mlp", "svr", "rfr"],
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
    "--num_layers",
    type=int, default=3,
    help="MLP model layer num (default : 3)"
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

parser.add_argument("--disable_embedding", action='store_true',
        help = "Disable embedding to raw data (Default : False)")

#----------------------------------------------------------------

# Criterion -----------------------------------------------------
parser.add_argument(
    "--criterion",
    type=str, default='MSE', choices=["MSE", "RMSE"],
    help="Criterion for training (default : MSE)")

parser.add_argument(
    "--eval_criterion",
    type=str, default='MAE', choices=["MAE", "RMSE"],
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

# args.device = torch.device("cpu")
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")
#-------------------------------------------------------------------------------------

## Set wandb ---------------------------------------------------------------------------
if args.ignore_wandb == False:
    wandb.init(entity="mlai_medical_ai", project="cluster-regression")
    wandb.config.update(args)
    if args.disable_embedding:
        wandb.run.name = f"raw_{args.model}({args.hidden_dim})-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}"
    else:
        wandb.run.name = f"embed_{args.model}({args.hidden_dim})-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}"
       
## Load Data --------------------------------------------------------------------------------
data = pd.read_csv(args.data_path)
tr_datasets = []; val_datasets = []; test_datasets = []; 
for i in range(1, 6):
    dataset = utils.Tabledata(data, i, args.scaling)
    train_dataset, val_dataset, test_dataset = random_split(dataset, utils.data_split_num(dataset))
    tr_datasets.append(train_dataset)
    val_datasets.append(val_dataset)
    test_datasets.append(test_dataset)

tr_dataset = ConcatDataset(tr_datasets)
tr_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
print(f"Number of training Clusters : {len(tr_dataset)}")
val_dataloaders=[]; test_dataloaders=[]
# index 0 -> all dataset / index i -> i-th data cut-off
val_dataloaders.append(DataLoader(ConcatDataset(val_datasets), batch_size=args.batch_size, shuffle=False)); test_dataloaders.append(DataLoader(ConcatDataset(test_datasets), batch_size=args.batch_size, shuffle=False))
for i in range(5):
    val_dataset = val_datasets[i]; test_dataset = test_datasets[i]
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    val_dataloaders.append(val_dataloader); test_dataloaders.append(test_dataloader)

print("Successfully load data!")
#-------------------------------------------------------------------------------------


## Model ------------------------------------------------------------------------------------
if args.model == 'transformer':
    model = models.TSTransformer(hidden_size=args.hidden_dim,
                    output_size=args.output_size).to(args.device)
   
elif args.model == "mlp":
    model = models.MLPRegressor(input_size=args.num_features,
                    hidden_size=args.hidden_dim,
                    num_layers=args.num_layers,
                    output_size=args.output_size,
                    drop_out=args.drop_out,
                    disable_embedding=args.disable_embedding).to(args.device)

elif args.model in ["linear", "ridge"]:
    model = models.LinearRegression(input_size=args.num_features,
                    out_channels=args.output_size,
                    disable_embedding=args.disable_embedding).to(args.device)

elif args.model in ["svr", "rfr"]:
    args.device = torch.device("cpu")
    ml_algorithm.fit(data, args.model, args.ignore_wandb, args.table_idx)

print(f"Successfully prepare {args.model} model")
# ---------------------------------------------------------------------------------------------


## Criterion ------------------------------------------------------------------------------
# Train Criterion
if args.criterion in ['MSE', 'RMSE']:
    criterion = nn.MSELoss(reduction="sum") 

# Validation Criterion
if args.eval_criterion == 'MAE':
    eval_criterion = nn.L1Loss(reduction="sum")

elif args.eval_criterion == "RMSE":
    eval_criterion = nn.MSELoss(reduction="sum")
    

# ---------------------------------------------------------------------------------------------

## Optimizer and Scheduler --------------------------------------------------------------------
if args.optim == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_init, weight_decay=args.wd)
elif args.optim == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init, momentum=args.momentum, weight_decay=args.wd)
else:
    raise NotImplementedError

if args.scheduler  == "cos_anneal":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max)    
else:
    scheduler = None
# ---------------------------------------------------------------------------------------------


## Training Phase -----------------------------------------------------------------------------
columns = ["ep", "lr", f"tr_loss_d({args.criterion})", f"tr_loss_y({args.criterion})", f"val_loss_d({args.eval_criterion})", f"val_loss_y({args.eval_criterion})",
           "te_loss_d(MAE)", "te_loss_y(MAE)", "te_loss_d(RMSE)", "te_loss_y(RMSE)", "time"]
## print table index, 0=cocnated data
best_epochs=[0] * 6 # 6 -> len(val_dataloaders)  ####
best_val_loss_d = best_val_loss_y = [9999] * 6
best_test_losses = [[9999] * 4] * 6

for epoch in range(1, args.epochs + 1):
    time_ep = time.time()
    lr = optimizer.param_groups[0]['lr']

    tr_epoch_loss_d = 0; tr_epoch_loss_y = 0; val_epoch_loss_d = 0; val_epoch_loss_y = 0; te_mae_epoch_loss_d = 0; te_mae_epoch_loss_y = 0; te_mse_epoch_loss_d = 0; te_mse_epoch_loss_y = 0

    concat_tr_num_data = 0; concat_val_num_data = 0; concat_te_num_data = 0

    tr_gt_y_list = []; val_gt_y_list = []; te_gt_y_list = []
    tr_pred_y_list = []; val_pred_y_list = []; te_pred_y_list = []
    
    tr_gt_d_list = []; val_gt_d_list = []; te_gt_d_list = []
    tr_pred_d_list = []; val_pred_d_list = []; te_pred_d_list = []

    for itr, data in enumerate(tr_dataloader):
        ## Training phase
        tr_batch_loss_d, tr_batch_loss_y, tr_num_data, tr_predicted, tr_ground_truth = utils.train(data, model, optimizer, criterion, args.lamb)
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
    if args.criterion == "RMSE":
        tr_loss_d = math.sqrt(tr_loss_d)
        tr_loss_y = math.sqrt(tr_loss_y)
    # ---------------------------------------------------------------------------------------


    
    val_output=[]; test_output=[]
    val_loss_d_list = []; val_loss_y_list = []
    test_mae_d_list = []; test_mae_y_list = [] ;test_rmse_d_list = []; test_rmse_y_list = []
    for i in range(6):
        ## Validation Phase ----------------------------------------------------------------------
        for itr, data in enumerate(val_dataloaders[i]):
            val_batch_loss_d, val_batch_loss_y, val_num_data, val_predicted, val_ground_truth = utils.valid(data, model, eval_criterion,
                                                                                args.scaling, dataset.a_y, dataset.b_y,
                                                                                dataset.a_d, dataset.b_d)
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

        # save list for all cut-off dates
        val_loss_d_list.append(val_loss_d)
        val_loss_y_list.append(val_loss_y)
        # ---------------------------------------------------------------------------------------

        ## Test Phase ----------------------------------------------------------------------
        for itr, data in enumerate(test_dataloaders[i]):
            te_mae_batch_loss_d, te_mae_batch_loss_y, te_mse_batch_loss_d, te_mse_batch_loss_y, te_num_data, te_predicted, te_ground_truth = utils.test(data, model,
                                                                                args.scaling, dataset.a_y, dataset.b_y,
                                                                                dataset.a_d, dataset.b_d)
            te_mae_epoch_loss_d += te_mae_batch_loss_d
            te_mae_epoch_loss_y += te_mae_batch_loss_y
            te_mse_epoch_loss_d += te_mse_batch_loss_d
            te_mse_epoch_loss_y += te_mse_batch_loss_y
            concat_te_num_data += te_num_data

            # Restore Prediction and Ground Truth
            te_pred_y, te_pred_d, te_gt_y, te_gt_d= utils.reverse_scaling(args.scaling, te_predicted, te_ground_truth, dataset.a_y, dataset.b_y, dataset.a_d, dataset.b_d)

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

        # ---------------------------------------------------------------------------------------
        
        # Save Best Model (Early Stopping)
        if val_loss_d + val_loss_y < best_val_loss_d[i] + best_val_loss_y[i]:
            best_epochs[i] = epoch

            best_val_loss_d[i] = val_loss_d
            best_val_loss_y[i] = val_loss_y

            best_test_losses[i][0] = te_mae_loss_d
            best_test_losses[i][1] = te_mae_loss_y
            best_test_losses[i][2] = te_rmse_loss_d
            best_test_losses[i][3] = te_rmse_loss_y
            
            # save state_dict
            os.makedirs(args.save_path, exist_ok=True)
            utils.save_checkpoint(file_path = f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}-date{i}_best_val.pt",
                                epoch = epoch,
                                state_dict = model.state_dict(),
                                optimizer = optimizer.state_dict(),
                                )
            # if args.save_pred:
            #     # save prediction and ground truth as csv
            #     val_df = pd.DataFrame({'val_pred_y':val_pred_y_list,
            #                     'val_ground_truth_y':val_gt_y_list,
            #                     'val_pred_d' : val_pred_d_list,
            #                     'val_ground_truth_d' : val_gt_d_list})
            #     val_df.to_csv(f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}_best_val_pred.csv", index=False)

            #     te_df = pd.DataFrame({'te_pred_y':te_pred_y_list,
            #                     'te_ground_truth_y':te_gt_y_list,
            #                     'te_pred_d' : te_pred_d_list,
            #                     'te_ground_truth_d' : te_gt_d_list})                
            #     te_df.to_csv(f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}_best_te_pred.csv", index=False)

    # print values
    values = [epoch, lr, tr_loss_d, tr_loss_y, val_loss_d_list[args.table_idx], val_loss_y_list[args.table_idx], test_mae_d_list[args.table_idx], test_mae_y_list[args.table_idx], test_rmse_d_list[args.table_idx], test_rmse_y_list[args.table_idx], ]    
    values.append(time.time() - time_ep)
    table = tabulate.tabulate([values], headers=columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 20 == 0 or epoch == 1:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)
    
    # step scheduler
    if args.scheduler == 'cos_anneal':
        scheduler.step()
    
    # update wandb
    if not args.ignore_wandb:
        wandb_log = {
        "train/d": tr_loss_d,
        "train/y": tr_loss_y,
        "concat/valid_d": val_loss_d_list[0],
        "concat/valid_y": val_loss_y_list[0],
        "concat/test_total (mae)": test_mae_d_list[0] + test_mae_y_list[0],
        "concat/test_d (mae)": test_mae_d_list[0],
        "concat/test_y (mae)": test_mae_y_list[0],
        "concat/test_total (rmse)": test_rmse_d_list[0] + test_rmse_y_list[0],
        "concat/test_d (rmse)": test_rmse_d_list[0],
        "concat/test_y (rmse)": test_rmse_y_list[0],
        "setting/lr": lr,
    }

        for i in range(1, 6):
            wandb_log.update({
                f"date_{i}/valid_d": val_loss_d_list[i],
                f"date_{i}/valid_y": val_loss_y_list[i],
                f"date_{i}/test_total (mae)": test_mae_d_list[i] + test_mae_y_list[i],
                f"date_{i}/test_d (mae)": test_mae_d_list[i],
                f"date_{i}/test_y (mae)": test_mae_y_list[i],
                f"date_{i}/test_total (rmse)": test_rmse_d_list[i] + test_rmse_y_list[i],
                f"date_{i}/test_d (rmse)": test_rmse_d_list[i],
                f"date_{i}/test_y (rmse)": test_rmse_y_list[i],
            })

        wandb.log(wandb_log)
# ---------------------------------------------------------------------------------------------



## Print Best Model ---------------------------------------------------------------------------
print(f"Best {args.model} achieved [d:{best_test_losses[args.table_idx][0]}, y:{best_test_losses[args.table_idx][1]}] on {best_epochs[args.table_idx]} epoch!!")
print(f"The model saved as '{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}_best_val.pt'!!")
if args.ignore_wandb == False:
    for i in range(6):
        date_key = "concat" if i == 0 else f"date_{i}"
        wandb.run.summary[f"best_epoch {date_key}"] = best_epochs[i]
        wandb.run.summary[f"best_val_loss (d) {date_key}"] = best_val_loss_d[i]
        wandb.run.summary[f"best_val_loss (y) {date_key}"] = best_val_loss_y[i]
        wandb.run.summary[f"best_val_loss {date_key}"] = best_val_loss_d[i] + best_val_loss_y[i]
        wandb.run.summary[f"best_te_mae_loss (d) {date_key}"] = best_test_losses[i][0]
        wandb.run.summary[f"best_te_mae_loss (y) {date_key}"] = best_test_losses[i][1]
        wandb.run.summary[f"best_te_mae_loss_{date_key}"] = best_test_losses[i][0] + best_test_losses[i][1]
        wandb.run.summary[f"best_te_rmse_loss (d) {date_key}"] = best_test_losses[i][2]
        wandb.run.summary[f"best_te_rmse_loss (y) {date_key}"] = best_test_losses[i][3]
        wandb.run.summary[f"best_te_rmse_loss {date_key}"] = best_test_losses[i][2] + best_test_losses[i][3]

    wandb.run.summary["tr_dat_num"] = concat_tr_num_data
    # wandb.run.summary["val_dat_num"] : concat_val_num_data
    # wandb.run.summary["te_dat_num"] : concat_te_num_data
# ---------------------------------------------------------------------------------------------
