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

# Data ---------------------------------------------------------
parser.add_argument(
    "--data_path",
    type=str,
    default='./data/data_norm.csv',
    help="path to datasets location",)
#----------------------------------------------------------------


# Model ---------------------------------------------------------
parser.add_argument(
    "--model",
    type=str, default='MLP',
    choices=["MagNet", "GCN", "GAT", "MLP", "Linear", "SVM"],
    help="model name (default : MLP)")

parser.add_argument("--save_path",
            type=str, default="./exp_result/",
            help="Path to save best model dict")

parser.add_argument(
    "--num_features",
    type=int, default=25,
    help="feature size (default : 25)"
)

parser.add_argument(
    "--hidden_dim",
    type=int, default=64,
    help="MLP model hidden size (default : 64)"
)

parser.add_argument(
    "--drop_out",
    type=float, default=0.0,
    help="Dropout Rate (Default : 0)"
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

parser.add_argument("--epochs", type=int, default=100, metavar="N",
    help="number epochs to train (Default : 100)")

parser.add_argument("--wd", type=float, default=5e-4, help="weight decay (Default: 5e-4)")

parser.add_argument("--scheduler", type=str, default=None, choices=[None, "cos_anneal"])

parser.add_argument("--t_max", type=int, default=50,
                help="T_max for Cosine Annealing Learning Rate Scheduler (Default : 50)")
#----------------------------------------------------------------

# Hyperparameter for setting --------------------------------------
parser.add_argument("--target_order", type=int, default=3,
          help="Decide how much order we get (Default : 3)") # 3: 3차까지는 주어진 상태에서, 4차 여부 예측

# parser.add_argument("--train_ratio", type=float, default=0.2,
#           help="Ratio of train data (Default : 0.2)") # 0.8: n차 감염의 20%를 train으로 활용
#----------------------------------------------------------------

args = parser.parse_args()
## ----------------------------------------------------------------------------------------------------



## Set seed and device ----------------------------------------------------------------
utils.set_seed(args.seed)

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {args.device}")
#-------------------------------------------------------------------------------------

## Set wandb ---------------------------------------------------------------------------
if args.ignore_wandb == False:
    wandb.init(project="cluster-medical-ai")
    wandb.config.update(args)
    wandb.run.name = f"{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out} for order {args.target_order}"


## Load Data --------------------------------------------------------------------------------
# final_data_list, y_min, y_max = utils.full_load_data(data_path = args.data_path,  
#                                         num_features = args.num_features,
#                                         target_order = args.target_order,
#                                         train_ratio = 0.2,
#                                         classification = True,
#                                         device = args.device,
#                                         model_name = args.model)
data = pd.read_csv(args.data_path)
dataset = utils.Tabledata(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print("Successfully load data!")
#-------------------------------------------------------------------------------------


## Model ------------------------------------------------------------------------------------
if args.model == "MagNet":
    model_type = 'graph'
    model = models.MagNet(in_channels=args.num_features, hidden=8, drop_out=args.drop_out).to(args.device) # hidden -> q -> K 차례로 수정하면서 성능 변화 있는지 파악
elif args.model == "GCN":
    model_type = 'graph'
    model = models.GCN_Net(in_channels = args.num_features, drop_out=args.drop_out).to(args.device)

elif args.model == "GAT":
    model_type = 'graph'
    model = models.GAT_Net(in_channels = args.num_features, drop_out=args.drop_out).to(args.device)

elif args.model == "MLP":
    model_type = 'non_graph'
    model = models.MLPRegressor(input_size=args.num_features,
                    hidden_size=args.hidden_dim,
                    output_size=1, drop_out=args.drop_out).to(args.device)

elif args.model == "Linear":
    model_type = 'non_graph'
    model = models.LinearRegression(input_size=args.num_features,
                    out_channels=1).to(args.device)

elif args.model == "SVM" :
    model_type = 'non_graph'
    raise NotImplementedError

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
columns = ["ep", "lr", f"tr_loss({args.criterion})", 
        f"val_data_loss({args.eval_criterion})", f"val_group_loss({args.eval_criterion})", f"val_worst_loss({args.eval_criterion})",
        f"te_data_loss({args.eval_criterion})",  f"te_group_loss({args.eval_criterion})", f"te_worst_loss({args.eval_criterion})",
        "time"]

best_val_data_loss = 9999

for epoch in range(1, args.epochs + 1):
    time_ep = time.time()
    lr = optimizer.param_groups[0]['lr']

    tr_epoch_loss = 0; val_epoch_loss = 0; te_epoch_loss = 0

    total_tr_num_data = 0; total_val_num_data = 0; total_te_num_data = 0

    tr_ground_truth_list = []; val_ground_truth_list = []; te_ground_truth_list = []

    tr_predicted_list = []; val_predicted_list = []; te_predicted_list = []

    for itr, data in enumerate(dataloader):
        tr_loss, tr_num_data, tr_predicted, tr_ground_truth = utils.train(data, model, optimizer, criterion)
        val_loss, val_num_data, val_predicted, val_ground_truth = utils.valid(data, model, eval_criterion)
        te_loss, te_num_data, te_predicted, te_ground_truth = utils.test(data, model, eval_criterion)

        tr_epoch_loss += tr_loss
        val_epoch_loss += val_loss
        te_epoch_loss += te_loss

        total_tr_num_data += tr_num_data
        total_val_num_data += val_num_data
        total_te_num_data += te_num_data

        tr_ground_truth_list += list(tr_ground_truth.cpu().numpy())
        val_ground_truth_list += list(val_ground_truth.cpu().numpy())
        te_ground_truth_list += list(te_ground_truth.cpu().numpy())

        tr_predicted_list += list(tr_predicted.detach().cpu().numpy())
        val_predicted_list += list(val_predicted.detach().cpu().numpy())
        te_predicted_list += list(te_predicted.detach().cpu().numpy())

    # Data Loss
    tr_avg_loss = tr_epoch_loss / total_tr_num_data
    val_avg_loss = val_epoch_loss / total_val_num_data
    te_avg_loss = te_epoch_loss / total_te_num_data
    
    tr_ground_truth_list = np.asarray(tr_ground_truth_list)
    val_ground_truth_list = np.asarray(val_ground_truth_list)
    te_ground_truth_list = np.asarray(te_ground_truth_list)

    tr_predicted_list = np.asarray(tr_predicted_list)
    val_predicted_list = np.asarray(val_predicted_list)
    te_predicted_list = np.asarray(te_predicted_list)
    

    # Calculate Validation Group loss
    val_group_loss_dict, val_group_loss = utils.cal_group_loss(val_ground_truth_list, val_predicted_list, args.eval_criterion)
    val_worst_loss = max(val_group_loss_dict.values())

    # Calculate Test Group loss
    te_group_loss_dict, te_group_loss = utils.cal_group_loss(te_ground_truth_list, te_predicted_list, args.eval_criterion)
    te_worst_loss = max(te_group_loss_dict.values())

    time_ep = time.time() - time_ep

    values = [epoch, lr, tr_avg_loss,
            val_avg_loss, val_group_loss, val_worst_loss,
            te_avg_loss, te_group_loss, te_worst_loss,
            time_ep,]

    table = tabulate.tabulate([values], headers=columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 20 == 0 or epoch == 1:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)

    if args.scheduler is not None:
        scheduler.step()

    # Save Best Model (Early Stopping)
    if val_avg_loss < best_val_data_loss:
        best_epoch = epoch
        best_val_data_loss = val_avg_loss
        best_val_group_loss = val_group_loss
        best_val_worst_loss = val_worst_loss

        best_te_data_loss = te_avg_loss
        best_te_group_loss = te_group_loss
        best_te_worst_loss = te_worst_loss

        
        # save state_dict
        os.makedirs(args.save_path, exist_ok=True)
        utils.save_checkpoint(file_path = f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}-{args.target_order}_best_val.pt",
                            epoch = epoch,
                            state_dict = model.state_dict(),
                            optimizer = optimizer.state_dict(),
                            )

        # save prediction and ground truth as csv
        val_df = pd.DataFrame({'val_pred':val_predicted_list,
                        'val_ground_truth':val_ground_truth_list})
        val_df.to_csv(f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}-{args.target_order}_best_val_pred.csv")

        te_df = pd.DataFrame({'te_pred' : te_predicted_list,
                        'te_ground_truth' : te_ground_truth_list})                
        te_df.to_csv(f"{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}-{args.target_order}_best_te_pred.csv")
        
    
    if args.ignore_wandb == False:
        wandb.log({"lr" : lr,
                "Training loss" : tr_avg_loss,
                "Validation data loss": val_avg_loss, "Validation group loss":val_group_loss, "Validation worst loss" : val_worst_loss, 
                "Test data loss" : te_avg_loss, "Test group loss" : te_group_loss, "Test worst loss" : te_worst_loss,
                })
# ---------------------------------------------------------------------------------------------



## Print Best Model ---------------------------------------------------------------------------
print(f"Best {args.model} achieved {best_te_data_loss} on {best_epoch} epoch!!")
print(f"The model saved as '{args.save_path}/{args.model}-{args.optim}-{args.lr_init}-{args.wd}-{args.drop_out}-{args.target_order}_best_val.pt'!!")

if args.ignore_wandb == False:
    wandb.run.summary["best_epoch"]  = best_epoch
    wandb.run.summary["best_val_data_loss"] = best_val_data_loss
    wandb.run.summary["best_val_group_loss"] = best_val_group_loss
    wandb.run.summary["best_val_worst_loss"] = best_val_worst_loss

    wandb.run.summary["best_test_loss"] = best_te_data_loss
    wandb.run.summary["best_test_group_loss"] = best_te_group_loss
    wandb.run.summary["best_test_worst_loss"] = best_te_worst_loss

    wandb.run.summary["Number of Traing Data"] = total_tr_num_data
    wandb.run.summary["Number of Validation Data"] : total_val_num_data
    wandb.run.summary["Number of Test Data"] : total_te_num_data
# ---------------------------------------------------------------------------------------------