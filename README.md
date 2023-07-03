# Robust Optimization for PPG-based Blood Pressure Estimation

This repository contains PyTorch implemenations of Sequential Cluster Regression for COVID Epidemiology by Python.

## Introduction
This is a machine learning code that uses coronavirus cluster data to predict how many people each cluster will infect and how long the cluster will last. 
For the layer that embeds each information, there are two layers: a layer that processes cluster information (e.g., severity index etc.) and a layer that processes patient information (e.g., age, address, etc.).
For the Transformer model, we used a sequence of dates for each cluster and performed a regression using the correlation of each sequence.

### 1. Project Tree
```
├── dir1
│   ├── file11.ext
│   └── file12.ext
├── dir2
│   ├── file21.ext
│   ├── file22.ext
│   └── file23.ext
├── dir3
├── file_in_root.ext
└── README.md
```

The dataset used in this work is the MIMIC-III Waveform Database Matched Subset, provided and publicly available by PhysioNet.
It's available to check the code for preprocessing step in  ``` ./preprocessing ```

We also suggest Time-CutMix (TC) to augment data by concatenating two sequences randomly in training phase.

### 2. Model perspective
![overall_method](https://user-images.githubusercontent.com/84635206/207836785-4983911c-f5c4-4ba1-9130-feb3515e74a2.png)

In the paper, we propose Transformer with Convolution layer to handle the local context.

### 3. Loss perspective
The tranditional ERM shows worse performance on minority group.
To alleviate this problem, we strongly minimize the training loss for group that has a small number of dataset more.

#### C-REx
C-REx compute the proportion of each group among the total dataset and add the variance regularization term with the original ERM term.

#### D-REx
D-REx apply Tukey's Ladder of Power Transformation to reduce the skewness and make the observed dataset more Gaussian-like distribution.
Using the transformed data distribution, we measure the symmetric KL divergence between the total and each group distribution.
Finally, D-REx add the distribution distances on the variance regularization term.

#### CD-REx
CD-REx employ regularization term of both C-REx and D-REx.



## Implementation
The script `main.py` allows to make data pickle file and train all the baselines we consider.

To make pickle file of data use this:
(In this paper, we only utilize erm data loader.)
```
main.py --method=erm --sampling=<SAMPLING> --small_ratio=<SMALL_RATIO> --save_pkl --exit
```

Parameters:
* ```SAMPLING``` &mdash; sampling methods to treat imbalance (default: normal) :
    - normal
    - upsampling
    - downsampling
    - small
* ```SMALL_RATIO``` &mdash; ratio of training data when SAMPLING is small


If u want to make ERM downsampling/upsampling small loader, you have to use this command respectively:
```
main.py --method=erm --sampling=small --small_ratio=<SMALL_RATIO> --down_small --save_pkl --exit
```
```
main.py --method=erm --sampling=small --small_ratio=<SMALL_RATIO> --up_small --save_pkl --exit
```
----

To train proposed methods use this:
```
main.py --method=<METHOD>                     \
        --max_epoch=<MAX_EPOCH>               \
        --model=<MODEL>                       \
        --wd=<WD>                             \
        --lr=<LR>                             \
        --dropout=<DROPOUT>                   \
        --annealing_epoch=<ANNEALING_EPOCH>   \
        --robust_step_size=<ROBUST_STEP_SIZE> \
        [--sbp_beta=<SBP_BETA>]               \
        [--dbp_beta=<DBP_BETA>]               \
        --erm_loader                          \
        [--tukey]                             \
        [--beta=<BETA>]                       \
        [--C1=<C1>]                           \
        [--C21=<C21>]                         \
        [--C22=<C22>]                         \
        [--mixup]
        [--wandb]
```
Parameters:
* ```METHOD``` &mdash; define training methods (default: ERM) :
    - erm
    - dro
    - vrex
    - ours-1 (C-REx)
    - ours-2 (D-REx)
    - ours-both (CD-REx)
* ```MAX_EPOCH``` &mdash; number of training epochs (default: 1000)
* ```MODEL``` &mdash; model name (default: ConvTransformer) :
    - Transformer
    - ConvTransformer
* ```WD``` &mdash; weight decay (default: 5e-3)
* ```LR``` &mdash; initial learning rate (default: 1e-3)
* ```DROPOUT``` &mdash; dropout rate (default: 0.1)
* ```ANNEALING_EPOCH``` &mdash; number of annealing epoch (default: 500)
* ```ROBUST_STEP_SIZE``` &mdash; step size for annealing in DRO (default: 0.01)
* ```SBP_BETA``` &mdash; scale of SBP variance regularization term (default: 1)
* ```DBP_BETA``` &mdash; scale of DBP variance regularization term (default: 0.1)
* ```--erm_loader``` &mdash; use dataloader of ERM without any resampling
* ```--tukey``` &mdash; employ Tukey's Ladder of Power Transformation
* ```BETA``` &mdash; scale of Tukey's Ladder of Power Transformation (default: 0.5)
* ```C1``` &mdash; scale variance with number of training data per group (default: 1e-5)
* ```C21``` &mdash; scale variance for SBP loss with kl divergence per group (default: 1e-4)
* ```C22``` &mdash; scale variance for DBP loss with kl divergence per group (default: 5e-4)
* ```--mixup``` &mdash; Run Time-CutMix for data augmentation
* ```--wandb``` &mdash; Use WandB to save results

----
### ERM best model reproduction
```
main.py --method=erm --model=ConvTransformer                                  \
        --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
        --wd=5e-3 --lr=1e-3 --dropout=0.1 --max_epoch=1000
```

### ERM Undersampling best model reproduction
```
main.py --method=erm --model=ConvTransformer --sampling=downsampling          \
        --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
        --wd=5e-3 --lr=1e-3 --dropout=0.1 --max_epoch=1000
```

### ERM Oversampling best model reproduction
```
main.py --method=erm --model=ConvTransformer --sampling=upsampling            \
        --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
        --wd=5e-3 --lr=1e-3 --dropout=0.1 --max_epoch=1000
```

### DRO best model reproduction
```
main.py --method=dro --model=ConvTransformer                                  \
        --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
        --wd=1e-1 --lr=1e-3 --dropout=0.1 --erm_loader --max_epoch=1000       \
        --robust_step_size=0.01
```


### V-REx best model reproduction
```
main.py --method=vrex --model=ConvTransformer                                    \
        --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4    \
        --wd=5e-3 --lr=1e-3 --dropout=0.1 --max_epoch=1000 --annealing_epoch=500 \
        --sbp_beta=1 --dbp_beta=0.1 --erm_loader
```


### C-REx best model reproduction
```
main.py --method=ours-1 --model=ConvTransformer                                   \
        --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4     \
        --wd=5e-3 --lr=1e-3 --dropout=0.1  --max_epoch=1000 --annealing_epoch=500 \
        --sbp_beta=1 --dbp_beta=0.1 --erm_loader --C1=1e-4
```


### D-REx best model reproduction
```
main.py --method=ours-2 --model=ConvTransformer                                   \
        --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4     \
        --wd=5e-3 --lr=1e-3 --dropout=0.1 --max_epoch=1000 --annealing_epoch=500  \
        --sbp_beta=1 --dbp_beta=0.1 --erm_loader --tukey --C21=1e-4 --C22=1e-4
```


### CD-REx + TimeCutMix best model reproduction
```
main.py --method=ours-both --model=ConvTransformer                                \
        --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4     \
        --wd=5e-3 --lr=1e-3 --dropout=0.1  --max_epoch=1000 --annealing_epoch=500 \
        --sbp_beta=1 --dbp_beta=0.1 --erm_loader                                  \
        --C1=1e-5 --tukey --C21=1e-4 --C22=5e-4 --mixup
```

-----
### CD-REx + TimeCutMix with small loader (small ratio = 0.5) best model reproduction
```
main.py --method=ours-both --model=ConvTransformer                               \
        --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4    \
        --wd=5e-3 --lr=1e-3 --dropout=0.1 --max_epoch=1000 --annealing_epoch=500 \
        --sbp_beta=1 --dbp_beta=0.1 --erm_loader                                 \
        --C1=5e-4 --tukey --C21=1e-3 --C22=1e-3                                  \
        --sampling=small --small_ratio=0.5 --mixup
```

------
To test pre-trained model use this:
```
test.py --model=ConvTransformer --num_filters=8 --d_model=96 --d_input=64 --num_layer=2 --num_heads=4 \
        --sampling=<SAMPLING> --small_ratio=<SMALL_RATIO> --seed=<SEED> --resume=<RESUME>             \
        [--rmse]
```
Parameters:
* ```SAMPLING``` &mdash; sampling methods to treat imbalance (default: normal) :
    - normal
    - upsampling
    - downsampling
    - small
* ```SMALL_RATIO``` &mdash; ratio of training data when SAMPLING is small
* ```SEED``` &mdash; random seed (default: 0)
* ```RESUME``` &mdash; pre-trained model we want to evaluate
* ```--rmse``` &mdash; set RMSE as evaluation criterion 
