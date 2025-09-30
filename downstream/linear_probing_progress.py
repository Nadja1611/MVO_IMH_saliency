import os
import sys
import re
import csv
import logging
from datetime import datetime
from pathlib import Path

import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_curve, auc as auc_sk
from scipy import stats
from helpers import ci_bounds
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from timm.scheduler import CosineLRScheduler

from collections import defaultdict

# Append parent directory to sys.path for module imports
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

# Project-specific imports
from ecg_data_250Hz import *
from engine_training import *
from models import load_encoder
from linear_probe_utils import (
    features_dataloader,
    train_multilabel,
    LinearClassifier,
    Finetune_Classifier
)
from augmentation import *

#%%


def parse():
    parser = argparse.ArgumentParser("ECG downstream training")

    parser.add_argument(
        "--ckpt_dir",
        default="../weights/multiblock_epoch100.pth",
        type=str,
        metavar="PATH",
        help="pretrained encoder checkpoint",
    )

    parser.add_argument(
        "--output_dir",
        default="./output/linear_eval",
        type=str,
        metavar="PATH",
        help="output directory",
    )
    parser.add_argument(
        "--factor_lr",
        default=0.01,
        type=float,
        help="factor by which LR is decreased in finetuning step",
    )

    parser.add_argument(
        "--decay_lr",
        default=0.975,
        type=float,
        help="decay for Exponential LR decay used in finetuning step",
    )

    parser.add_argument(
        "--dropout",
        default=0.2,  # "/mount/ecg/cpsc_2018/"
        type=float,
        help=0.0,
    )


    parser.add_argument(
        "--task", default="multilabel", type=str, help="downstream task"
    )
    parser.add_argument(
        "--validation", default="crossval", type=str, help="crossvalidation or bootstrapping"
    )


    # Use parse_known_args instead of parse_args
    args, unknown = parser.parse_known_args()

    with open(
        os.path.realpath(f"../configs/downstream/linear_eval/linear_eval_ejepa.yaml"),
        "r",
    ) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    for k, v in vars(args).items():
        if v:
            config[k] = v

    return config



#%%
def main(config):

    os.makedirs(config["output_dir"], exist_ok=True)
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create log filename with current time
    ckpt_name = os.path.splitext(os.path.basename(config["ckpt_dir"]))[0]
    log_filename = os.path.join(
        config["output_dir"],
        f"log_{ckpt_name}_{config['task']}_{current_time}.txt",
    )
    # Configure logging
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Log the config dictionary
    logging.info("Configuration:")
    logging.info(yaml.dump(config, default_flow_style=False))

    #%%
    seed = 24
    torch.manual_seed(seed)
    np.random.seed(seed)

    #%%
    # Initialize TensorBoard writer
    run_name = (
        f"FT_{config['task']}_ckpt_{ckpt_name}_"
        f"data"
    )


    log_dir = Path("./tensorboard") / run_name
    log_base = Path("./tensorboard")
    #%%
    csv_base = Path("./csv/csv_weights_finetune_bootstrap_paper_tb")
    csv_dir = csv_base / run_name
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"metrics_{ckpt_name}.csv"

    #%%
    #### define empty lists where metrics will be saved

    MEAN_AUC = []
    Fold_mean_auc = []
    Fold_mean_acc = []
    Fold_mean_f1 = []
    Fold_mean_acc_class = []
    Fold_mean_auc_class = []
    Fold_mean_f1_class = []
    Fold_mean_auprc = []
    Fold_mean_auprc_per_class = []
    Fold_mean_recall = []
    Fold_mean_recall_per_class = []
    Fold_mean_precision = []
    Fold_mean_precision_per_class = []
    Fold_mean_acc = []
    #%%
    ### load the data ###
    data_path = './training_data_250/'

    ID = torch.load(os.path.join(data_path, "IDs_train.pt"), weights_only=False)
    ID = [s.lstrip('0') for s in ID]
    ID = [s.replace('.pt', '') for s in ID]
    waves = torch.load(os.path.join(data_path,  "ecgs_train.pt"))
    labels = torch.load(os.path.join(data_path, "mvo_train.pt"))

    print('shape of ecgs, ', waves.shape, flush = True) 


    #%%
    # === Create patient dictionary for splits from ID only
    patients = defaultdict(list)
    for i, fname in enumerate(ID):
        pid, time_str = fname.replace('.pt', '').split('_')
        label = labels[i]
        time_val = float(time_str)
        ## consists of: 'Pat_ID', [(Pat_ID_time.pt, time, label),....]
        patients[pid].append((fname, time_val, label))

    print(f"Subset patient count: {len(patients)}", flush=True)
    #%%
    # === Bootstrap/Crossval splits using this subset
    splits = crossval_split_ids(patients, n_folds=5, seed = seed)
    #%%
    # === Define transforms ===
    aug = {
        "rand_augment": {
            "use": True,
            "kwargs": {
                "op_names": ["shift", "cutout", "drop", "flip", "erase", "sine", "partial_sine", "partial_white_noise"],
                "level": 10,
                "num_layers": 2,
                "prob": 0.5,
            },
        },
        "train_transforms": [
            {"highpass_filter": {"fs": 250, "cutoff": 0.67}},
            {"lowpass_filter": {"fs": 250, "cutoff": 40}},
        ],
        "eval_transforms": [
            {"highpass_filter": {"fs": 250, "cutoff": 0.67}},
            {"lowpass_filter": {"fs": 250, "cutoff": 40}},
        ],
    }

    train_transforms = get_transforms_from_config(aug["train_transforms"])
    if aug["rand_augment"]["use"]:
        print("We use RandAug", flush=True)
        randaug = get_rand_augment_from_config(aug["rand_augment"]["kwargs"])
        train_transforms.append(randaug)
    train_transforms = Compose(train_transforms + [ToTensor()])
    test_transforms = Compose(get_transforms_from_config(aug["eval_transforms"]) + [ToTensor()])
    #%%
    # === Evaluation Storage ===
    n_classes = 3
    tpr_folds = [[] for _ in range(n_classes)]
    auc_folds = [[] for _ in range(n_classes)]
    #%%
    ## now we loop over created folds
    fold = 0
    for fold_idx, (train_ids, val_ids, test_ids) in enumerate(splits):
        fold += 1
        # Initialize empty lists before training loop
        All_probs, All_targets = [], []
        # Create training, validation and test folds
        train_mask = [id_str in train_ids for id_str in ID]
        val_mask   = [id_str in val_ids   for id_str in ID]
        test_mask  = [id_str in test_ids  for id_str in ID]

        # Slice the data
        ids_train, waves_train, labels_train = np.array(ID)[train_mask], waves[train_mask], labels[train_mask]
        ids_val, waves_val, labels_val     = np.array(ID)[val_mask], waves[val_mask], labels[val_mask]
        ids_test, waves_test, labels_test   = np.array(ID)[test_mask], waves[test_mask], labels[test_mask]

        print(f"Fold {fold_idx+1}: Train={len(waves_train)}, Val={len(waves_val)}, Test={len(waves_test)})")
        print(len(labels_test), flush = True)

        # Assume args.data_percentage exists and is a float in (0, 1] 

        ### take only eight leads
        waves_train = np.concatenate(
            (waves_train[:, :2, :], waves_train[:, 6:, :]), axis=1
        )
        waves_val = np.concatenate(
            (waves_val[:, :2, :], waves_val[:, 6:, :]), axis=1)
        waves_test = np.concatenate(
            (waves_test[:, :2, :], waves_test[:, 6:, :]), axis=1
        )
        #%%
        ### we save the testing data 
        if config['validation'] == 'crossval':
            os.makedirs(f'./test_data/All_{ckpt_name}/', exist_ok=True)

            data_to_save = {
                'ecgs_test': waves_test,
                'mvo_test': labels_test,
                'id_test': ids_test
            }

            input_save_path = f"./test_data/All_{ckpt_name}/input_fold_{i}.pt"
            torch.save(data_to_save, input_save_path)

            print(f"Inputs saved to {input_save_path}", flush = True)

        print('shape of training data after subsampling', waves_train.shape, flush = True)
        
        if config["task"] == "multilabel":
            _, n_labels = labels_train.shape

        else:
            """ for the regression case, the output should be one number"""
            n_labels = 1

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"Loading encoder from {config['ckpt_dir']}...")
        print(f"Loading encoder from {config['ckpt_dir']}...")
        encoder, embed_dim = load_encoder(ckpt_dir=config["ckpt_dir"], drop_path_rate=0.0, drop_rate=0.0)
        encoder = encoder.to(device)

        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False



        logging.info(f"Start training...")
        print(f"Start training...")
        num_samples = len(waves_train)


        num_workers = config["dataloader"]["num_workers"]


        print('shape after subsampling, ', waves_train.shape, flush = True)
            
        if config["task"] == "multilabel":
            _, n_labels = labels_train.shape
        else:
            n_labels = len(np.unique(labels_train))

        device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info(f"Loading encoder from {config['ckpt_dir']}...")
        print(f"Loading encoder from {config['ckpt_dir']}...")
        encoder, embed_dim = load_encoder(ckpt_dir=config["ckpt_dir"])
        encoder = encoder.to(device)

        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False


        logging.info(f"Start training...")
        print(f"Start training...")




        num_workers = config["dataloader"]["num_workers"]

        #### we load the dataset for linear probing , no augmentations used
        train_dataset = ECGDataset(waves_train, labels_train)
        val_dataset = ECGDataset(waves_val, labels_val)
        test_dataset = ECGDataset(waves_test, labels_test)

        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=32, shuffle=True, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=num_workers
        )




        bs = config["dataloader"]["batch_size"]
        train_loader_linear = features_dataloader(
            encoder, train_loader, batch_size=bs, shuffle=True, device=device
        )
        val_loader_linear = features_dataloader(
            encoder, val_loader, batch_size=bs, shuffle=False, device=device
        )
        test_loader_linear = features_dataloader(
            encoder, test_loader, batch_size=bs, shuffle=False, device=device
        )

        num_epochs = config["train"]["epochs"]
        num_epochs = 5
        lr = config["train"]["lr"]
        print('learning rate, ', lr, flush = True)
        criterion = (
            nn.BCEWithLogitsLoss()
            if config["task"] == "multilabel"
            else nn.CrossEntropyLoss()
        )
        linear_model = LinearClassifier(embed_dim, n_labels).to(device)
        optimizer = optim.AdamW(linear_model.parameters(), lr=lr)
        iterations_per_epoch = len(train_loader_linear)
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs * iterations_per_epoch,
            cycle_mul=1,
            lr_min=lr * 0.1,
            cycle_decay=0.1,
            warmup_lr_init=lr * 0.1,
            warmup_t=10,
            cycle_limit=1,
            t_in_epochs=True,
        )

       
        fold_log_dir = Path(log_dir) / f"bootstrap_fold_{fold}_{lr}"
        writer = SummaryWriter(log_dir=str(fold_log_dir))
        writer.flush()


        if config["task"] == "multilabel":
            acc, acc_per_class, auc, auc_per_class, f1, f1_per_class, auprc_macro, auprc_per_class, recall_macro, recall_per_class,precision_macro, precision_per_class, y_probs, y_true = train_multilabel(
                num_epochs,
                linear_model,
                optimizer,
                criterion,
                scheduler,
                train_loader_linear,
                val_loader_linear,
                test_loader_linear,
                device,
                print_every=True,
            )


        # Save only the linear classifier weights
        weight_dir = Path(f'./weights/Classifier_all_{ckpt_name}')
        weight_dir.mkdir(parents=True, exist_ok=True)
        linear_save_path = weight_dir / f"weights_fold_{fold_idx}.pth"
        torch.save(linear_model.state_dict(), linear_save_path)

       
        All_probs.append(y_probs)
        All_targets.append(y_true)
        auc_test = auc if auc is not None else float('nan')
        f1_test = f1 if f1 is not None else float('nan')
        logging.info(f"AUC: {auc_test:.3f}, F1: {f1_test:.3f}")
        print(f"AUC: {auc_test:.3f}, F1: {f1_test:.3f}")



        Fold_mean_acc.append(acc)
        Fold_mean_acc_class.append(acc_per_class)
        Fold_mean_auc.append(auc)
        Fold_mean_f1.append(f1)
        Fold_mean_auc_class.append(auc_per_class)
        Fold_mean_f1_class.append(f1_per_class)
        Fold_mean_auprc.append(auprc_macro)
        Fold_mean_auprc_per_class.append(auprc_per_class)
        Fold_mean_recall.append(recall_macro)
        Fold_mean_recall_per_class.append(recall_per_class)
        Fold_mean_precision.append(precision_macro)
        Fold_mean_precision_per_class.append(precision_per_class)       
        print(MEAN_AUC, flush = True)
        print(Fold_mean_auc, flush = True)
        print("length of the list is ", len(Fold_mean_auc), flush=True)

        # Example if you're gathering across folds
        all_y_probs = np.concatenate(All_probs, axis=0)   # shape: (N, C)
        all_y_true = np.concatenate(All_targets, axis=0)  # shape: (N, C)

        # Compute confidence intervals
        lower_auc, upper_auc = ci_bounds(Fold_mean_auc)
        lower_f1, upper_f1 = ci_bounds(Fold_mean_f1)
        lower_auprc, upper_auprc = ci_bounds(Fold_mean_auprc)
        lower_acc, upper_acc = np.nan, np.nan  # Not available in provided lists

        # Per-class AUC mean + CI
        per_class_auc = np.mean(np.array(Fold_mean_auc_class), axis=0)
        per_class_auc_ci_lows = []
        per_class_auc_ci_highs = []

        for i in range(len(per_class_auc)):
            per_class_values = [fold[i] for fold in Fold_mean_auc_class if len(fold) > i]
            if len(per_class_values) >= 2:
                low, high = ci_bounds(per_class_values)
            else:
                low, high = np.nan, np.nan
            per_class_auc_ci_lows.append(low)
            per_class_auc_ci_highs.append(high)


        # ---- Save to CSV ----
        write_header = not os.path.exists(csv_path)

        with open(csv_path, mode='a', newline='') as f:
            writer_csv = csv.writer(f)

            if write_header:
                header = [
                    "Step", "Accuracy", "AUC", "AUPRC", "F1", "Precision", "Recall",
                    "CI_AUC_low", "CI_AUC_high", "CI_AUPRC_low", "CI_AUPRC_high",
                    "CI_F1_low", "CI_F1_high", "CI_ACC_low", "CI_ACC_high"
                ]
                header += [f"AUC_class_{i}" for i in range(len(per_class_auc))]
                header += [f"CI_AUC_class_{i}_low" for i in range(len(per_class_auc))]
                header += [f"CI_AUC_class_{i}_high" for i in range(len(per_class_auc))]
                writer_csv.writerow(header)

            row = [
                step,
                np.mean(Fold_mean_acc),
                np.mean(Fold_mean_auc),
                np.mean(Fold_mean_auprc),
                np.mean(Fold_mean_f1),
                np.mean(Fold_mean_precision),
                np.mean(Fold_mean_recall),
                lower_auc,
                upper_auc,
                lower_auprc,
                upper_auprc,
                lower_f1,
                upper_f1,
                lower_acc,
                upper_acc
            ]
            row += list(per_class_auc)
            row += per_class_auc_ci_lows
            row += per_class_auc_ci_highs
            writer_csv.writerow(row)
            f.flush()


        # Per-class AUC mean + CI
        per_class_auc = np.mean(np.array(Fold_mean_auc_class), axis=0)
        per_class_auc_ci_lows = []
        per_class_auc_ci_highs = []

        for i in range(len(per_class_auc)):
            per_class_values = [fold[i] for fold in Fold_mean_auc_class if len(fold) > i]
            if len(per_class_values) >= 2:
                low, high = ci_bounds(per_class_values)
            else:
                low, high = np.nan, np.nan
            per_class_auc_ci_lows.append(low)
            per_class_auc_ci_highs.append(high)


        mean_fpr = np.linspace(0, 1, 100)

        # Map class order (i.e., 0 = IMH+, 1 = IMH-, 2 = both negative)
        # if you need to reorder targets/probs accordingly (optional, if All_probs/targets use fixed class order)
        for probs, targets in zip(All_probs, All_targets):
            for i, idx in enumerate([0,1,2]):  # reorder class indices: IMH+, IMH-, both negative
                fpr, tpr, _ = roc_curve(targets[:, idx], probs[:, idx])
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                tpr_folds[i].append(interp_tpr)
                auc_score = auc_sk(fpr, tpr)
                auc_folds[i].append(auc_score)
                print(auc_folds,flush = True)



        for i in range(n_classes):
            tprs = np.array(tpr_folds[i])
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)
            mean_auc = np.mean(auc_folds[i])
            std_auc = np.std(auc_folds[i])





        writer.flush()


if __name__ == "__main__":
    config = parse()

    main(config)

