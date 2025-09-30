import os
import sys
import re
import csv
import logging
from datetime import datetime
from pathlib import Path
import umap
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
from ecg_data_500Hz import *
from engine_training import *
from models import load_encoder
from linear_probe_utils import features_dataloader



from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Patch

# Define colormaps

soft_red = '#C62828'   # warm side (positive)
petrol = '#77B5A9'     # cool side (negative)

# Define your colors
colors = [petrol, '#ffffff', soft_red]  # diverging: cool → neutral → warm
cmap_name = 'petrol_softred'

# Create the colormap
custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)


def parse():
    parser = argparse.ArgumentParser("ECG downstream training")

    # parser.add_argument('--model_name',
    #                     default="ejepa_random",
    #                     type=str,
    #                     help='resume from checkpoint')

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


    parser.add_argument("--dataset", default="ptbxl", type=str, help="dataset name")

    parser.add_argument(
        "--data_dir",
        default="/mount/ecg/ptb-xl-1.0.3/",  # "/mount/ecg/cpsc_2018/"
        type=str,
        help="dataset directory",
    )

    parser.add_argument(
        "--task", default="multilabel", type=str, help="downstream task"
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

def main(config):
    os.makedirs(config["output_dir"], exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Create log filename with current time
    ckpt_name = os.path.splitext(os.path.basename(config["ckpt_dir"]))[0]
    log_filename = os.path.join(
        config["output_dir"],
        f"log_{ckpt_name}_{config['task']}_{config['dataset']}_{current_time}.txt",
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


    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    data_path = './training_data/'
    logging.info(f"Loading {config['dataset']} dataset...")
    print(f"Loading {config['dataset']} dataset...")




    ID_train = torch.load(os.path.join(data_path, "IDs_train.pt"), weights_only=False)
    waves_train = torch.load(os.path.join(data_path,  "ecgs_train.pt"))
    labels_train = torch.load(os.path.join(data_path, "mvo_train.pt"))

    binary_labels = []

    for raw_id in ID_train:
        if isinstance(raw_id, bytes):
            raw_id = raw_id.decode("utf-8")

        try:
            # assuming raw_id is something like "698_-0.3.pt"
            timing_str = raw_id.replace('.pt', '').split('_')[-1]
            timing_value = float(timing_str)
        except ValueError:
            raise ValueError(f"Could not convert ID '{raw_id}' to float.")

        # 1 if positive, 0 otherwise
        binary = torch.tensor([1, 1]) if 0 < timing_value < 6 else \
         torch.tensor([0, 0]) if timing_value < 0 else \
         torch.tensor([2, 2]) if 6 < timing_value < 24 else \
         torch.tensor([3, 3])
        binary_labels.append(binary)

    # Stack all tensors into one tensor [N, 2]
    binary_labels_tensor = torch.stack(binary_labels)
    # Update patients dict
    ID = [s.lstrip('0') for s in ID_train]


    # Get all unique patient IDs from IDs_train
    all_patient_ids = {s.split('_')[0].lstrip('0') for s in ID_train}


    SEED = 42  # Or any fixed number
    np.random.seed(SEED)
    # Assume args.data_percentage exists and is a float in (0, 1] 

    ### take only eight leads
    waves_train = np.concatenate(
        (waves_train[:, :2, :], waves_train[:, 6:, :]), axis=1
    )
    

    print('shape of training data after subsampling', waves_train.shape, flush = True)


    device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Loading encoder from {config['ckpt_dir']}...")
    print(f"Loading encoder from {config['ckpt_dir']}...")
    encoder, embed_dim = load_encoder(ckpt_dir=config["ckpt_dir"])
    encoder = encoder.to(device)

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Settings
    method = "umap"  # Choose between "pca", "tsne", "umap"

    # Define colors for MVO (target label)
    soft_red = '#C62828'   # MVO yes
    petrol = '#77B5A9'     # MVO no
    binary_colors = [petrol, soft_red]

    # Prepare figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # Loop over 4 timing groups
    for selected_group in range(4):
        print(f"\nProcessing group {selected_group}...")

        selected_indices = (binary_labels_tensor[:, 0] == selected_group)
        dataset_with_labels = ECGDataset(waves_train[selected_indices], labels_train[selected_indices])

        all_labels = []
        all_features = []

        dataloader_with_labels = torch.utils.data.DataLoader(
            dataset_with_labels, batch_size=128, shuffle=False, num_workers=2
        )

        with torch.no_grad():
            for wave, target in dataloader_with_labels:
                repr = encoder.representation(wave.to(device))  # (bs, dim)
                all_features.append(repr.cpu())
                all_labels.append(target[:, 1])

        if len(all_features) == 0:
            print(f"No data in group {selected_group}. Skipping.")
            continue

        embeddings_valid = torch.cat(all_features)
        times = torch.cat(all_labels).int()

        # Dimensionality reduction
        if method == "pca":
            reducer = PCA(n_components=2)
        elif method == "tsne":
            reducer = TSNE(n_components=2, perplexity=100, random_state=12)
        elif method == "umap":
            reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.5, n_neighbors=15)

        embeddings_2d = reducer.fit_transform(embeddings_valid)

        # Assign colors based on MVO label
        color_list = [binary_colors[val] for val in times]

        # Plot into subplot
        ax = axes[selected_group]
        ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=color_list,
            s=40,
            alpha=0.7
        )

        ax.set_title(
            ["Before PCI", "0–6h After", "6–24h After", ">24h After"][selected_group],
            fontsize=15
        )
        ax.grid(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    # Global styling
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 13,
        'text.usetex': False
    })

    # Legend
    legend_elements = [
        Patch(facecolor=petrol, label='IMH no'),
        Patch(facecolor=soft_red, label='IMH yes')
    ]
    fig.legend(handles=legend_elements, loc='upper center', fontsize=14, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('./plots/latent_space_IMH.png')
if __name__ == "__main__":
    config = parse()

    main(config)




