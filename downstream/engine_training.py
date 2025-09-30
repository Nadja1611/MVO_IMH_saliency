from sklearn.metrics import roc_auc_score
from collections import defaultdict
import random
import math
import sys
from typing import Dict, Iterable, Optional, Tuple
import os
import torch

import util.misc as misc
import util.lr_sched as lr_sched

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,roc_auc_score
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold

import torchmetrics

def bootstrap_split(
    patients_dict,
    num_iterations=500,
    seed=42,
    val_ratio=0.15,
    test_ratio=0.15,
    label_tolerance=0.05,
    task="regression",  # or "regression"
):
    random.seed(seed)
    np.random.seed(seed)
    splits = []

    all_pids = list(patients_dict.keys())

    for iteration in range(num_iterations):
        random.shuffle(all_pids)

        n_total = len(all_pids)
        n_test = int(n_total * test_ratio)
        n_val = int(n_total * val_ratio)

        test_pids = all_pids[:n_test]
        remaining_pids = all_pids[n_test:]

        if task == "regression":
            # No need for label balancing
            random.shuffle(remaining_pids)
            val_pids = remaining_pids[:n_val]
            train_pids = remaining_pids[n_val:]

        else:  # multilabel or classification
            # Compute per-patient mean label vector (rounded for classification)
            pid_to_labels = {
                pid: np.mean([np.array(label) for _, _, label in patients_dict[pid]], axis=0).round().astype(int)
                for pid in remaining_pids
            }

            label_matrix = np.array(list(pid_to_labels.values()))
            avg_label_distribution = label_matrix.mean(axis=0)
            N = len(remaining_pids)

            # Try to find a balanced train/val split
            balanced = False
            attempt = 0
            while not balanced and attempt < 100:
                sampled_pids = random.sample(remaining_pids, N)
                unique_sampled_pids = list(set(sampled_pids))

                sampled_labels = np.array([pid_to_labels[pid] for pid in unique_sampled_pids])
                sampled_distribution = sampled_labels.mean(axis=0)

                deviation = np.abs(sampled_distribution - avg_label_distribution)
                if np.all(deviation <= label_tolerance):
                    balanced = True
                else:
                    attempt += 1

            # Train/val split
            random.shuffle(unique_sampled_pids)
            val_pids = unique_sampled_pids[:n_val]
            train_pids = unique_sampled_pids[n_val:]

        # Debug
        print('patients training ', train_pids[:14], flush=True)
        print('patients validation', val_pids[:14], flush=True)
        print('patients test      ', test_pids[:14], flush=True)

        # Sanity checks
        assert len(set(train_pids) & set(val_pids)) == 0
        assert len(set(train_pids) & set(test_pids)) == 0
        assert len(set(val_pids) & set(test_pids)) == 0
        
        if task == 'regression':
            train_files = [fname for pid in train_pids for fname, _, _,_ in patients_dict[pid]]
            val_files   = [fname for pid in val_pids   for fname, t, _, _ in patients_dict[pid] if t < 0]
            test_files  = [fname for pid in test_pids  for fname, t, _, _ in patients_dict[pid] if t < 0]
        else:    
            train_files = [fname for pid in train_pids for fname, t, _ in patients_dict[pid]]
            val_files   = [fname for pid in val_pids   for fname, t, _ in patients_dict[pid]]
            test_files  = [fname for pid in test_pids  for fname, t, _ in patients_dict[pid]]

        print(f'[{iteration+1}/{num_iterations}] Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}', flush=True)
        splits.append((train_files, val_files, test_files))

    return splits


def crossval_split_ids(patient_dict, n_folds=5, seed=42):
    """
    Perform k-fold cross-validation split at patient level into train, val, and test sets.
    Returns sets of 'pat_time' strings for direct indexing.

    Args:
        patient_dict (dict): {patient_id: [(filename, time after PCI, tensor), ...]}
        n_folds (int): Number of folds (default: 5)
        seed (int): Random seed for reproducibility.

    Returns:
        list of tuples: [(train_ids, val_ids, test_ids), ...] for each fold,
        where each element is a set of strings like '698_-0.3'.
    """
    random.seed(seed)

    # Shuffle patient IDs
    patient_ids = list(patient_dict.keys())
    random.shuffle(patient_ids)

    # Partition into folds
    folds = [patient_ids[i::n_folds] for i in range(n_folds)]

    results = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(sgkf.split(patient_ids, patient_labels, groups=patient_ids)):
        test_pats = set(patient_ids[i] for i in test_idx)

        # Step 3: Inner split (train vs val) — stratified again
        train_val_ids = [patient_ids[i] for i in train_val_idx]
        train_val_labels = [multilabel_to_class(patient_dict[pid][0][-1]) for pid in train_val_ids]

        inner_sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed + fold_idx)
        inner_train_idx, inner_val_idx = next(inner_sgkf.split(train_val_ids, train_val_labels, groups=train_val_ids))

        train_pats = set(train_val_ids[i] for i in inner_train_idx)
        val_pats   = set(train_val_ids[i] for i in inner_val_idx)

        # Flatten patient sets into file IDs
        train_ids = {fname for pid in train_pats for fname, *_ in patient_dict[pid]}
        val_ids   = {fname for pid in val_pats   for fname, *_ in patient_dict[pid]}
        test_ids  = {fname for pid in test_pats  for fname, *_ in patient_dict[pid]}

        # Get labels for printing
        train_labels = [multilabel_to_class(patient_dict[pid][0][-1]) for pid in train_pats]
        val_labels = [multilabel_to_class(patient_dict[pid][0][-1]) for pid in val_pats]
        test_labels = [multilabel_to_class(patient_dict[pid][0][-1]) for pid in test_pats]

        print(f"Fold {fold_idx + 1}")
        print(f"Train patients: {train_pats}")
        print(f"Val patients: {val_pats}")
        print(f"Test patients: {test_pats}")
        print(f"Train label distribution: {Counter(train_labels)}")
        print(f"Val label distribution: {Counter(val_labels)}")
        print(f"Test label distribution: {Counter(test_labels)}")
        print("-" * 40)

        results.append((train_ids, val_ids, test_ids))

    return results







def crossval_split(
    patients_dict,
    seed=23,
    val_ratio=0.15,
    label_tolerance=0.05,
    task="regression",  # or "classification"
    volume_dict=None
):
    random.seed(seed)
    np.random.seed(seed)
    splits = []

    all_pids = np.array(list(patients_dict.keys()))
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    for fold_idx, (trainval_idx, test_idx) in enumerate(kf.split(all_pids)):
        test_pids = all_pids[test_idx].tolist()
        trainval_pids = all_pids[trainval_idx].tolist()

        n_val = int(len(trainval_pids) * val_ratio)

        if task == "regression":
            random.shuffle(trainval_pids)
            val_pids = trainval_pids[:n_val]
            train_pids = trainval_pids[n_val:]

        else:  # classification: balance labels in trainval split
            pid_to_labels = {
                pid: np.mean([label for _, _, label in patients_dict[pid]], axis=0).round().astype(int)
                for pid in trainval_pids
            }
            avg_label_distribution = np.mean(list(pid_to_labels.values()), axis=0)
            N = len(trainval_pids)

            for _ in range(100):
                sampled_pids = random.sample(trainval_pids, N)
                sampled_labels = np.array([pid_to_labels[pid] for pid in sampled_pids])
                sampled_distribution = sampled_labels.mean(axis=0)

                if np.all(np.abs(sampled_distribution - avg_label_distribution) <= label_tolerance):
                    break

            random.shuffle(sampled_pids)
            val_pids = sampled_pids[:n_val]
            train_pids = sampled_pids[n_val:]

        # Sanity checks
        assert not set(train_pids) & set(val_pids)
        assert not set(train_pids) & set(test_pids)
        assert not set(val_pids) & set(test_pids)

        # Collect file paths and optionally volumes
        if task == "regression":
            train_files = [fname for pid in train_pids for fname, _, _, _ in patients_dict[pid]]
            val_files = [fname for pid in val_pids for fname, _, _, _ in patients_dict[pid]]
            test_files = [fname for pid in test_pids for fname, _, _, _ in patients_dict[pid]]

            train_volumes = [vol for pid in train_pids for vol in volume_dict[pid]] if volume_dict else None
            val_volumes = [vol for pid in val_pids for vol in volume_dict[pid]] if volume_dict else None
            test_volumes = [vol for pid in test_pids for vol in volume_dict[pid]] if volume_dict else None

        else:
            train_files = [fname for pid in train_pids for fname, _, _ in patients_dict[pid]]
            val_files = [fname for pid in val_pids for fname, _, _ in patients_dict[pid]]
            test_files = [fname for pid in test_pids for fname, _, _ in patients_dict[pid]]

            train_volumes = [vol for pid in train_pids for _, _, vol in volume_dict[pid]] if volume_dict else None
            val_volumes = [vol for pid in val_pids for _, _, vol in volume_dict[pid]] if volume_dict else None
            test_volumes = [vol for pid in test_pids for _, _, vol in volume_dict[pid]] if volume_dict else None

        print(f'[Fold {fold_idx+1}/5] Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}')

        if volume_dict:
            splits.append((train_files, val_files, test_files, test_volumes))
        else:
            splits.append((train_files, val_files, test_files))

    return splits






def filter_smallest_positive_time(ID, waves, labels, volumes, mvo_volumes):
    """
    Filters the data so that for each patient, only the ECG with the smallest positive time is kept.

    Parameters:
    - ID: list or numpy array of strings like '1234_15.2.pt'
    - waves, labels, volumes, mvo_volumes: torch.Tensor or list

    Returns:
    Filtered versions of ID, waves, labels, volumes, mvo_volumes.
    """

    # Ensure all IDs are strings
    ID = [str(x) for x in ID]

    patient_dict = defaultdict(list)

    for idx, id_str in enumerate(ID):
        basename = os.path.splitext(os.path.basename(id_str))[0]  # remove '.pt'
        if "_" not in basename:
            continue  # skip malformed entries

        try:
            patient_id, time_str = basename.split("_")
            time = float(time_str)
        except ValueError:
            continue  # skip if time is not a float

        if time > 0:
            patient_dict[patient_id].append((time, idx))

    # Select index with smallest positive time per patient
    selected_indices = [
        min(times, key=lambda x: x[0])[1]
        for times in patient_dict.values()
    ]

    # Return filtered data
    if isinstance(waves, torch.Tensor):
        return (
            [ID[i] for i in selected_indices],
            waves[selected_indices],
            labels[selected_indices],
            volumes[selected_indices],
            mvo_volumes[selected_indices]
        )
    else:
        return (
            [ID[i] for i in selected_indices],
            [waves[i] for i in selected_indices],
            [labels[i] for i in selected_indices],
            [volumes[i] for i in selected_indices],
            [mvo_volumes[i] for i in selected_indices]
        )


def plot_confusion_for_class(y_true, y_pred, class_index, class_name, path):
    # Convert multilabel indicator to binary vectors for the specified class index
    y_true_binary = y_true[:, class_index]
    y_pred_binary = y_pred[:, class_index]

    cm = confusion_matrix(y_true_binary, y_pred_binary)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix for {class_name}")
    plt.savefig(path)
    plt.close()


def compute_f1_precision_recall(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return f1, recall, precision


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    log_writer=None,
    config: Optional[dict] = None,
    use_amp: bool = True,
) -> Dict[str, float]:
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 50

    accum_iter = config["accum_iter"]
    max_norm = config["max_norm"]

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
       # if data_iter_step % accum_iter == 0:
      #      lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)


        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(samples)
            #  heatmap = compute_grad_cam(model.encoder_blocks.blocks[-1].attn)
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                targets = targets.to(dtype=outputs.dtype)
            # targets = torch.argmax(targets, dim=1)  # if targets are one-hot

            loss = criterion(
                outputs, targets
            )  # + custom_loss_with_prior(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((epoch + data_iter_step / len(data_loader)) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def plot_tp_vs_fn_volumes(
    y_true, y_probs, infarct_volumes, mvo_volumes,
    threshold=0.4, class_index=1, class_name="Class 1"
):
   
    y_pred = (np.array(y_probs) >= threshold).astype(int)
    y_true = np.array(y_true)

    tp_mask = (y_true[:, class_index] == 1) & (y_pred[:, class_index] == 1)
    fn_mask = (y_true[:, class_index] == 1) & (y_pred[:, class_index] == 0)

    infarct_volumes = np.array(infarct_volumes, dtype=object)
    mvo_volumes = np.array(mvo_volumes, dtype=object)

    # Filter out None or invalid entries
    valid_tp = tp_mask & np.array([
        isinstance(iv, (int, float)) and isinstance(mv, (int, float))
        for iv, mv in zip(infarct_volumes, mvo_volumes)
    ])
    valid_fn = fn_mask & np.array([
        isinstance(iv, (int, float)) and isinstance(mv, (int, float))
        for iv, mv in zip(infarct_volumes, mvo_volumes)
    ])

    tp_infarct = infarct_volumes[valid_tp].astype(float)
    tp_mvo = mvo_volumes[valid_tp].astype(float)

    fn_infarct = infarct_volumes[valid_fn].astype(float)
    fn_mvo = mvo_volumes[valid_fn].astype(float)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for boxplot: groups and positions
    data = [tp_infarct, tp_mvo, fn_infarct, fn_mvo]
    positions = [1, 2, 4, 5]
    colors = ['darkred', 'teal', 'darkred', 'teal']

    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True, showfliers=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Set x-axis labels centered between pairs
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(['True Positives', 'False Negatives'])
    ax.set_ylabel('Volume')
    ax.set_title(f'Infarct and MVO Volumes for {class_name} (Threshold = {threshold})')

    # Legend manually
    import matplotlib.patches as mpatches
    infarct_patch = mpatches.Patch(color='darkred', label='Infarct Volume')
    mvo_patch = mpatches.Patch(color='teal', label='MVO Volume')
    ax.legend(handles=[infarct_patch, mvo_patch])

    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig('vol.png')

import numpy as np

def tp_class1_and_fn_class0_within(
    y_true, y_probs, threshold=0.4, class0_index=0, class1_index=0
):
    """
    Among the true positives for class 1, count how many are false negatives for class 0,
    and also how often class 1 is predicted when:
    - only class 0 is present,
    - only class 1 is present,
    - both class 0 and class 1 are present.

    Parameters:
        y_true (np.ndarray): Ground truth labels, shape (n_samples, n_classes)
        y_probs (np.ndarray): Predicted probabilities, shape (n_samples, n_classes)
        threshold (float): Threshold to binarize predictions
        class0_index (int): Index of class 0
        class1_index (int): Index of class 1

    Returns:
        dict: A dictionary with counts of various conditions
    """
    y_pred = (np.array(y_probs) >= threshold).astype(int)
    y_true = np.array(y_true)

    print('prediction', y_pred[:10], flush=True)
    print('truth', y_true[:10], flush=True)

    # True Positives for class 1
    tp_class1_mask = (y_true[:, class1_index] == 1) & (y_pred[:, class1_index] == 1)
    tp_class1 = np.sum(tp_class1_mask)

    # False Negatives for class 0 within TP class 1
    fn_class0_within_tp1 = np.sum(
        (y_true[tp_class1_mask, class0_index] == 1) &
        (y_pred[tp_class1_mask, class0_index] == 0)
    )

    # Ground truth values
    gt0 = y_true[:, class0_index]
    gt1 = y_true[:, class1_index]

    # Predicted class 1
    pred1 = y_pred[:, class1_index]

    # Case 1: Only class 0 is 1 and class 1 is predicted
    case_only_0_and_pred1 = (gt0 == 1) & (gt1 == 0) & (pred1 == 1)
    count_only_0_and_pred1 = np.sum(case_only_0_and_pred1)

    # Case 2: Only class 1 is 1 and class 1 is predicted
    case_only_1_and_pred1 = (gt0 == 0) & (gt1 == 1) & (pred1 == 1)
    count_only_1_and_pred1 = np.sum(case_only_1_and_pred1)

    # Case 3: Both class 0 and class 1 are 1 and class 1 is predicted
    case_both_and_pred1 = (gt0 == 1) & (gt1 == 1) & (pred1 == 1)
    count_both_and_pred1 = np.sum(case_both_and_pred1)

    # Total positives in ground truth for class 1
    positives_class1 = int(np.sum(gt1 == 1))
    print('real IMH (class 1 positives):', positives_class1, flush=True)

    return {
        "tp_class1": int(tp_class1),
        "fn_class0_within_tp1": int(fn_class0_within_tp1),
        "count_only_class0_and_pred_class1": int(count_only_0_and_pred1),
        "count_only_class1_and_pred_class1": int(count_only_1_and_pred1),
        "count_both_classes_and_pred_class1": int(count_both_and_pred1),
        "positives_class1": positives_class1
    }




@torch.no_grad()
def evaluate(model: torch.nn.Module,
             criterion: torch.nn.Module,
             data_loader: Iterable,
             device: torch.device,
             metric_fn: torchmetrics.Metric,
             output_act: torch.nn.Module,
             target_dtype: torch.dtype = torch.long,
             use_amp: bool = True,
             ) -> Tuple[Dict[str, float], Dict[str, float]]:
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 50, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            if samples.ndim == 4:  # batch_size, n_drops, n_channels, n_frames
                logits_list = []
                for i in range(samples.size(1)):
                    logits = model(samples[:, i])
                    logits_list.append(logits)
                logits_list = torch.stack(logits_list, dim=1)
                outputs_list = output_act(logits_list)
                logits = logits_list.mean(dim=1)
                outputs = outputs_list.mean(dim=1)
            else:
                logits = model(samples)
                outputs = output_act(logits)
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                targets = targets.to(dtype=outputs.dtype)
            loss = criterion(logits, targets)

        outputs = misc.concat_all_gather(outputs)
        targets = misc.concat_all_gather(targets).to(dtype=target_dtype)
        metric_fn.update(outputs, targets)
        metric_logger.meters['loss'].update(loss.item(), n=samples.size(0))

    metric_logger.synchronize_between_processes()
    valid_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    metrics = metric_fn.compute()
    if isinstance(metrics, dict):  # MetricCollection
        metrics = {k: v.item() for k, v in metrics.items()}
    else:
        metrics = {metric_fn.__class__.__name__: metrics.item()}
    metric_str = "  ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
    metric_str = f"{metric_str} loss: {metric_logger.loss.global_avg:.3f}"
    # print(f"* {metric_str}")
    metric_fn.reset()

    return valid_stats, metrics
