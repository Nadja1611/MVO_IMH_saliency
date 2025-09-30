import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch


def filter_multilabel_stemi_vs_nonstemi(waves, labels):
    """
    Filters samples with MI = 1 and returns multilabels:
      - [MI, STTC]
        e.g., Non-STEMI → [1, 0]
              STEMI     → [1, 1]

    Excludes all other classes (e.g., Normal, CD, HYP).

    Args:
        waves (torch.Tensor or np.ndarray): ECG data of shape (N, ...)
        labels (torch.Tensor or np.ndarray): One-hot encoded labels of shape (N, num_classes)

    Returns:
        filtered_waves (same type as input): ECG samples with only MI cases
        new_labels (torch.Tensor): Labels of shape (N_filtered, 2)
    """
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    mi = labels[:, 2]
    sttc = labels[:, 4]

    # Select MI cases only (with or without STTC)
    selected = (mi == 1)

    filtered_waves = waves[selected] if isinstance(waves, torch.Tensor) else waves[selected.numpy()]
    new_labels = torch.stack([mi[selected], sttc[selected]], dim=1)  # MI always 1, STTC 0 or 1

    return filtered_waves, new_labels


def filter_multilabel_stemi_nonstemi_normal(waves, labels):
    """
    Filters samples to include only those where:
      - NORMAL = 1
      - MI = 1 (with or without STTC)

    Returns multilabels:
      - [NORMAL, MI, STTC]
        e.g., Normal → [1, 0, 0]
              Non-STEMI → [0, 1, 0]
              STEMI → [0, 1, 1]

    Args:
        waves (torch.Tensor or np.ndarray): ECG data of shape (N, ...)
        labels (torch.Tensor or np.ndarray): One-hot encoded labels of shape (N, num_classes)

    Returns:
        filtered_waves (same type as input): ECG samples with only selected multilabel classes
        new_labels (torch.Tensor): Labels of shape (N_filtered, 3)
    """
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    normal = labels[:, 3]
    mi = labels[:, 2]
    sttc = labels[:, 4]

    # Keep only Normal or MI cases (with/without STTC)
    selected = (normal == 1) | (mi == 1)

    filtered_waves = waves[selected] if isinstance(waves, torch.Tensor) else waves[selected.numpy()]
    new_labels = torch.stack([normal[selected], mi[selected], sttc[selected]], dim=1)

    return filtered_waves, new_labels

def filter_mi_and_whatever(waves, labels, cd_idx=3, hypertrophy_idx=5):
    """
    Keeps samples where MI (index 2) is present (regardless of other labels),
    and returns only MI, STTC, CD, hypertrophy columns.

    Args:
        waves (torch.Tensor or np.ndarray): ECG data of shape (N, ...)
        labels (torch.Tensor): One-hot encoded labels of shape (N, num_classes)
        cd_idx (int): Index for CD label column
        hypertrophy_idx (int): Index for hypertrophy label column

    Returns:
        filtered_waves (same type as input): ECG samples with MI present
        new_labels (torch.Tensor): Reduced labels of shape (N_filtered, 4)
    """

    """
    Keep only samples where MI == 1,
    keep all label columns as is.

    Args:
        waves (np.ndarray or torch.Tensor): ECG data, shape (N, ...)
        labels (np.ndarray or torch.Tensor): labels with shape (N,5)

    Returns:
        filtered_waves: filtered ECG samples
        filtered_labels: labels for samples where MI == 1 (shape (N_filtered, 5))
    """
    # Create mask where MI == 1
    if isinstance(labels, torch.Tensor):
        mask = labels[:, 2].bool()
    else:
        mask = labels[:, 2] == 1

    filtered_waves = waves[mask]
    filtered_labels = labels[mask]

    return filtered_waves, filtered_labels

def filter_mi_sttc_only(waves, labels):
    """
    Filters samples that have MI (index 2) or STTC (index 4) and reduces labels to [MI, STTC].

    Args:
        waves (torch.Tensor or np.ndarray): ECG data of shape (N, ...)
        labels (torch.Tensor): One-hot encoded labels of shape (N, num_classes)

    Returns:
        filtered_waves (same type as input): ECG samples with only MI/STTC labels
        new_labels (torch.Tensor): Reduced labels of shape (N_filtered, 2)
    """
    mi = labels[:, 2]
    sttc = labels[:, 4]
    mask = (mi + sttc) > 1  # keep if MI and STTC is present

    filtered_waves = waves[mask]
    new_labels = torch.stack([
        torch.tensor(mi[mask]), 
        torch.tensor(sttc[mask])
    ], dim=1)
    return filtered_waves, new_labels

def filter_mi_sttc_normal_only(waves, labels):
    """
    Filters samples that have MI (index 2), NORMAL (index 3), or STTC (index 4)
    and reduces labels to [MI, NORMAL, STTC].

    Args:
        waves (torch.Tensor or np.ndarray): ECG data of shape (N, ...)
        labels (torch.Tensor or np.ndarray): One-hot encoded labels of shape (N, num_classes)

    Returns:
        filtered_waves (same type as input): ECG samples with only MI/NORMAL/STTC labels
        new_labels (torch.Tensor): Reduced labels of shape (N_filtered, 3)
    """
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    mi = labels[:, 2]
    normal = labels[:, 3]
    cd = labels[:,0]
    hyp = labels[:,1]
    sttc = labels[:, 4]
    
    mask = (mi + normal + sttc) > 0  # keep if any of MI, NORMAL, or STTC is present

    filtered_waves = waves[mask] if isinstance(waves, torch.Tensor) else waves[mask.numpy()]
    new_labels = torch.stack([mi[mask], normal[mask], sttc[mask]], dim=1)
    return filtered_waves, new_labels



def filter_stemi_vs_nonstemi(waves, labels):
    """
    Filters samples to include only MI cases and distinguishes:
      - Non-STEMI: MI = 1 and STTC = 0 → [1, 0]
      - STEMI:    MI = 1 and STTC = 1 → [0, 1]

    Excludes all other classes. Returns binary labels [Non-STEMI, STEMI].

    Args:
        waves (torch.Tensor or np.ndarray): ECG data of shape (N, ...)
        labels (torch.Tensor or np.ndarray): One-hot encoded labels of shape (N, num_classes)

    Returns:
        filtered_waves (same type as input): ECG samples with only MI cases
        new_labels (torch.Tensor): Labels of shape (N_filtered, 2)
    """
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    mi = labels[:, 2]
    sttc = labels[:, 4]

    # Select only samples with MI = 1
    mi_cases = (mi == 1)

    # Within MI cases, distinguish STEMI vs Non-STEMI
    non_stemi = mi_cases & (sttc == 0)
    stemi = mi_cases & (sttc == 1)
    selected = non_stemi | stemi

    filtered_waves = waves[selected] if isinstance(waves, torch.Tensor) else waves[selected.numpy()]
    new_labels = torch.stack([non_stemi[selected], stemi[selected]], dim=1)

    return filtered_waves, new_labels



def filter_ami_stemi_normal_only(waves, labels):
    """
    Filters samples to include:
      - MI = 1 (with or without STTC)
      - NORMAL = 1
    Excludes all other labels. Returns reduced labels [MI, NORMAL, STTC].

    Args:
        waves (torch.Tensor or np.ndarray): ECG data of shape (N, ...)
        labels (torch.Tensor or np.ndarray): One-hot encoded labels of shape (N, num_classes)

    Returns:
        filtered_waves (same type as input): ECG samples with only selected labels
        new_labels (torch.Tensor): Labels of shape (N_filtered, 3)
    """
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels)

    mi = labels[:, 2]
    normal = labels[:, 3]
    sttc = labels[:, 4]

    # Make sure no other class is active (e.g., CD, HYP, etc.)
    only_these = mi + normal + sttc == 1  # exactly one of them is active
    condition = ((mi == 1) | (normal == 1)) & only_these

    filtered_waves = waves[condition] if isinstance(waves, torch.Tensor) else waves[condition.numpy()]
    new_labels = torch.stack([mi[condition], normal[condition], sttc[condition]], dim=1)

    return filtered_waves, new_labels
