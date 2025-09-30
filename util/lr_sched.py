# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math



def get_layerwise_lr_decay_params(model, base_lr, layer_decay=0.6, weight_decay=0.05):
    """
    Prepare parameter groups for optimizer with layer-wise learning rate decay.

    Parameters:
    - model: your FinetuningClassifier model
    - base_lr: learning rate for the final layer (head)
    - layer_decay: decay factor for earlier layers (e.g., 0.8)
    - weight_decay: weight decay for optimizer

    Returns:
    - A list of parameter groups with layer-specific learning rates
    """
    # Collect the encoder blocks
    encoder_blocks = list(model.encoder.encoder_blocks.blocks)
    num_layers = len(encoder_blocks)
    
    param_groups = []

    # Handle each encoder layer with decayed learning rate
    for layer_idx, block in enumerate(encoder_blocks):
        scale = layer_decay ** (num_layers - 1 - layer_idx)  # later layers have higher LR
        lr = base_lr * scale
        param_groups.append({
            "params": block.parameters(),
            "lr": lr,
            "weight_decay": weight_decay,
        })

    # Add normalization layers (often added separately without weight decay)
    if hasattr(model.encoder, "norm"):
        param_groups.append({
            "params": model.encoder.norm.parameters(),
            "lr": base_lr * (layer_decay ** num_layers),
            "weight_decay": 0.0,
        })

    # Add classifier head (final layer) — use base LR
    classifier_params = []
    for name, param in model.named_parameters():
        if "encoder" not in name:
            classifier_params.append(param)

    param_groups.append({
        "params": classifier_params,
        "lr": base_lr,
        "weight_decay": weight_decay,
    })

    return param_groups



def adjust_learning_rate(optimizer, epoch, config):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if config['lr'] is None:
        config['lr'] = config['blr'] * config['dataloader']['batch_size'] / 256

    if epoch < config['warmup_epochs']:
        lr = config['lr'] * epoch / config['warmup_epochs']
    else:
        lr = config['min_lr'] + (config['lr'] - config['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr



def adjust_learning_rate(optimizer, epoch, config):
    if config['lr'] is None:
        config['lr'] = config['blr'] * config['dataloader']['batch_size'] / 256

    if epoch < config['warmup_epochs']:
        base_lr = config['lr'] * epoch / config['warmup_epochs']
    else:
        base_lr = config['lr']
        #base_lr = config['min_lr'] + (config['lr'] - config['min_lr']) * 0.5 * \
         #   (1. + math.cos(math.pi * (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])))

    for param_group in optimizer.param_groups:
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = base_lr * scale

    return base_lr


def get_parameter_groups(model, layer_decay=0.75):
    """
    Assigns layer-wise learning rate decay to parameter groups.
    Assumes model has named modules like transformer blocks or CNN stages.
    """
    param_groups = []
    num_layers = 12  # Adjust this to match your model depth

    # Helper: assign a layer_id based on param name (adjust logic for your model)
    def get_layer_id(name):
        if name.startswith("patch_embed"):
            return 0
        elif name.startswith("blocks"):
            block_id = int(name.split('.')[1])
            return block_id + 1
        else:
            return num_layers  # head, norm, etc.

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        layer_id = get_layer_id(name)
        lr_scale = layer_decay ** (num_layers - layer_id - 1)
        param_groups.append({
            "params": [param],
            "lr_scale": lr_scale,
        })
    return param_groups
