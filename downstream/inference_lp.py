import os
import torch
import argparse
import sys
import numpy as np
from torch.utils.data import DataLoader
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from models import load_encoder
from linear_probe_utils import LinearClassifier, Finetune_Classifier, features_dataloader
from ecg_data_250Hz import ECGDataset
from augmentation import get_transforms_from_config, Compose, ToTensor
from sklearn.metrics import roc_auc_score
from scipy.special import expit, softmax


def parse_args():
    parser = argparse.ArgumentParser("ECG Prediction Script")
    parser.add_argument("--ckpt_pretrained_model", type=str, default = '../weights/ptbxl_stemi_non-stemi_normal_epoch100_cropa_4.pth')
    parser.add_argument("--ckpt_model", type=str, default = './weights/Classifier_all_ptbxl_stemi_non-stemi_normal_epoch100_cropa_4')
    parser.add_argument("--test_data", type=str,  default = "./test_data/All_ptbxl_stemi_non-stemi_normal_epoch100_cropa_4/")
    parser.add_argument("--output_probs", type=str, default="./predictions/LP_all_ptbxl_stemi_non-stemi_normal_epoch100_cropa_4/")
    parser.add_argument("--embeddings_dir", type=str, default="./embeddings_after_finetuning/LP_all_ptbxl_stemi_non-stemi_normal_epoch100_cropa_4/")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
    return parser.parse_args()

def main():
    fold = 4
    print("we do fold ,", fold, flush = True)
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    ckpt_dir = args.ckpt_pretrained_model


    # === Load encoder ===
    print("Loading encoder...")
    ckpt = torch.load(ckpt_dir, map_location=device)
    print(ckpt.keys(), flush=True)

    encoder, embed_dim = load_encoder(ckpt_dir=ckpt_dir)
    encoder = encoder.to(device)

    # === Load classifier ===
    print("Loading classifier weights...")
    classifier = LinearClassifier(embed_dim, 3)
    ckpt_dir = os.path.join(args.ckpt_model, f"weights_fold_{fold}.pth")

    checkpoint = torch.load(ckpt_dir, map_location=device)
    print(checkpoint.keys(), flush=True)
    # Load full model state dict from "full_model" key
    classifier.load_state_dict(checkpoint)
    classifier.eval()


    # === Load test data ===
    data = torch.load(os.path.join(args.test_data, f"input_fold_{fold}.pt"), weights_only=False)
    ID_test = data['id_test']
    waves_test = data['ecgs_test']
    labels_test = data['mvo_test']

           
    print("Test ECGs shape (after selecting 8 leads):", waves_test.shape)

    # === Define test transforms ===
    eval_transforms = [
        {"highpass_filter": {"fs": 250, "cutoff": 0.67}},
        {"lowpass_filter": {"fs": 250, "cutoff": 40}},
    ]
    test_transforms = Compose(get_transforms_from_config(eval_transforms) + [ToTensor()])

    # === Create DataLoader ===
    test_dataset = ECGDataset(waves_test, labels_test, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader_linear = features_dataloader(
            encoder, test_loader, batch_size=args.batch_size, shuffle=False, device=device
        )


    gt = []
    features = []

    with torch.no_grad():
        for wave, target in test_loader:
            print(wave.shape, flush = True)
            repr = encoder.representation(wave.to(device))  # (bs, dim)
            features.append(repr.cpu())
            gt.append(target[:, 1])

    # === Save embeddings AND their corresponding labels ===
    os.makedirs(args.embeddings_dir, exist_ok=True)  # ensure directory exists
    emb_save_path = os.path.join(args.embeddings_dir,  f"embeddings_fold_{fold}.npz")
   # np.savez(emb_save_path, embeddings=features, labels=gt)
   # print(f"Saved embeddings and labels to {emb_save_path}", flush=True)

    # === Inference ===
    print("Running inference...", flush = True)
    all_probs = []

    with torch.no_grad():
        for x, _ in test_loader_linear:
        #    x = x.to(device)
            logits = classifier(x)
            probs = expit(logits.cpu().numpy())
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    print("Predictions shape:", all_probs.shape)
    auc = roc_auc_score(labels_test, all_probs)
    auc_mvo = roc_auc_score(labels_test[:, :1], all_probs[:,:1])
    auc_imh = roc_auc_score(labels_test[:, 1:2], all_probs[:,1:2])

    print(f"AUC: {auc:.4f}", f"AUC MVO: {auc_mvo:.4f}",  f"AUC IMH: {auc_imh:.4f}", flush = True)


    # === Save ===
    os.makedirs(args.output_probs, exist_ok=True)  # ensure directory exists
    output_file = os.path.join(args.output_probs, f"predictions_fold_{fold}.npz")
    np.savez(output_file, probs=all_probs, labels=labels_test)
    print(f"Saved predicted probabilities to {output_file}", flush = True)
 

if __name__ == "__main__":
    main()
