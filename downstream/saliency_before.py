# %%
#! %load_ext autoreload
#! %autoreload 2
# %%
import os
import torch
import dataclasses
import captum
import sys
import numpy as np
import tqdm
from torch.utils.data import DataLoader
current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)
from models import load_encoder
from linear_probe_utils import LinearClassifier, Finetune_Classifier
from ecg_data_250Hz import ECGDataset
from augmentation import get_transforms_from_config, Compose, ToTensor
from sklearn.metrics import roc_auc_score
from scipy.special import expit, softmax
import matplotlib.pyplot as plt
import plot_attributions
import wfdb.processing
import waveforms


import neurokit2
target = 'output'
    ### compute the saliency maps for each ecg with label MVO- in the test set
def get_mean_signals(waves_test, labels_test, label = 'MVO'):
    n = 0
    all_waves = []
    all_contribs = []
    # x.requires_grad = True
    for x,y in zip(waves_test, labels_test):
        if label == 'MVO':
            target = 0
        elif label == 'IMH':
            target = 1
        else:
            target = 2
        attribution = saliency.attribute(torch.tensor(x).unsqueeze(0), target=target, abs=False)[0]
                
        rpeaks_left0 =  neurokit2.ecg_findpeaks(x[0], 250)["ECG_R_Peaks"]
        rpeaks_left = neurokit2.ecg_findpeaks(x[1], 250)["ECG_R_Peaks"]
        # rpeaks_right = wfdb.processing.xqrs_detect(x[-2], 250)
        rpeaks_right0 = neurokit2.ecg_findpeaks(x[-1], 250)["ECG_R_Peaks"]
        rpeaks_right = neurokit2.ecg_findpeaks(x[-2], 250)["ECG_R_Peaks"]
        # rpeaks_right = wfdb.processing.xqrs_detect(x[-2], 250)

        if not (len(rpeaks_left0) and len(rpeaks_left) and len(rpeaks_right0) and len(rpeaks_right)):
            n += 1
            print("not enough peaks", n)
            continue 
        comp = wfdb.processing.compare_annotations(rpeaks_left, rpeaks_left0, 20)
        matched = comp.matched_ref_sample
        # print(matched, rpeaks_right)
        comp = wfdb.processing.compare_annotations(rpeaks_right, rpeaks_right0, 20)
        matched_right = comp.matched_ref_sample

        if len(matched) > 1 and len(matched_right) > 1:
            waves_left = waveforms.extract_normalized_waveforms(x[:2].T, matched)[0].mean(axis=0)
            waves_right = waveforms.extract_normalized_waveforms(x[2:].T, matched_right)[0].mean(axis=0)
            waves = np.concatenate((waves_left, waves_right), axis=-1)
            # waves_left = [waveforms.extract_normalized_waveforms(signal, matched)[0] for signal in x[:2]]
            # waves_right = [waveforms.extract_normalized_waveforms(signal, rpeaks_right)[0] for signal in x[2:]]
            # waves_right = [waveforms.extract_normalized_waveforms(signal, rpeaks_right)[0] for signal in x[2:]]
            attribs_left = waveforms.extract_normalized_waveforms(attribution[:2].T, matched)[0].mean(axis=0)
            attribs_right = waveforms.extract_normalized_waveforms(attribution[2:].T, matched_right)[0].mean(axis=0)
            attribs = np.concatenate((attribs_left, attribs_right), axis=-1)

            all_waves.append(waves)
            all_contribs.append(attribs)
        else: 
            n += 1
            print('failed', n)

    return np.stack(all_waves), np.stack(all_contribs)

# %%

# def parse_args():
#     parser = argparse.ArgumentParser("ECG Prediction Script")
#     parser.add_argument("--ckpt_model", type=str, default = './weights/FT_all_ptbxl_stemi_non-stemi_normal_epoch100_cropa_4/')
#     parser.add_argument("--test_data", type=str,  default = "./test_data/FT_all_ptbxl_stemi_non-stemi_normal_epoch100_cropa_4/")
#     parser.add_argument("--output_probs", type=str, default="./predictions/FT_all_ptbxl_stemi_non-stemi_normal_epoch100_cropa_4/")
#     parser.add_argument("--embeddings_dir", type=str, default="./embeddings_after_finetuning/FT_all_ptbxl_stemi_non-stemi_normal_epoch100_cropa_4/")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference")
#     return parser.parse_args()

@dataclasses.dataclass
class Arguments:
   # ckpt_model = './weights/FT_0-12_ptbxl_stemi_non-stemi_normal_epoch100_cropa_5/'
   # test_data=  "./test_data/data_all_ptbxl_stemi_non-stemi_normal_epoch100_cropa_5_0-12/"
    ckpt_model = './weights/FT_before_ptbxl_stemi_non-stemi_normal_epoch100_cropa_5/'
    test_data=  "./test_data/data_all_ptbxl_stemi_non-stemi_normal_epoch100_cropa_5_before/"
    output_probs ="./predictions/FT_all_ptbxl_stemi_non-stemi_normal_epoch100_cropa_4/"
    embeddings_dir="./embeddings_after_finetuning/FT_all_ptbxl_stemi_non-stemi_normal_epoch100_cropa_4/"
    batch_size = 32

"""---------------------parameters to choose-------------------"""
saliency = 'gt'
"""----------------------------------------------------"""
All_waves = []
All_labels = []
All_preds = []
All_ids = []

# def main():
for fold in range(1,6):
    args = Arguments()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    ckpt_dir = os.path.join(args.ckpt_model, f"weights_fold_{fold}.pth")
    # === Load encoder ===
    print("Loading encoder...")
    ckpt = torch.load(ckpt_dir, map_location=device)
    print(ckpt.keys(), flush=True)
   
    encoder, embed_dim = load_encoder(ckpt_dir=ckpt_dir, device=device)
    encoder = encoder.to(device)

    # === Load classifier ===
    print("Loading classifier weights...")
    classifier = LinearClassifier(embed_dim, 3)
    model = Finetune_Classifier(encoder, classifier)

    # Load full model state dict from "full_model" key
    model.load_state_dict(ckpt["full_model"])
    model.to(device)
    model.eval()
    
    # === Load test data ===
    data = torch.load(os.path.join(args.test_data, f"input_fold_{fold}.pt"))
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

    
    saliency = captum.attr.Saliency(model)
    gradcam = captum.attr.LayerGradCam(model, model.encoder.W_P)
    ig = captum.attr.IntegratedGradients(model, False)
    x, y = test_dataset[1]
    # x.requires_grad = True
    attribution = saliency.attribute(x.unsqueeze(0), target=0, abs=False)[0]
    # attribution = gradcam.attribute(x.unsqueeze(0), target=0)[0]
    # attribution = ig.attribute(x.unsqueeze(0), target=0)[0]
    fig, axarr = plot_attributions.plot_attribution(x, attribution.detach().numpy(), fs=250, cmap="bwr")
    channel_names = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    for i, ax in enumerate(axarr.T.ravel()):
        plot_attributions.add_channel_label(ax, channel_names[i])
    
    idx = np.random.randint(len(test_dataset))
    saliency = captum.attr.Saliency(model)
    x, y = test_dataset[idx]
    attribution = saliency.attribute(x.unsqueeze(0), target=0, abs=False)[0]

    rpeaks_left0 = wfdb.processing.xqrs_detect(x[0].numpy(), 250)
    rpeaks_left = wfdb.processing.xqrs_detect(x[1].numpy(), 250)
    rpeaks_right = wfdb.processing.xqrs_detect(x[-2].numpy(), 250)

    waves_left = [waveforms.extract_normalized_waveforms(signal, rpeaks_left)[0] for signal in x[:2]]
    waves_right = [waveforms.extract_normalized_waveforms(signal, rpeaks_right)[0] for signal in x[2:]]
    print(attribution.min(), attribution.max())
    attribs_left = [waveforms.extract_normalized_waveforms(attr, rpeaks_left)[0] for attr in attribution[:2]]
    plt.imshow(attribs_left[0].mean(axis=0, keepdims=True), aspect="auto")
    fig, axarr = plot_attributions.plot_attribution(x, attribution.detach().numpy(), fs=250, cmap="bwr", vscale=(-0.03, 0.03))
    channel_names = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]
    for i, ax in enumerate(axarr.T.ravel()):
        plot_attributions.add_channel_label(ax, channel_names[i])
    t = axarr[0, 0].lines[0].get_xdata()
    plt.suptitle(y.numpy())
    axarr[0, 0].plot(t[rpeaks_left], x[0, rpeaks_left], "x")
    axarr[-2, 1].plot(t[rpeaks_right], x[-2, rpeaks_right], "x")
    attribs_right = [waveforms.extract_normalized_waveforms(attr, rpeaks_right)[0] for attr in attribution[2:]]

    
    comp = wfdb.processing.compare_annotations(rpeaks_left, rpeaks_left0, 20)
    comp.matched_ref_sample

    
    for wave in waves_left[1]:
        plt.plot(wave, alpha=0.8)
    plt.plot(waves_left[1].mean(axis=0), color="k", lw=2)
    
    # === Inference ===
    print("Running inference...", flush = True)
    all_probs = []
    with torch.no_grad():
        for x, _ in tqdm.tqdm(test_loader):
            x = x.to(device)
            logits = model(x)
            probs = expit(logits.cpu().numpy())
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    print("Predictions shape:", all_probs.shape)
    auc = roc_auc_score(labels_test, all_probs)
    auc_mvo = roc_auc_score(labels_test[:, :1], all_probs[:,:1])
    auc_imh = roc_auc_score(labels_test[:, 1:2], all_probs[:,1:2])
    preds = np.zeros_like(all_probs)
    preds[all_probs>0.5] = 1 
    print(f"AUC: {auc:.4f}", f"AUC MVO: {auc_mvo:.4f}",  f"AUC IMH: {auc_imh:.4f}", flush = True)
   

    # === Save ===
    os.makedirs(args.output_probs, exist_ok=True)  # ensure directory exists
    output_file = os.path.join(args.output_probs, f"predictions_fold_{fold}.npz")
    np.savez(output_file, probs=all_probs, labels=labels_test)
    print(f"Saved predicted probabilities to {output_file}", flush = True)
   
    All_waves.append(waves_test)
    All_labels.append(labels_test)
    All_preds.append(preds)
    All_ids.append(ID_test)


#%%
waves_test = np.concatenate(All_waves)    
labels_test = np.concatenate(All_labels)
preds = np.concatenate(All_preds)
patients = np.concatenate(All_ids)
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
sal = 'pred'
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.labelsize': 22,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 22,
    'figure.figsize': (20, 8),
    'text.usetex': False
})

# Define custom colors: petrol, white, darkred
colors = ["#006666", "white", "darkred"]  # petrol ~ #006666
#mask = (labels_test[:, 0] == 1) | (labels_test[:, 1] == 1)
if sal == 'gt':
    mask = (labels_test[:, 0] == 1) | (labels_test[:, 1] == 1)
else:
    mask = (preds[:, 0] == 1) | (preds[:, 1] == 1)

# Apply mask to only MVO positive patients
waves_test = waves_test[mask]
if saliency == 'gt':
    labels_masked = labels_test[mask]
    mean_hb, mean_sal= get_mean_signals(waves_test, labels_masked, label = 'MVO')
if sal == 'pred':
    preds_masked = preds[mask]
    mean_hb, mean_sal= get_mean_signals(waves_test, preds_masked, label = 'MVO')
#%%
# Create custom colormap
custom_cmap = LinearSegmentedColormap.from_list("petrol_white_darkred", colors)
fig, axarr = plot_attributions.plot_attribution(np.mean(mean_hb, axis=0).T, mean_sal.mean(axis=0).T, cmap=custom_cmap, vscale=(-0.003, 0.003),ncols=4,
                                                gridspec_kw={"hspace": 0.09, "wspace": 0.09, "top": 0.92}, sharey=True)
for ax, name in zip(axarr.T.ravel(), channel_names):
    plot_attributions.add_channel_label(ax, name)
    ax.set_xticks([])
    ax.set_yticks([])
    # Make box (spines) thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)   # change 2 to a larger value if you want even thicker lines

plt.suptitle(f'MVO 12-36h  {sal}', fontsize=24)

#%%
custom_cmap = LinearSegmentedColormap.from_list("petrol_white_darkred", colors)
fig, axarr = plot_attributions.plot_attribution(np.mean(mean_hb, axis=0).T, mean_sal.mean(axis=0).T, cmap=custom_cmap, vscale=(-0.004, 0.004),ncols=4,
                                                gridspec_kw={"hspace": 0.09, "wspace": 0.09, "top": 0.92}, sharey=True)
for ax, name in zip(axarr.T.ravel(), channel_names):
    plot_attributions.add_channel_label(ax, name)
    ax.set_xticks([])
    ax.set_yticks([])
    # Make box (spines) thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)   # change 2 to a larger value if you want even thicker lines
plt.suptitle(f'MVO$^+$/IMH$^+$ before', fontsize=24)
plt.savefig(f'/Users/nadjagruber/Documents/MVO_Project_paper/plots/MVO_IMH_mean_saliency_before.svg')



#%%

# Apply mask to only MVO positive patients
if sal == 'gt':
    labels_masked = labels_test[mask]
    mean_hb, mean_sal= get_mean_signals(waves_test, labels_masked, label = 'IMH')
if sal == 'pred':
    preds_masked = preds[mask]
    mean_hb, mean_sal= get_mean_signals(waves_test, preds_masked, label = 'IMH')

fig, axarr = plot_attributions.plot_attribution(np.mean(mean_hb, axis=0).T, mean_sal.mean(axis=0).T, cmap=custom_cmap, vscale=(-0.007, 0.007),ncols=4,
                                                gridspec_kw={"hspace": 0.09, "wspace": 0.09, "top": 0.92}, sharey=True)
for ax, name in zip(axarr.T.ravel(), channel_names):
    plot_attributions.add_channel_label(ax, name)
    ax.set_xticks([])
    ax.set_yticks([])
    # Make box (spines) thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)   # change 2 to a larger value if you want even thicker lines

#plt.suptitle(f'IMH 12-36h  {sal}')
#plt.savefig(f'IMH 12-36 {fold} target {target}.svg')
#%%
patients = np.concatenate(All_ids)
pat_masked = patients[mask]
idx = 3
hb, sal= get_mean_signals(waves_test[idx:idx+1], preds[idx:idx+1], label = 'MVO')
fig, axarr = plot_attributions.plot_attribution(np.mean(hb, axis=0).T, sal.mean(axis=0).T, cmap=custom_cmap, vscale=(-0.01, 0.01),ncols=4,
                                                gridspec_kw={"hspace": 0.09, "wspace": 0.09, "top": 0.92}, sharey=True)
for ax, name in zip(axarr.T.ravel(), channel_names):
    plot_attributions.add_channel_label(ax, name)
    ax.set_xticks([])
    ax.set_yticks([])
    # Make box (spines) thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)   # change 2 to a larger value if you want even thicker lines
    plt.suptitle(f'Patient {pat_masked[idx]}')

#%%
hb, sal= get_mean_signals(waves_test[idx:idx+1], preds[idx:idx+1], label = 'MVO')
fig, axarr = plot_attributions.plot_attribution(np.mean(hb, axis=0).T, sal.mean(axis=0).T, cmap=custom_cmap, vscale=(-0.01, 0.01),ncols=4,
                                                gridspec_kw={"hspace": 0.09, "wspace": 0.09, "top": 0.92}, sharey=True)
for ax, name in zip(axarr.T.ravel(), channel_names):
    plot_attributions.add_channel_label(ax, name)
    ax.set_xticks([])
    ax.set_yticks([])
    # Make box (spines) thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)   # change 2 to a larger value if you want even thicker lines
    plt.suptitle(f'MVO$^+$/IMH$^+$ anterior infarction', fontsize=24)
    plt.savefig(f'/Users/nadjagruber/Documents/MVO_Project_paper/plots/MVO_IMH_example_anterior_saliency.svg')

# %%
idx = 63
hb, sal= get_mean_signals(waves_test[idx:idx+1], preds[idx:idx+1], label = 'MVO')
fig, axarr = plot_attributions.plot_attribution(np.mean(hb, axis=0).T, sal.mean(axis=0).T, cmap=custom_cmap, vscale=(-0.007, 0.007),ncols=4,
                                                gridspec_kw={"hspace": 0.09, "wspace": 0.09, "top": 0.92}, sharey=True)
for ax, name in zip(axarr.T.ravel(), channel_names):
    plot_attributions.add_channel_label(ax, name)
    ax.set_xticks([])
    ax.set_yticks([])
    # Make box (spines) thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)   # change 2 to a larger value if you want even thicker lines
    plt.suptitle(f'Patient {pat_masked[idx]}')

#%%
hb, sal= get_mean_signals(waves_test[idx:idx+1], preds[idx:idx+1], label = 'MVO')
fig, axarr = plot_attributions.plot_attribution(np.mean(hb, axis=0).T, sal.mean(axis=0).T, cmap=custom_cmap, vscale=(-0.01, 0.01),ncols=4,
                                                gridspec_kw={"hspace": 0.09, "wspace": 0.09, "top": 0.92}, sharey=True)
for ax, name in zip(axarr.T.ravel(), channel_names):
    plot_attributions.add_channel_label(ax, name)
    ax.set_xticks([])
    ax.set_yticks([])
    # Make box (spines) thicker
    for spine in ax.spines.values():
        spine.set_linewidth(2)   # change 2 to a larger value if you want even thicker lines
    plt.suptitle(f'MVO$^+$/IMH$^+$ anterior infarction before', fontsize=24)
    plt.savefig(f'/Users/nadjagruber/Documents/MVO_Project_paper/plots/MVO_IMH_example_anterior_saliency_before.svg')

# %%
