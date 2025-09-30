import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import  mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,mean_squared_error, mean_absolute_error, r2_score, average_precision_score
from scipy.special import expit, softmax
from tqdm import tqdm
import copy
import torch
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import os

# Precompute the features from the encoder and store them
def precompute_features(encoder, loader, device):
    encoder.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        print("Precomputing features...")
        for wave, label in tqdm(loader):
            bs, _, _ = wave.shape
            wave = wave.to(device)
            feature = encoder.representation(wave)  # (bs,c*50,384)
            all_features.append(feature.cpu())
            all_labels.append(label)

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    return all_features, all_labels


def features_dataloader(encoder, loader, batch_size=32, shuffle=True, device="cpu"):
    features, labels = precompute_features(encoder, loader, device=device)
    dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4
    )

    return dataloader


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, apply_bn=False):
        super(LinearClassifier, self).__init__()
        self.apply_bn = apply_bn
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=1e-6)
        self.fc = nn.Linear(input_dim, num_labels)

    def forward(self, x):
        if self.apply_bn:
            x = self.bn(x)

        x = self.fc(x)
        return x





class FinetuningPTBXL(nn.Module):
    def __init__(self, encoder, encoder_dim, num_labels, device="cpu", apply_bn=False, dropout_rate=0.5):
        """
        Args:
            encoder (nn.Module): Pretrained model encoder (e.g., a CNN, Transformer, etc.)
            encoder_dim (int): Dimension of the encoder output (i.e., the feature size).
            num_labels (int): Number of output labels for the classification task.
            device (str): Device to use, default is "cpu".
            apply_bn (bool): Whether to apply batch normalization or not.
            dropout_rate (float): Dropout rate to apply after encoder output, default is 0.5.
        """
        super(FinetuningPTBXL, self).__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.fc = LinearClassifier(encoder_dim, num_labels, apply_bn=apply_bn)
                
        #self.fc = MLPClassifier(encoder_dim, hidden_dim=256, num_labels=num_labels, apply_bn=apply_bn, dropout=dropout_rate)

        # Dropout layer with specified rate
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout rate specified during initialization
        
    def forward(self, x):
        """
        Forward pass of the model. It passes the input through the encoder and then through the classifier.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output logits for classification.
        """
        bs, _, _ = x.shape
        
        # Get the encoder's feature representation
        x = self.encoder.representation(x)
        
        # Apply dropout after the encoder output
        x = self.dropout(x)
        
        # Classifier output
        x = self.fc(x)
        
        return x




        
class SimpleLinearRegression(nn.Module):
    def __init__(self, input_dim=384):
        super(SimpleLinearRegression, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x)



class Finetune_Regression(nn.Module):
    def __init__(self, encoder, linear_head, dropout_rate=0.5):
        super(Finetune_Regression, self).__init__()
        self.encoder = encoder
        self.linear_head = linear_head
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        # Same shape as classifier: (B, C, L)
        x = self.encoder.representation(x)  # This is KEY
        x = self.dropout(x)
        x = self.linear_head(x)
        return x




class Finetune_Classifier(nn.Module):
    def __init__(self, encoder, classifier_head, dropout_rate=0.5):
        super(Finetune_Classifier, self).__init__()
        self.encoder = encoder
        self.classifier_head = classifier_head
        self.dropout = nn.Dropout(p=dropout_rate)
        
    def forward(self, x):
        # Same shape as regression: (B, C, L)
        x = self.encoder.representation(x)  # This is KEY
        x = self.dropout(x)
        x = self.classifier_head(x)
        return x

class Finetune_Classifier_timing(nn.Module):
    def __init__(self, encoder, classifier_head, classifier_head_time, dropout_rate=0.25):
        super(Finetune_Classifier_timing, self).__init__()
        self.encoder = encoder
        self.classifier_head = classifier_head
        self.classifier_head_time = classifier_head_time
        self.dropout = nn.Dropout(p=dropout_rate)
        print('dropi ', self.dropout,flush=True)
        
    def forward(self, x):
        # Same shape as regression: (B, C, L)
        x = self.encoder.representation(x)  # This is KEY
        x = self.dropout(x)
        z = torch.clone(x)
        x = self.classifier_head(x)
        y = self.classifier_head_time(z)
        return x, y


def train_multilabel(
    num_epochs,
    model,
    optimizer,
    criterion,
    scheduler,
    train_loader_linear,
    val_loader_linear,
    test_loader_linear,
    device,
    print_every=True,
    writer=None,
    classes = 3,
    best_model_path='best_model.pth'  # add a path to save the best model,

):
    iterations_per_epoch = len(train_loader_linear)
    max_auc = 0.0
    avg_auc_list, val_loss_list, train_loss_list = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
            batch_features = batch_features.to(device)
            if classes == 3:
                batch_labels = batch_labels[:,:3].to(device)
            if classes == 2:
                batch_labels = batch_labels[:,3:].to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch * iterations_per_epoch + minibatch)
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader_linear)
        train_loss_list.append(avg_train_loss)

        # Validation phase
        if val_loader_linear is not None:
            model.eval()
            val_loss = 0.0
            all_val_labels, all_val_outputs = [], []

            with torch.no_grad():
                for val_features, val_labels in val_loader_linear:
                    val_features = val_features.to(device)
                   # print('shape of the labels ', val_labels.shape, flush = True)
                    if classes == 3:
                        val_labels = val_labels[:,:3].to(device)
                    if classes == 2:
                        val_labels = val_labels[:,3:].to(device)
                    val_preds = model(val_features)
                    val_loss += criterion(val_preds, val_labels.float()).item()
                    all_val_labels.append(val_labels.cpu().numpy())
                    all_val_outputs.append(val_preds.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader_linear)
            all_val_labels = np.vstack(all_val_labels)
            all_val_outputs = np.vstack(all_val_outputs)
            val_loss_list.append(avg_val_loss)


            val_auc_scores = [
                roc_auc_score(all_val_labels[:, i], all_val_outputs[:, i])
                if np.unique(all_val_labels[:, i]).size > 1 else float("nan")
                for i in range(all_val_labels.shape[1])
            ]
            val_avg_auc = np.nanmean(val_auc_scores)

            # Save best model
            if val_avg_auc > max_auc:
                max_auc = val_avg_auc
                torch.save(model.state_dict(), best_model_path)

            if writer is not None:
                writer.add_scalar("Loss/Val", avg_val_loss, epoch)
                writer.add_scalar("AUC/Val", val_avg_auc, epoch)

        if print_every:
            print(f"Epoch({epoch}) Val AUC: {val_avg_auc:.4f}, Best Val AUC: {max_auc:.4f}")

    # Load best model for testing after all epochs
  #  linear_model.load_state_dict(torch.load(best_model_path))
 
    # === Evaluation on test set ===
    model.eval()
    all_labels = []
    all_outputs = []
    total_test_loss = 0.0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader_linear:
            batch_features = batch_features.to(device)
            if classes == 3:
                batch_labels = batch_labels[:,:3].to(device)
            if classes == 2:
                batch_labels = batch_labels[:,3:].to(device)
            outputs = model(batch_features)
            test_loss = criterion(outputs, batch_labels.float())
            total_test_loss += test_loss.item()
            all_labels.append(batch_labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)
    all_probs = expit(all_outputs)
    avg_test_loss = total_test_loss / len(test_loader_linear)

    # AUC
    auc_scores = [
        roc_auc_score(all_labels[:, i], all_outputs[:, i])
        if np.unique(all_labels[:, i]).size > 1 else float("nan")
        for i in range(all_labels.shape[1])
    ]
    avg_auc = np.nanmean(auc_scores)
    avg_auc_list.append(avg_auc)

    if avg_auc > max_auc:
        max_auc = avg_auc

    # Thresholded predictions
    predicted_labels = (all_probs >= 0.5).astype(int)

    # F1 scores
    per_class_f1 = f1_score(all_labels, predicted_labels, average=None)
    macro_f1 = f1_score(all_labels, predicted_labels, average="macro")

    # Accuracy
    per_class_acc = accuracy_score(all_labels, predicted_labels)
    classes = np.unique(all_labels)
    class_acc = []
    for cls in classes:
        mask = (all_labels == cls)
        correct = (predicted_labels[mask] == all_labels[mask]).sum()
        total = mask.sum()
        acc = correct / total
        class_acc.append(acc)
    macro_acc = np.mean(class_acc)

    # Precision / Recall
    precision_macro = precision_score(all_labels, predicted_labels, average='macro')
    recall_macro = recall_score(all_labels, predicted_labels, average='macro')
    precision_per_class = precision_score(all_labels, predicted_labels, average=None)
    recall_per_class = recall_score(all_labels, predicted_labels, average=None)

    # AUPRC
    auprc_per_class = average_precision_score(all_labels, all_probs, average=None)
    auprc_macro = average_precision_score(all_labels, all_probs, average='macro')

    # TensorBoard logging
    if writer is not None:
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("AUC/Test", avg_auc, epoch)
        writer.add_scalar("F1/Macro", macro_f1, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

    if print_every:
        print(
            f"Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), "
            f"Macro F1: {macro_f1:.3f}, Per class F1: {per_class_f1.mean():.3f}"
        )
        for idx, auc in enumerate(auc_scores):
            print(f"Class {idx}: AUC = {auc:.4f}", flush=True)

    # Return test metrics only (or you can also return training info as needed)
    return (
        macro_acc,
        class_acc,
        avg_auc,
        auc_scores,
        macro_f1,
        per_class_f1,
        auprc_macro,
        auprc_per_class,
        recall_macro,
        recall_per_class,
        precision_macro,
        precision_per_class,
        all_probs,
        all_labels,
    )



def train_multilabel_with_training_data(
    num_epochs,
    linear_model,
    optimizer,
    criterion,
    scheduler,
    train_loader_linear,
    train_loader_linear_not_random,
    val_loader_linear,    # <- added
    test_loader_linear,
    device,
    print_every=True,
    writer=None
):
    iterations_per_epoch = len(train_loader_linear)
    max_auc = 0.0
    avg_auc_list, val_loss_list, train_loss_list = [], [], []
    # for saving training outputs
    training_outputs_epoch_end = []

    for epoch in range(num_epochs):
        linear_model.train()
        total_train_loss = 0.0

        for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = linear_model(batch_features)
            loss = criterion(outputs, batch_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch * iterations_per_epoch + minibatch)
            total_train_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}', flush=True)

        # Save final training predictions (unshuffled) after last epoch
        if epoch == num_epochs - 1:
            linear_model.eval()
            with torch.no_grad():
                outputs_list = []
                for batch_features, _ in train_loader_linear_not_random:
                    batch_features = batch_features.to(device)
                    outputs = linear_model(batch_features)
                    outputs_list.append(outputs.cpu().numpy())

                training_outputs_epoch_end = np.vstack(outputs_list)
                training_probs = expit(training_outputs_epoch_end)  # Apply sigmoid for probabilities
        # === Validation (just for TensorBoard logging) ===
        if val_loader_linear is not None and writer is not None:
            linear_model.eval()
            val_loss = 0.0
            all_val_labels, all_val_outputs = [], []

            with torch.no_grad():
                for val_features, val_labels in val_loader_linear:
                    val_features = val_features.to(device)
                    val_labels = val_labels.to(device)
                    val_preds = linear_model(val_features)
                    val_loss += criterion(val_preds, val_labels.float()).item()
                    all_val_labels.append(val_labels.cpu().numpy())
                    all_val_outputs.append(val_preds.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader_linear)
            val_loss_list.append(avg_val_loss)

            all_val_labels = np.vstack(all_val_labels)
            all_val_outputs = np.vstack(all_val_outputs)
            val_probs = expit(all_val_outputs)

            val_auc_scores = [
                roc_auc_score(all_val_labels[:, i], all_val_outputs[:, i])
                if np.unique(all_val_labels[:, i]).size > 1 else float("nan")
                for i in range(all_val_labels.shape[1])
            ]
            val_avg_auc = np.nanmean(val_auc_scores)

            writer.add_scalar("Loss/Val", avg_val_loss, epoch)
            writer.add_scalar("AUC/Val", val_avg_auc, epoch)

        # === Evaluation on test set ===
        linear_model.eval()
        all_labels = []
        all_outputs = []
        total_test_loss = 0.0

        with torch.no_grad():
            for batch_features, batch_labels in test_loader_linear:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                outputs = linear_model(batch_features)
                test_loss = criterion(outputs, batch_labels.float())
                total_test_loss += test_loss.item()
                all_labels.append(batch_labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)
        all_probs = expit(all_outputs)
        avg_test_loss = total_test_loss / len(test_loader_linear)

        # AUC
        auc_scores = [
            roc_auc_score(all_labels[:, i], all_outputs[:, i])
            if np.unique(all_labels[:, i]).size > 1 else float("nan")
            for i in range(all_labels.shape[1])
        ]
        avg_auc = np.nanmean(auc_scores)
        avg_auc_list.append(avg_auc)

        if avg_auc > max_auc:
            max_auc = avg_auc

        # Thresholded predictions
        predicted_labels = (all_probs >= 0.5).astype(int)

        # F1 scores
        per_class_f1 = f1_score(all_labels, predicted_labels, average=None)
        macro_f1 = f1_score(all_labels, predicted_labels, average="macro")

        # Accuracy
        per_class_acc = accuracy_score(all_labels, predicted_labels)
        classes = np.unique(all_labels)
        class_acc = []
        for cls in classes:
            mask = (all_labels == cls)
            correct = (predicted_labels[mask] == all_labels[mask]).sum()
            total = mask.sum()
            acc = correct / total
            class_acc.append(acc)
        macro_acc = np.mean(class_acc)

        # Precision / Recall
        precision_macro = precision_score(all_labels, predicted_labels, average='macro')
        recall_macro = recall_score(all_labels, predicted_labels, average='macro')
        precision_per_class = precision_score(all_labels, predicted_labels, average=None)
        recall_per_class = recall_score(all_labels, predicted_labels, average=None)

        # AUPRC
        auprc_per_class = average_precision_score(all_labels, all_probs, average=None)
        auprc_macro = average_precision_score(all_labels, all_probs, average='macro')

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("Loss/Train", avg_train_loss, epoch)
            writer.add_scalar("AUC/Test", avg_auc, epoch)
            writer.add_scalar("F1/Macro", macro_f1, epoch)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

        if print_every:
            print(
                f"Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), "
                f"Macro F1: {macro_f1:.3f}, Per class F1: {per_class_f1.mean():.3f}"
            )
            for idx, auc in enumerate(auc_scores):
                print(f"Class {idx}: AUC = {auc:.4f}", flush=True)

    return (
        macro_acc,
        class_acc,
        avg_auc,
        auc_scores,
        macro_f1,
        per_class_f1,
        auprc_macro,
        auprc_per_class,
        recall_macro,
        recall_per_class,
        precision_macro,
        precision_per_class,
        all_probs,
        all_labels, 
        training_probs
    )


def train_multilabel_with_validation(
    num_epochs,
    model,
    optimizer,
    criterion,
    scheduler,
    train_loader_linear,
    val_loader_linear,  # NEW
    test_loader_linear,
    device,
    print_every=True,
):
    iterations_per_epoch = len(train_loader_linear)
    max_auc = 0.0
    best_model_state = None  # NEW

    for epoch in range(num_epochs):
        model.train()
        for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch * iterations_per_epoch + minibatch)

        ### -------- VALIDATION PHASE -------- ###
        all_labels = []
        all_outputs = []
        model.eval()
        with torch.no_grad():
            for batch_features, batch_labels in val_loader_linear:
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                all_labels.append(batch_labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
                val_loss = criterion(outputs, batch_labels.float())
            val_loss_list.append(val_loss.item())

        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)
        all_probs = expit(all_outputs)

        auc_scores = [
            roc_auc_score(all_labels[:, i], all_outputs[:, i])
            if np.unique(all_labels[:, i]).size > 1
            else float("nan")
            for i in range(all_labels.shape[1])
        ]
        avg_auc = np.nanmean(auc_scores)
        
        if avg_auc > max_auc:
            max_auc = avg_auc
       #     best_model_state = copy.deepcopy(linear_model.state_dict())  # Save best weights

        if print_every:
            print(f"\nEpoch({epoch}) VALIDATION AUC: {avg_auc:.3f} (Best: {max_auc:.3f})", flush=True)
            for idx, auc in enumerate(auc_scores):
                print(f"  Class {idx}: AUC = {auc:.4f}", flush=True)

    ### -------- LOAD BEST MODEL -------- ###
   # if best_model_state is not None:
    #    model.load_state_dict(best_model_state)

    ### -------- FINAL TEST PREDICTION -------- ###
    all_labels = []
    all_outputs = []
    model.eval()
    with torch.no_grad():
        for batch_features, batch_labels in test_loader_linear:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            all_labels.append(batch_labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)
    all_probs = expit(all_outputs)

    predicted_labels = (all_probs >= 0.5).astype(int)

    auc_scores = [
        roc_auc_score(all_labels[:, i], all_outputs[:, i])
        if np.unique(all_labels[:, i]).size > 1
        else float("nan")
        for i in range(all_labels.shape[1])
    ]
    avg_auc = np.nanmean(auc_scores)
    per_class_f1 = f1_score(all_labels, predicted_labels, average=None)
    macro_f1 = f1_score(all_labels, predicted_labels, average="macro")

    precision_macro = precision_score(all_labels, predicted_labels, average='macro')
    recall_macro = recall_score(all_labels, predicted_labels, average='macro')
    precision_per_class = precision_score(all_labels, predicted_labels, average=None)
    recall_per_class = recall_score(all_labels, predicted_labels, average=None)
    auprc_per_class = average_precision_score(all_labels, all_probs, average=None)
    auprc_macro = average_precision_score(all_labels, all_probs, average='macro')

    # Compute class-wise accuracy
    classes = np.unique(all_labels)
    class_acc = []
    for cls in classes:
        mask = (all_labels == cls)
        correct = (predicted_labels[mask] == all_labels[mask]).sum()
        total = mask.sum()
        acc = correct / total
        class_acc.append(acc)
    macro_acc = np.mean(class_acc)

    return (
        macro_acc, class_acc, max_auc, avg_auc, auc_scores, macro_f1, per_class_f1,
        auprc_macro, auprc_per_class, recall_macro, recall_per_class,
        precision_macro, precision_per_class, all_probs, all_labels, model
    )



def train_regression(
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
):
    best_val_rmse = float("inf")
    best_val_pearson = -float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        linear_model.train()

        for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = linear_model(batch_features)
            loss = criterion(outputs, batch_labels.view(-1, 1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step(epoch * len(train_loader_linear) + minibatch)

        # Validation
        linear_model.eval()
        val_outputs = []
        val_labels = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader_linear:
                batch_features = batch_features.to(device)
                outputs = linear_model(batch_features)
                val_outputs.append(outputs.cpu().numpy())
                val_labels.append(batch_labels.cpu().numpy())

        val_outputs = np.hstack([output[:, 0].flatten() for output in val_outputs])
        val_labels = np.hstack([label.flatten() for label in val_labels])

        val_rmse = np.sqrt(mean_squared_error(val_labels, val_outputs))
        val_pearson_r, _ = pearsonr(val_labels, val_outputs)

       # if val_rmse < best_val_rmse:
        #    best_val_rmse = val_rmse
        #    best_model_state = copy.deepcopy(linear_model.state_dict())
        if val_pearson_r > best_val_pearson:
            best_val_pearson = val_pearson_r
            best_model_state = copy.deepcopy(linear_model.state_dict())

        if print_every:
            print(f"Epoch {epoch}: Validation RMSE: {val_rmse:.4f} PearsonL {val_pearson_r:.4f}")

    # Load best model and evaluate on test set
    linear_model.load_state_dict(best_model_state)
    linear_model.eval()
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader_linear:
            batch_features = batch_features.to(device)
            outputs = linear_model(batch_features)
           # print('predictions, ', outputs[:10], flush=True)
           # print('gt, ', batch_labels[:10], flush=True)
            all_outputs.append(outputs.cpu().numpy())
            all_labels.append(batch_labels.cpu().numpy())

    all_outputs = np.hstack([output[:, 0].flatten() for output in all_outputs])
    all_labels = np.hstack([label.flatten() for label in all_labels])

    mse = mean_squared_error(all_labels, all_outputs)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_labels, all_outputs)
    r2 = r2_score(all_labels, all_outputs)
    pearson_r, _ = pearsonr(all_labels, all_outputs)

    print(f"\nBest Model on Test Set => MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Pearson's r: {pearson_r:.4f}")
    linear_model.eval()
    return mse, rmse, mae, r2, pearson_r, all_outputs, all_labels


def train_multilabel_timing(
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
    writer=None,
    saliency=False,
    best_model_path='best_model.pth'  # add a path to save the best model
):
    iterations_per_epoch = len(train_loader_linear)
    max_auc = 0.0
    avg_auc_list, val_loss_list, train_loss_list = [], [], []

    for epoch in range(num_epochs):
        linear_model.train()
        total_train_loss = 0.0
        total_train_loss_timing = 0.0
        for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
            batch_features = batch_features.to(device)
            batch_labels_timing = batch_labels[:,3:].to(device)
            batch_labels = batch_labels[:,:3].to(device)
            outputs = linear_model(batch_features)[0]
            outputs_timing = linear_model(batch_features)[1]
           # print(' timings', outputs_timing[:20], batch_labels_timing[:20,:], flush = True)

            loss = criterion(outputs, batch_labels.float())
            loss_timing = criterion(outputs_timing, batch_labels_timing.float())
            

            optimizer.zero_grad()
            loss.backward()
            loss_timing.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch * iterations_per_epoch + minibatch)
            total_train_loss += loss.item()
            total_train_loss_timing += loss_timing.item()

        avg_train_loss = total_train_loss / len(train_loader_linear)
        avg_train_loss_timing = total_train_loss_timing / len(train_loader_linear)

        # Validation phase
        if val_loader_linear is not None:
            linear_model.eval()
            val_loss = 0.0
            all_val_labels, all_val_outputs = [], []

            with torch.no_grad():
                for val_features, val_labels in val_loader_linear:
                    val_features = val_features[:,:].to(device)
                    val_labels = val_labels[:,:3].to(device)
                    val_preds = linear_model(val_features)[0]
                    val_preds_timing = linear_model(val_features)[1]
                    val_loss += criterion(val_preds, val_labels.float()).item()
                    all_val_labels.append(val_labels.cpu().numpy())
                    all_val_outputs.append(val_preds.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader_linear)
            all_val_labels = np.vstack(all_val_labels)
            all_val_outputs = np.vstack(all_val_outputs)

            val_auc_scores = [
                roc_auc_score(all_val_labels[:, i], all_val_outputs[:, i])
                if np.unique(all_val_labels[:, i]).size > 1 else float("nan")
                for i in range(all_val_labels.shape[1])
            ]
            val_avg_auc = np.nanmean(val_auc_scores)

            # Save best model
            if val_avg_auc > max_auc:
                max_auc = val_avg_auc
                torch.save(linear_model.state_dict(), best_model_path)

            if writer is not None:
                writer.add_scalar("Loss/Val", avg_val_loss, epoch)
                writer.add_scalar("AUC/Val", val_avg_auc, epoch)

        if print_every:
            print(f"Epoch({epoch}) Val AUC: {val_avg_auc:.4f}, Best Val AUC: {max_auc:.4f}")

    # Load best model for testing after all epochs
  #  linear_model.load_state_dict(torch.load(best_model_path))
 
    # === Evaluation on test set ===
    linear_model.eval()
    all_labels = []
    all_outputs = []
    total_test_loss = 0.0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader_linear:
            batch_features = batch_features.to(device)
           # print('shape of input',batch_features.shape, flush = True)

            batch_labels = batch_labels[:,:3].to(device)
            outputs = linear_model(batch_features)[0]
            test_loss = criterion(outputs, batch_labels.float())
            total_test_loss += test_loss.item()
            all_labels.append(batch_labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_outputs = np.vstack(all_outputs)
    all_probs = expit(all_outputs)
    avg_test_loss = total_test_loss / len(test_loader_linear)

    # AUC
    auc_scores = [
        roc_auc_score(all_labels[:, i], all_outputs[:, i])
        if np.unique(all_labels[:, i]).size > 1 else float("nan")
        for i in range(all_labels.shape[1])
    ]
    avg_auc = np.nanmean(auc_scores)
    avg_auc_list.append(avg_auc)

    if avg_auc > max_auc:
        max_auc = avg_auc

    # Thresholded predictions
    predicted_labels = (all_probs >= 0.5).astype(int)

    # F1 scores
    per_class_f1 = f1_score(all_labels, predicted_labels, average=None)
    macro_f1 = f1_score(all_labels, predicted_labels, average="macro")

    # Accuracy
    per_class_acc = accuracy_score(all_labels, predicted_labels)
    classes = np.unique(all_labels)
    class_acc = []
    for cls in classes:
        mask = (all_labels == cls)
        correct = (predicted_labels[mask] == all_labels[mask]).sum()
        total = mask.sum()
        acc = correct / total
        class_acc.append(acc)
    macro_acc = np.mean(class_acc)

    # Precision / Recall
    precision_macro = precision_score(all_labels, predicted_labels, average='macro')
    recall_macro = recall_score(all_labels, predicted_labels, average='macro')
    precision_per_class = precision_score(all_labels, predicted_labels, average=None)
    recall_per_class = recall_score(all_labels, predicted_labels, average=None)

    # AUPRC
    auprc_per_class = average_precision_score(all_labels, all_probs, average=None)
    auprc_macro = average_precision_score(all_labels, all_probs, average='macro')

    # TensorBoard logging
    if writer is not None:
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("AUC/Test", avg_auc, epoch)
        writer.add_scalar("F1/Macro", macro_f1, epoch)
        writer.add_scalar("LearningRate", optimizer.param_groups[0]["lr"], epoch)

    if print_every:
        print(
            f"Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), "
            f"Macro F1: {macro_f1:.3f}, Per class F1: {per_class_f1.mean():.3f}"
        )
        for idx, auc in enumerate(auc_scores):
            print(f"Class {idx}: AUC = {auc:.4f}", flush=True)

    # Return test metrics only (or you can also return training info as needed)
    return (
        macro_acc,
        class_acc,
        avg_auc,
        auc_scores,
        macro_f1,
        per_class_f1,
        auprc_macro,
        auprc_per_class,
        recall_macro,
        recall_per_class,
        precision_macro,
        precision_per_class,
        all_probs,
        all_labels,
    )
