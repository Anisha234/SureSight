
import torch
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from sklearn.metrics import f1_score
import copy


from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import numpy as np

def compute_balanced_accuracy(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    tpr = tp / (tp + fn + 1e-8)   # sensitivity
    tnr = tn / (tn + fp + 1e-8)   # specificity

    return 0.5 * (tpr + tnr)

def evaluate_thresholds(all_probs, all_labels, T1, T2):
    """
    all_probs: predicted probability of class 1 (numpy or list)
    all_labels: true labels 0/1
    T1, T2: thresholds such that
            p < T1 → predict 0 confidently
            p > T2 → predict 1 confidently
            else → abstain (not confident)

    Returns: (coverage, balanced_accuracy, num_confident)
    """
    metrics={}
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Confident predictions:
    mask_confident = (all_probs < T1) | (all_probs > T2)
    num_confident = mask_confident.sum()

    if num_confident == 0:
        return 0.0, np.nan, 0  # no confident predictions

    # Predictions on confident samples
    preds_conf = np.where(all_probs[mask_confident] > T2, 1, 0)
    labels_conf = all_labels[mask_confident]

    # Balanced Accuracy on confident subset
    metrics["accuracy"] = accuracy_score(labels_conf, preds_conf)
    metrics["ba"] = compute_balanced_accuracy(labels_conf, preds_conf)
    metrics["f1"] = f1_score(labels_conf, preds_conf)
    metrics["conf"] = confusion_matrix(labels_conf, preds_conf)
    # Coverage = proportion of samples classified confidently
    metrics["coverage"] = num_confident / len(all_probs)
    metrics["n_conf"] = num_confident
    return metrics
    



def compute_metrics(labels, preds, probs):
    """
    labels: list/array of true labels (0/1)
    preds: argmax predictions (0/1)
    probs: predicted probability of class 1 (float)
    """
    metrics = {}

    metrics["accuracy"] = accuracy_score(labels, preds)
    metrics["ba"] = balanced_accuracy_score(labels, preds)
    metrics["f1"] = f1_score(labels, preds)
    metrics["roc_auc"] = roc_auc_score(labels, probs)
    metrics["auprc"] = average_precision_score(labels, probs)
    metrics["conf_matrix"] = confusion_matrix(labels, preds)

    return metrics
    
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    scaler = GradScaler()

    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []

    for imgs, labels,_ in tqdm(loader, desc="Train"):
        imgs = imgs.to(device)
        labels = labels.long().to(device)

        optimizer.zero_grad()

        with autocast(dtype=torch.float16):
            logits = model(imgs)        # if model returns features, logits, change to logits = model(imgs)[1]
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1).cpu().detach().numpy()
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().detach().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)

    # compute advanced metrics
    metrics = compute_metrics(all_labels, all_preds, all_probs)

    return epoch_loss, metrics



def train_model(model, train_loader, val_loader, optimizer, loss_fn, device,
                epochs=10, patience=5, str_prefix='mBRSET'):

    model.to(device)

    best_models = []   # list of (ba, state_dict)
    no_improve = 0
    best_ba = -1

    for epoch in range(1, epochs+1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        train_loss, trn_metrics = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device
        )

        val_loss, val_metrics, _, _, _ = validate(
            model, val_loader, loss_fn, device
        )

        current_ba = val_metrics["ba"]
        print(f"Validation BA: {current_ba:.4f}")

        # -----------------------------------
        # Store top 5 models
        # -----------------------------------
        state_copy = copy.deepcopy(model.state_dict())
        best_models.append((current_ba, state_copy))

        # Sort descending by BA
        best_models = sorted(best_models, key=lambda x: x[0], reverse=True)

        # Keep only top 5
        if len(best_models) > 5:
            best_models = best_models[:5]

        # -----------------------------------
        # Early stopping logic
        # -----------------------------------
        if current_ba > best_ba:
            best_ba = current_ba
            no_improve = 0
            print("→ Improvement.")
        else:
            no_improve += 1
            print(f"→ No improvement ({no_improve}/{patience})")

            if no_improve >= patience:
                print("\n⚠️ Early stopping triggered!")
                break

    # -----------------------------------
    # Save all top 5 models
    # -----------------------------------
    print("\nSaving top 5 models:")
    for i, (ba, state) in enumerate(best_models):
        filename = f"{str_prefix}_img_diagnosis_model_top{i+1}_BA_{ba:.4f}.pth"
        torch.save(state, filename)
        print(f"Saved: {filename}")

    # Load best model (rank 1)
    if best_models:
        print(f"\nLoading best model with BA = {best_models[0][0]:.4f}")
        model.load_state_dict(best_models[0][1])

    return model


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()

    running_loss = 0
    all_labels = []
    all_preds  = []
    all_probs  = []

    correct_files   = []   # predicted == true label
    incorrect_files = []   # predicted != true label
    all_files =[]
    for imgs, labels, files in tqdm(loader, desc="Val"):
        imgs = imgs.to(device)
        labels = labels.long().to(device)

        with autocast(dtype=torch.float16):
            logits = model(imgs)
            loss = loss_fn(logits, labels)

        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)          # (B,)
        probs = torch.softmax(logits, dim=1)[:, 1]   # prob of class 1

        # CPU conversion
        preds_np  = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        probs_np  = probs.cpu().numpy()

        all_preds.extend(preds_np)
        all_labels.extend(labels_np)
        all_probs.extend(probs_np)

        # ---- Store files ----
        for f, p, gt in zip(files, preds_np, labels_np):
            if p == gt:
                correct_files.append(f)
            else:
                incorrect_files.append(f)
            all_files.append(f)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["all_labels"] = all_labels
    metrics["all_probs"] = all_probs
    return epoch_loss, metrics, all_files,correct_files, incorrect_files

