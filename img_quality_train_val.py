import torch
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm


from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

import numpy as np
from sklearn.metrics import confusion_matrix

def score_checkpoint_for_operating_point(labels, probs, target_recall=0.99):
    thresholds = np.linspace(0, 1, 2000)
    labels = np.array(labels)
    probs = np.array(probs)

    best_tnr = -1
    best_threshold = None

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

        recall_1 = tp / (tp + fn)
        if recall_1 < target_recall:
            continue

        tnr = tn / (tn + fp)

        if tnr > best_tnr:
            best_tnr = tnr
            best_threshold = t

    return best_tnr, best_threshold

def find_thresholds_for_recall(labels, probs,
                               recall_targets=[1.0, 0.99, 0.95]):
    """
    labels: true labels (0/1)
    probs: probability of class 1
    recall_targets: recall targets to satisfy for each class
    """

    labels = np.array(labels)
    probs = np.array(probs)

    thresholds = np.linspace(0, 1, 2000)

    results = {
        "class_1": {},
        "class_0": {}
    }

    for target in recall_targets:
        # ---------- CLASS 1 (positive class) ----------
        #tp / (tp + fn) --> minimize good quality marked as bad quality
        thr_1 = None
        for t in thresholds:
            preds = (probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            recall_1 = tp / (tp + fn)
            if recall_1 >= target:
                thr_1 = t
                
        results["class_1"][target] = thr_1

        # ---------- CLASS 0 (negative class) ----------
        # NOTE: class 0 recall = TN / (TN + FP)   --> minimize bad quality marked as good quality
        thr_0 = None
        for t in thresholds:
            preds = (probs >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            recall_0 = tn / (tn + fp)
            if recall_0 >= target:
                thr_0 = t
                break
        results["class_0"][target] = thr_0

    return results
def show_conf_matrix_at_threshold(labels, probs, threshold):
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    return cm

def compute_metrics(labels, preds, probs):
    """
    labels: list/array of true labels (0/1)
    preds: argmax predictions (0/1)
    probs: predicted probability of class 1 (float)
    """
    metrics = {}

    metrics["accuracy"] = accuracy_score(labels, preds)
    metrics["balanced_accuracy"] = balanced_accuracy_score(labels, preds)
    metrics["f1"] = f1_score(labels, preds)
    metrics["roc_auc"] = roc_auc_score(labels, probs)
    metrics["auprc"] = average_precision_score(labels, probs)
    metrics["conf_matrix"] = confusion_matrix(labels, preds)

    return metrics


def compute_metrics_test(labels, preds, probs):
    """
    labels: list/array of true labels (0/1)
    preds: argmax predictions (0/1)
    probs: predicted probability of class 1 (float)
    """
    metrics = {}

    metrics["accuracy"] = accuracy_score(labels, preds)
    metrics["balanced_accuracy"] = balanced_accuracy_score(labels, preds)
    metrics["f1"] = f1_score(labels, preds)
    metrics["roc_auc"] = roc_auc_score(labels, probs)
    metrics["auprc"] = average_precision_score(labels, probs)
    metrics["conf_matrix"] = confusion_matrix(labels, preds)
    '''
    recall_targets = [1.0, 0.99, 0.95, 0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]
    thresholds = find_thresholds_for_recall(labels, probs, recall_targets)
    
    print("\n=== Thresholds for Class 1 (Positive Class) ===")
    for target, thr in thresholds["class_1"].items():
        print(f"Recall {target*100:.0f}% → threshold = {thr:.4f}")
        print(show_conf_matrix_at_threshold(labels, probs, thr))
    
    print("\n=== Thresholds for Class 0 (Negative Class) ===")
    for target, thr in thresholds["class_0"].items():
        print(f"Recall {target*100:.0f}% → threshold = {thr:.4f}")
        print(show_conf_matrix_at_threshold(labels, probs, thr))
    '''
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


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()

    running_loss = 0
    all_labels = []
    all_preds  = []
    all_probs  = []

    for imgs, labels, _ in tqdm(loader, desc="Val"):
        imgs = imgs.to(device)
        labels = labels.long().to(device)

        with autocast(dtype=torch.float16):
            logits = model(imgs)
            loss = loss_fn(logits, labels)

        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1).cpu().numpy()
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['all_labels'] = all_labels
    metrics['all_probs'] = all_probs
    return epoch_loss, metrics


@torch.no_grad()
def test(model, loader, loss_fn, device,T=0.5):
    model.eval()

    running_loss = 0
    all_labels = []
    all_preds  = []
    all_probs  = []
    good_quality_files   = []   # predicted == true label
    for imgs, labels,img_files in tqdm(loader, desc="Test"):
        imgs = imgs.to(device)
        labels = labels.long().to(device)

        logits = model(imgs)
        loss = loss_fn(logits, labels)

        running_loss += loss.item() * imgs.size(0)

        #apply image quality standalone model filtering
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= T).astype(int)   # if p0 > T → 0, else 1
        for f, p in zip(img_files, preds):
            if p == 1:
                good_quality_files.append(f)

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics_test(all_labels, all_preds, all_probs)

    return epoch_loss, metrics, good_quality_files

def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, epochs=10, patience=2, str_prefix=''):
    model.to(device)

    best_ba = -1
    best_state = None

    for epoch in range(1, epochs+1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        train_loss, trn_metrics = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_metrics = validate(model, val_loader, loss_fn, device)
        print("train loss", format(train_loss, ".4f"))
        

        print("Train Metrics:")
        for k, v in trn_metrics.items():
        # Case 1: confusion matrix or any array-like
            if isinstance(v, np.ndarray):
                print(f"  {k}:")
                print(v)
        # Case 2: normal number
            else:
                print(f"  {k}: {format(v, '.2f')}")

        


        print("val loss", format(val_loss, ".4f"))
        print("\nVal Metrics:")
        for k, v in val_metrics.items():
        # Case 1: confusion matrix or any array-like
            if isinstance(v, np.ndarray):
                print(f"  {k}:")
                print(v)
        # Case 2: normal number
            elif not isinstance(v, list):
                print(f"  {k}: {format(v, '.2f')}")



        # --- Early stopping check ---
        current_ba = val_metrics["balanced_accuracy"]

        if current_ba > best_ba:
            best_ba = current_ba
            best_state = model.state_dict().copy()
            wait = 0
            print("→ Best model updated.")
            torch.save(best_state, str_prefix+"img_quality_model_392.pth")
        else:
            wait += 1
            print(f"No improvement. Patience counter: {wait}/{patience}")
            
            if wait >= patience:
                print(" Early stopping triggered.")
                break

    # Load best weights before returning
    if best_state is not None:
        model.load_state_dict(best_state)

    return model
