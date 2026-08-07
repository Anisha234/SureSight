import os
import sys
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "src")
sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import UnifiedBackbone
from fundus_dataset import FundusDataset
from diagnosis_train_eval import validate as validate_diagnosis
from img_quality_train_val import (validate as validate_iq,
                                   find_thresholds_for_recall,
                                   test as test_iq)

DATA_DIR  = r"C:\Users\preet\Documents\mBRSET\mBRSET_image_quality"
DATA_DIR2 = r"C:\Users\preet\Documents\BRSET\data"

IQ_CHECKPOINTS = {
    "BRSET":  "IQ_BRSETimg_quality_model_392.pth",
    "mBRSET": "IQ_mBRSETimg_quality_model_392.pth",
}

RECALL_TARGETS = [0.99, 0.95, 0.75, 0.55, 0.35, 0.15, 0.05, 0.01]


def get_transforms(model_name):
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    if model_name != "retfound_green":
        return A.Compose([
            A.Resize(392, 392),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(392, 392),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def load_data(dataset, split):
    if dataset == "BRSET":
        fname = {"val": "brset_val_524.pkl", "test": "brset_test_524.pkl"}[split]
        df = pd.read_pickle(os.path.join(DATA_DIR2, fname))
        df = df.rename(columns={"patient_id": "patient"})
        img_root = r"C:\Users\preet\Documents\BRSET\data\resized_fundus_photos"
    elif dataset == "mBRSET":
        fname = {"val":  "mbrset_icdr_quality_524_val_full.pkl",
                 "test": "mbrset_icdr_quality_524_test_full.pkl"}[split]
        df = pd.read_pickle(os.path.join(DATA_DIR, "data", fname))
        df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
        df.reset_index(drop=True, inplace=True)
        img_root = r"C:\Users\preet\Documents\mBRSET\mbrset-a-mobile-brazilian-retinal-dataset-1.0\images"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    df.dropna(subset=["final_icdr"], inplace=True)
    return df, img_root


def load_model(checkpoint, model_name, device):
    model = UnifiedBackbone(model_name=model_name).to(device).float()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def run_iq_validate(iq_model, df, img_root, model_name, device):
    tf = get_transforms(model_name)
    ds = FundusDataset(df, img_root, high_quality_tf=tf, low_quality_tf=tf,
                       label_col="final_quality")
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    loss_fn = torch.nn.CrossEntropyLoss()
    _, metrics = validate_iq(iq_model, loader, loss_fn, device)
    return metrics["all_labels"], metrics["all_probs"]


def get_good_quality_files(iq_model, df, img_root, model_name, device, threshold):
    tf = get_transforms(model_name)
    ds = FundusDataset(df, img_root, high_quality_tf=tf, low_quality_tf=tf,
                       label_col="final_quality")
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=0)
    loss_fn = torch.nn.CrossEntropyLoss()
    _, _, good_files = test_iq(iq_model, loader, loss_fn, device, T=threshold)
    return good_files


def run_diagnosis(diag_model, df, img_root, model_name, device):
    tf = get_transforms(model_name)
    ds = FundusDataset(df, img_root, high_quality_tf=tf, low_quality_tf=tf,
                       label_col="final_icdr")
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    loss_fn = torch.nn.CrossEntropyLoss()
    _, metrics, _, _, _ = validate_diagnosis(diag_model, loader, loss_fn, device)
    ba = metrics["ba"]
    cm = metrics["conf_matrix"]
    denom = cm[1][1] + cm[1][0]
    sensitivity = cm[1][1] / denom if denom > 0 else 0.0
    return ba, sensitivity


def main():
    parser = argparse.ArgumentParser(
        description="Cascade (IQ filter → diagnosis) coverage curve generator."
    )
    parser.add_argument("--dataset", required=True, choices=["BRSET", "mBRSET"])
    parser.add_argument("--diagnosis_checkpoints", nargs="+", required=True,
                        help="One or more diagnosis checkpoint paths to ensemble.")
    parser.add_argument("--model_name", default="retfound_green")
    parser.add_argument("--run_id",     default="")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "coverage_graphs",
                                       f"cascade_{args.dataset.lower()}")
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dataset: {args.dataset}  |  device: {device}")

    # ── Load IQ model ─────────────────────────────────────────────────────────
    iq_ckpt = os.path.join(SCRIPT_DIR, IQ_CHECKPOINTS[args.dataset])
    print(f"IQ checkpoint: {iq_ckpt}")
    iq_model = load_model(iq_ckpt, args.model_name, device)

    # ── Find IQ thresholds on val set ─────────────────────────────────────────
    val_df, img_root = load_data(args.dataset, "val")
    print(f"Val set: {len(val_df)} images — running IQ inference...")
    val_labels, val_probs = run_iq_validate(iq_model, val_df, img_root,
                                            args.model_name, device)
    thresholds = find_thresholds_for_recall(val_labels, val_probs, RECALL_TARGETS)
    print("Class-0 (bad quality) recall thresholds:")
    for tgt, thr in thresholds["class_0"].items():
        print(f"  {tgt*100:.0f}% recall → threshold = {thr}")

    # ── Test set: one image per patient ───────────────────────────────────────
    test_df, img_root = load_data(args.dataset, "test")
    test_df = test_df.groupby("patient", as_index=False).first()
    n_patients = len(test_df)
    print(f"Test set (1 img/patient): {n_patients} patients")

    # ── Load diagnosis models ─────────────────────────────────────────────────
    diag_models = []
    for ckpt in args.diagnosis_checkpoints:
        print(f"Loading diagnosis model: {os.path.basename(ckpt)}")
        diag_models.append(load_model(ckpt, args.model_name, device))

    # ── Sweep recall targets ──────────────────────────────────────────────────
    rows = []
    for target in RECALL_TARGETS:
        thr = thresholds["class_0"].get(target)
        if thr is None:
            print(f"  Skipping recall {target*100:.0f}% — no threshold found")
            continue

        good_files = get_good_quality_files(iq_model, test_df, img_root,
                                            args.model_name, device, thr)
        df_filtered = test_df[test_df["file"].isin(set(good_files))].copy()
        coverage = len(df_filtered) / n_patients
        print(f"  Recall {target*100:.0f}%: thr={thr:.4f}  "
              f"coverage={coverage:.3f}  n_kept={len(df_filtered)}")

        if len(df_filtered) == 0:
            continue

        ba_vals, sens_vals = [], []
        for diag_model in diag_models:
            ba, sens = run_diagnosis(diag_model, df_filtered, img_root,
                                     args.model_name, device)
            ba_vals.append(ba)
            sens_vals.append(sens)

        rows.append({
            "recall_target":    target,
            "img_quality_th":   thr,
            "coverage":         coverage,
            "n_kept":           len(df_filtered),
            "BA_mean":          float(np.mean(ba_vals)),
            "BA_std":           float(np.std(ba_vals)),
            "sensitivity_mean": float(np.mean(sens_vals)),
            "sensitivity_std":  float(np.std(sens_vals)),
        })

    df_out = pd.DataFrame(rows).sort_values("coverage").reset_index(drop=True)
    print("\nResults:")
    print(df_out.to_string(index=False))

    tag = f"cascade_{args.dataset}" + (f"_run{args.run_id}" if args.run_id else "")
    csv_path = os.path.join(args.output_dir, f"{tag}.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
