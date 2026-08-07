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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DECISION_T = 0.5
BIN_SIZE   = 0.2


# ── Data / inference helpers ───────────────────────────────────────────────

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


DATA_DIR  = r"C:\Users\preet\Documents\mBRSET\mBRSET_image_quality"
DATA_DIR2 = r"C:\Users\preet\Documents\BRSET\data"


def load_split(dataset, split):
    if dataset == "BRSET":
        fname = "brset_val_524.pkl" if split == "val" else "brset_test_524.pkl"
        df = pd.read_pickle(os.path.join(DATA_DIR2, fname))
        df = df.rename(columns={"patient_id": "patient"})
        img_root = r"C:\Users\preet\Documents\BRSET\data\resized_fundus_photos"
    elif dataset == "mBRSET":
        fname = ("mbrset_icdr_quality_524_val_full.pkl" if split == "val"
                 else "mbrset_icdr_quality_524_test_full.pkl")
        df = pd.read_pickle(os.path.join(DATA_DIR, "data", fname))
        df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
        df.reset_index(drop=True, inplace=True)
        img_root = r"C:\Users\preet\Documents\mBRSET\mbrset-a-mobile-brazilian-retinal-dataset-1.0\images"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return df, img_root


def load_model(checkpoint, model_name, device):
    import torch
    from model import UnifiedBackbone
    model = UnifiedBackbone(model_name=model_name).to(device).float()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def run_inference(checkpoint_or_model, dataset, split, model_name, device=None):
    import torch
    from torch.utils.data import DataLoader
    from fundus_dataset import FundusDataset
    from diagnosis_train_eval import validate

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Accept either a path (str) or an already-loaded model
    if isinstance(checkpoint_or_model, str):
        model = load_model(checkpoint_or_model, model_name, device)
    else:
        model = checkpoint_or_model

    df, img_root = load_split(dataset, split)
    tf = get_transforms(model_name)
    ds = FundusDataset(df, img_root, high_quality_tf=tf, low_quality_tf=tf, label_col="final_icdr")
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    loss_fn = torch.nn.CrossEntropyLoss()
    _, metrics, all_files, _, _ = validate(model, loader, loss_fn, device)
    df_probs = pd.DataFrame({"file": all_files, "prob": metrics["all_probs"]})
    df_eval = df.merge(df_probs, on="file", how="left").sort_values(["patient", "file"])
    return df_eval


# ── Calibration logic ─────────────────────────────────────────────────────────

def compute_calibration(df, prob_col, label_col, binarize):
    df = df[[prob_col, label_col]].dropna(subset=[prob_col]).copy()

    raw_labels = df[label_col].astype(int)
    if binarize:
        df["label"] = (raw_labels > 0).astype(int)
    else:
        df["label"] = raw_labels

    df["pred"]    = (df[prob_col] > DECISION_T).astype(int)
    df["correct"] = (df["pred"] == df["label"]).astype(int)

    rows = []
    for b_lo in np.arange(0.0, 1.0, BIN_SIZE).round(2):
        b_hi  = round(b_lo + BIN_SIZE, 2)
        # last bin: include 1.0
        if b_hi == round(1.0, 2):
            mask = (df[prob_col] >= b_lo) & (df[prob_col] <= 1.0)
        else:
            mask = (df[prob_col] >= b_lo) & (df[prob_col] < b_hi)

        subset = df[mask]
        n = len(subset)
        pred_label = int(b_lo >= DECISION_T)

        rows.append({
            "bin":           f"[{b_lo:.1f}, {b_hi:.1f})",
            "bin_lo":        b_lo,
            "bin_hi":        b_hi,
            "n":             n,
            "mean_prob":     subset[prob_col].mean() if n else np.nan,
            "frac_positive": subset["label"].mean()  if n else np.nan,
            "pred_label":    pred_label,
            "accuracy":      subset["correct"].mean() if n else np.nan,
        })

    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_calibration(calib_df, title, output_path):
    nonempty = calib_df[calib_df["n"] > 0].reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left: accuracy per bin ────────────────────────────────────────────────
    ax = axes[0]
    bar_colors = ["steelblue" if r["pred_label"] == 0 else "darkorange"
                  for _, r in nonempty.iterrows()]
    ax.bar(range(len(nonempty)), nonempty["accuracy"],
           color=bar_colors, edgecolor="gray", alpha=0.85)

    for i, (_, row) in enumerate(nonempty.iterrows()):
        if not np.isnan(row["accuracy"]):
            ax.text(i, row["accuracy"] + 0.025, f"n={row['n']}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.axhline(0.5, color="red", linestyle="--", lw=1.2)

    # vertical line between pred=0 and pred=1 bins
    boundary = nonempty[nonempty["pred_label"] == 0].index
    if len(boundary):
        ax.axvline(boundary[-1] + 0.5, color="black", linestyle=":", lw=1.2)

    ax.set_xticks(range(len(nonempty)))
    ax.set_xticklabels(nonempty["bin"], rotation=45, ha="right", fontsize=8.5)
    ax.set_ylim(0, 1.2)
    ax.set_xlabel("Probability bin", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy per probability bin", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    legend_els = [
        Patch(facecolor="steelblue",  label="pred = 0  (prob < 0.5)"),
        Patch(facecolor="darkorange", label="pred = 1  (prob ≥ 0.5)"),
        plt.Line2D([0], [0], color="red", linestyle="--", lw=1.2, label="Chance (0.5)"),
    ]
    ax.legend(handles=legend_els, fontsize=9)

    # ── Right: fraction positive vs mean prob ─────────────────────────────────
    ax2 = axes[1]
    ax2.bar(range(len(nonempty)), nonempty["frac_positive"],
            color="seagreen", edgecolor="gray", alpha=0.85, label="Observed DR rate")
    ax2.plot(range(len(nonempty)), nonempty["mean_prob"],
             marker="o", color="navy", lw=1.5, ms=5, label="Mean predicted confidence")

    ax2.set_xticks(range(len(nonempty)))
    ax2.set_xticklabels(nonempty["bin"], rotation=45, ha="right", fontsize=8.5)
    ax2.set_ylim(0, 1.2)
    ax2.set_xlabel("Model Confidence", fontsize=12)
    ax2.set_ylabel("Fraction of DR-Positive Labels", fontsize=12)
    ax2.set_title("Calibration of predicted DR probability", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(fontsize=9)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")

    # ── Standalone calibration plot (right panel only) ────────────────────────
    stem, ext = os.path.splitext(output_path)
    solo_path = f"{stem}_FINAL{ext}"
    fig2, ax3 = plt.subplots(figsize=(7, 5))
    ax3.bar(range(len(nonempty)), nonempty["frac_positive"],
            color="seagreen", edgecolor="gray", alpha=0.85, label="Observed DR rate")
    ax3.plot(range(len(nonempty)), nonempty["mean_prob"],
             marker="o", color="navy", lw=1.5, ms=5, label="Mean predicted confidence")
    ax3.set_xticks(range(len(nonempty)))
    ax3.set_xticklabels(nonempty["bin"], rotation=45, ha="right", fontsize=8.5)
    ax3.set_ylim(0, 1.2)
    ax3.set_xlabel("Model Confidence", fontsize=12)
    ax3.set_ylabel("Fraction of DR-Positive Labels", fontsize=12)
    ax3.set_title("Calibration of predicted DR probability", fontsize=12)
    ax3.grid(axis="y", alpha=0.3)
    ax3.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(solo_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {solo_path}")


def plot_accuracy_vs_midpoint(calib_df, title, output_path):
    nonempty = calib_df[calib_df["n"] > 0].copy()
    nonempty["midpoint"] = (nonempty["bin_lo"] + nonempty["bin_hi"]) / 2

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(nonempty["midpoint"], nonempty["accuracy"],
            marker="o", ms=7, lw=1.8, color="steelblue")

    for _, row in nonempty.iterrows():
        ax.annotate(f"n={row['n']}",
                    xy=(row["midpoint"], row["accuracy"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color="dimgray")

    ax.axhline(0.5, color="red", linestyle="--", lw=1.2, label="Chance (0.5)")
    ax.axvline(0.5, color="black", linestyle=":", lw=1.2, label="Decision boundary (0.5)")

    ax.set_xlabel("Bin midpoint (probability)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.15)
    ax.set_xticks(nonempty["midpoint"])
    ax.set_xticklabels([f"{v:.2f}" for v in nonempty["midpoint"]], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def run_calibration_for_split(df, split, base_tag, ckpt_stem, dataset,
                               prob_col, label_col, binarize, output_dir):
    tag   = f"{base_tag}_{split}" if base_tag else f"{dataset}_{split}_{ckpt_stem}"
    title = f"Calibration  |  {dataset}  |  {split}  |  {ckpt_stem}"

    if prob_col not in df.columns:
        raise ValueError(f"Column '{prob_col}' not found. Available: {list(df.columns)}")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found. Available: {list(df.columns)}")

    calib_df = compute_calibration(df, prob_col, label_col, binarize)
    print(f"\n[{split}] Calibration summary:")
    print(calib_df.to_string(index=False))

    csv_out = os.path.join(output_dir, f"calibration_{tag}.csv")
    calib_df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    plot_calibration(calib_df, title,
                     os.path.join(output_dir, f"calibration_{tag}.png"))
    plot_accuracy_vs_midpoint(calib_df, f"Accuracy vs bin midpoint  |  {title}",
                              os.path.join(output_dir, f"calibration_accuracy_vs_midpoint_{tag}.png"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibration check: accuracy per 0.1-width probability bin."
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv",        help="Pre-saved CSV with prob and label columns")
    src.add_argument("--checkpoint", help="Model checkpoint — runs inference and calibrates")

    parser.add_argument("--dataset",    choices=["BRSET", "mBRSET"],
                        help="Required when using --checkpoint")
    parser.add_argument("--split",      default="test", choices=["val", "test", "both"],
                        help="Data split to use: val, test, or both (default: test)")
    parser.add_argument("--model_name", default="retfound_green")
    parser.add_argument("--prob_col",   default="prob")
    parser.add_argument("--label_col",  default="final_icdr")
    parser.add_argument("--binarize",   action="store_true", default=True)
    parser.add_argument("--no_binarize", dest="binarize", action="store_false")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tag",        default="")
    args = parser.parse_args()

    if args.checkpoint and not args.dataset:
        parser.error("--dataset is required when using --checkpoint")

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "coverage_graphs", "calibration")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── CSV mode (single split only) ──────────────────────────────────────────
    if args.csv:
        df    = pd.read_csv(args.csv)
        tag   = args.tag or os.path.splitext(os.path.basename(args.csv))[0]
        title = f"Calibration  |  {os.path.basename(args.csv)}"
        print(f"Loaded {len(df)} rows from {args.csv}")
        calib_df = compute_calibration(df, args.prob_col, args.label_col, args.binarize)
        print("\nCalibration summary:")
        print(calib_df.to_string(index=False))
        calib_df.to_csv(os.path.join(args.output_dir, f"calibration_{tag}.csv"), index=False)
        plot_calibration(calib_df, title,
                         os.path.join(args.output_dir, f"calibration_{tag}.png"))
        plot_accuracy_vs_midpoint(calib_df, f"Accuracy vs bin midpoint  |  {title}",
                                  os.path.join(args.output_dir, f"calibration_accuracy_vs_midpoint_{tag}.png"))
        return

    # ── Checkpoint mode ───────────────────────────────────────────────────────
    import torch
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]

    # Load model once
    print(f"Loading model: {ckpt_stem}")
    model = load_model(args.checkpoint, args.model_name, device)

    if args.split == "both":
        print(f"\nRunning inference on {args.dataset} val …")
        df_val  = run_inference(model, args.dataset, "val",  args.model_name, device)
        print(f"Running inference on {args.dataset} test …")
        df_test = run_inference(model, args.dataset, "test", args.model_name, device)
        df = pd.concat([df_val, df_test], ignore_index=True)
        print(f"Combined val+test: {len(df)} image-level rows.")
        run_calibration_for_split(df, "val_and_test", args.tag, ckpt_stem, args.dataset,
                                   args.prob_col, args.label_col, args.binarize,
                                   args.output_dir)
    else:
        print(f"\nRunning inference on {args.dataset} {args.split} …")
        df = run_inference(model, args.dataset, args.split, args.model_name, device)
        print(f"Inference done. {len(df)} image-level rows.")
        run_calibration_for_split(df, args.split, args.tag, ckpt_stem, args.dataset,
                                   args.prob_col, args.label_col, args.binarize,
                                   args.output_dir)


if __name__ == "__main__":
    main()
