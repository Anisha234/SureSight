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

DATA_DIR  = r"C:\Users\preet\Documents\mBRSET\mBRSET_image_quality"
DATA_DIR2 = r"C:\Users\preet\Documents\BRSET\data"

CHECKPOINTS = {
    "BRSET":  "may24_BRSET__img_diagnosis_model_top1_BA_0.9162.pth",
    "mBRSET": "may24_mBRSET__img_diagnosis_model_top1_BA_0.8202.pth",
}


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


def load_test_data(dataset):
    if dataset == "BRSET":
        df = pd.read_pickle(os.path.join(DATA_DIR2, "brset_test_524.pkl"))
        df = df.rename(columns={"patient_id": "patient"})
        img_root = r"C:\Users\preet\Documents\BRSET\data\resized_fundus_photos"
    elif dataset == "mBRSET":
        df = pd.read_pickle(os.path.join(DATA_DIR, "data", "mbrset_icdr_quality_524_test_full.pkl"))
        df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")
        df.reset_index(drop=True, inplace=True)
        img_root = r"C:\Users\preet\Documents\mBRSET\mbrset-a-mobile-brazilian-retinal-dataset-1.0\images"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return df, img_root


def run_inference(checkpoint, dataset, model_name, device):
    import torch
    from torch.utils.data import DataLoader
    from model import UnifiedBackbone
    from fundus_dataset import FundusDataset
    from diagnosis_train_eval import validate

    model = UnifiedBackbone(model_name=model_name).to(device).float()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    df, img_root = load_test_data(dataset)
    tf = get_transforms(model_name)
    ds = FundusDataset(df, img_root, high_quality_tf=tf, low_quality_tf=tf, label_col="final_icdr")
    loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)

    loss_fn = torch.nn.CrossEntropyLoss()
    _, metrics, all_files, _, _ = validate(model, loader, loss_fn, device)

    df_probs = pd.DataFrame({"file": all_files, "prob": metrics["all_probs"]})
    df_eval = df.merge(df_probs, on="file", how="left")
    df_eval["binary_label"] = (df_eval["final_icdr"].astype(int) > 0).astype(int)
    return df_eval


def plot_histogram(probs, class_name, class_idx, dataset, output_dir, n_bins=20):
    fig, ax = plt.subplots(figsize=(8, 5))

    color = "steelblue" if class_idx == 0 else "darkorange"
    ax.hist(probs, bins=n_bins, range=(0.0, 1.0), color=color,
            edgecolor="white", linewidth=0.6, alpha=0.85)

    ax.axvline(0.5, color="red", linestyle="--", lw=1.4, label="Decision boundary (0.5)")
    ax.set_xlabel("Predicted probability (DR)", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    ax.set_title(
        f"Probability distribution  |  {dataset}  |  {class_name}  (n={len(probs)})",
        fontsize=12
    )
    ax.set_xlim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    fname = f"prob_hist_{dataset}_class{class_idx}_{class_name.replace(' ', '_')}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Probability histograms split by class (healthy vs DR) on the test set."
    )
    parser.add_argument("--dataset", required=True, choices=["BRSET", "mBRSET"])
    parser.add_argument("--model_name", default="retfound_green")
    parser.add_argument("--n_bins", type=int, default=20,
                        help="Number of histogram bins (default: 20)")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(SCRIPT_DIR, "coverage_graphs", "prob_histograms")
    os.makedirs(args.output_dir, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = os.path.join(SCRIPT_DIR, CHECKPOINTS[args.dataset])
    print(f"Checkpoint : {checkpoint}")
    print(f"Dataset    : {args.dataset}  |  device: {device}")

    df_eval = run_inference(checkpoint, args.dataset, args.model_name, device)
    print(f"Inference done. {len(df_eval)} image-level rows.")

    class0_probs = df_eval.loc[df_eval["binary_label"] == 0, "prob"].dropna().values
    class1_probs = df_eval.loc[df_eval["binary_label"] == 1, "prob"].dropna().values

    print(f"Class 0 (healthy): {len(class0_probs)} images")
    print(f"Class 1 (DR)     : {len(class1_probs)} images")

    plot_histogram(class0_probs, "healthy (class 0)", 0, args.dataset, args.output_dir, args.n_bins)
    plot_histogram(class1_probs, "DR (class 1)",      1, args.dataset, args.output_dir, args.n_bins)


if __name__ == "__main__":
    main()
