import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

DECISION_T = 0.5

def selective_pool(group, num_imgs, dataset,
                   t_low, t_high,
                   min_conf_imgs,
                   img_idx=None):

    # Image selection
    if img_idx:
        if num_imgs < 4 and dataset == "mBRSET":
            group = group.iloc[img_idx]
        if num_imgs < 2 and dataset == "BRSET":
            group = group.iloc[img_idx]

    confident = (group["prob"] <= t_low) | (group["prob"] >= t_high)
    confident_probs = group.loc[confident, "prob"]

    if len(confident_probs) == 0:
        return pd.Series({
            "prob": np.nan,
            "covered": 0,
            "n_good": len(confident_probs)
        })

    pooled = confident_probs.mean()

    return pd.Series({
        "prob": pooled,
        "covered": 1, # for one patient atleast one good image
        "n_good": len(confident_probs)
    })


def compute_perf_metrics(df_val_eval,
                         t_low, t_high,
                         p_low, p_high,
                         num_imgs, dataset,
                         img_idx=None):

    df_eye = (
        df_val_eval
        .groupby("patient")
        .apply(lambda g: selective_pool(
            g, num_imgs, dataset,
            t_low, t_high, img_idx
        ))
        .reset_index()
        .merge(
            df_val_eval[["patient", "final_icdr"]]
            .drop_duplicates("patient"),
            on="patient",
            how="left"
        )
    )

    df_eye["label"] = df_eye["final_icdr"].astype(int)
    df_eye["n_good"] = df_eye["n_good"].astype(int)

    # Only apply thresholds to covered patients
    mask = df_eye["covered"] == 1

    df_eye.loc[mask, "p_low_patient"] = p_low
    df_eye.loc[mask, "p_high_patient"] = p_high

    df_eye.loc[mask, "patient_confident"] = (
        (df_eye.loc[mask, "prob"] <= df_eye.loc[mask, "p_low_patient"]) |
        (df_eye.loc[mask, "prob"] >= df_eye.loc[mask, "p_high_patient"])
    )

    df_eye.loc[~mask, "patient_confident"] = False

    df_eye["covered"] = (
        (df_eye["covered"] == 1) &
        (df_eye["patient_confident"])
    ).astype(int)

    coverage = df_eye["covered"].mean()

    df_eval = df_eye[df_eye["covered"] == 1].copy()

    if len(df_eval) == 0:
        return 0, 0, 0, 0, df_eval

    df_eval["pred"] = (df_eval["prob"] > DECISION_T).astype(int)

    ba = balanced_accuracy_score(df_eval["label"], df_eval["pred"])

    cm = confusion_matrix(df_eval["label"],
                          df_eval["pred"],
                          labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    sens = tp / (tp + fn) if (tp + fn) else 0
    spec = tn / (tn + fp) if (tn + fp) else 0

    return coverage, ba, sens, spec, df_eye


from itertools import product
import numpy as np
import pandas as pd

import numpy as np

def determine_test_params(df_results, bin_size):

    BIN_SIZE = bin_size
    df_binned = df_results.copy()

    df_binned["coverage_bin"] = (
        np.floor(df_binned["coverage"] / BIN_SIZE) * BIN_SIZE
    ).round(2)

    df_binned = df_binned.sort_values(
        ["coverage_bin", "BA"],
        ascending=[False, False]
    )


    df_summary = (
        df_binned
        .drop_duplicates(subset=["coverage_bin"], keep="first")
        .loc[:, [
            "t_low",
            "t_high",
            "patient_margin",
            "p_low",
            "p_high",
            "coverage",
            "BA",
            "sensitivity",
            "specificity"
        ]]
        .reset_index(drop=True)
    )

    return df_summary
