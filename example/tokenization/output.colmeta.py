#!/usr/bin/env python3
import os
import re
import numpy as np
import csv

def extract_gsm_id(path: str) -> str:
    """
    Extract GSM ID from filenames like:
    GSM5808054_new_matrix.npy / GSM5808054_new_spots.npy / GSM5808054_new_genes.npy
    """
    base = os.path.basename(path)
    m = re.match(r"(GSM\d+)_", base)
    if not m:
        raise ValueError(f"Could not parse GSM ID from filename: {base}")
    return m.group(1)

def main():
    spots_path  = "GSM5808057_new_spots.npy"
    matrix_path = "GSM5808057_new_matrix.npy"
    out_path = "col.metadata/GSM5808057_with_nei.pkl"

    gsm_id = extract_gsm_id(matrix_path)

    # --- load spots (goes to first column) ---
    spots = np.load(spots_path, allow_pickle=True)
    spots = np.asarray(spots).astype(str)

    # --- load and process matrix exactly as specified ---
    mat = np.load(matrix_path, allow_pickle=True)

    scale = 6000.0
    mat = mat.astype(np.float32, copy=False)
    col_sums = mat.sum(axis=0, keepdims=True)
    mat_norm = mat / col_sums * scale  # computed per your instructions (not used after, per spec)
    mat = np.log1p(mat)               # per spec: log1p on original mat

    # n_counts: total counts of (mat) per spot (2nd dimension => columns)
    # i.e., sum over genes (rows) for each spot (column)
    n_counts = mat.sum(axis=0)

    if spots.shape[0] != n_counts.shape[0]:
        raise ValueError(
            f"Mismatch: spots has {spots.shape[0]} entries, but matrix has {n_counts.shape[0]} columns."
        )

    # --- write CSV ---
    # Header must start with an empty first field:
    header = ["", "n_counts", "study", "Tissue", "Preparation", "Sample ID"]

    study = "GSE193460"
    tissue = "lung"
    prep = "fresh frozen"
    sample_id = gsm_id

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for spot_id, cnt in zip(spots, n_counts):
            w.writerow([spot_id, int(cnt), study, tissue, prep, sample_id])

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()

