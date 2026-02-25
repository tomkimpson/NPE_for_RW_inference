#!/usr/bin/env python3
"""
Convert 2D training data to 1D by summing observations along the Ly axis.

2D observations have shape (N, Ly, Lx) = (N, 50, 200).
Summing along axis 1 gives column counts of shape (N, 200), matching
the 1D training data format.

Usage:
    python src/convert_2d_to_1d.py
"""

import pickle
from pathlib import Path

import torch

# Source 2D training data paths (all 50k sims)
SOURCES = {
    "original": "results/workflow_original_npe2d_20260223_104550/training_data.pkl",
    "A": "results/workflow_A_npe2d_20260206_160159/training_data.pkl",
    "B": "results/workflow_B_npe2d_20260223_224309/training_data.pkl",
    "C": "results/workflow_C_npe2d_20260224_171232/training_data.pkl",
}

OUTPUT_DIR = Path("results/training_data_1d_50k")


def convert(model_name: str, source_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"Converting {model_name}: {source_path}")

    with open(source_path, "rb") as f:
        data = pickle.load(f)

    params = data["parameters"]
    obs_2d = data["observations"]
    metadata = data["metadata"]

    print(f"  Parameters: {params.shape}")
    print(f"  Observations 2D: {obs_2d.shape}")

    # Sum along Ly axis (axis 1) to get column counts
    obs_1d = obs_2d.sum(dim=1)
    print(f"  Observations 1D: {obs_1d.shape}")

    # Sanity checks
    assert obs_2d.shape[0] == obs_1d.shape[0] == params.shape[0]
    assert obs_1d.shape == (params.shape[0], metadata["Lx"])

    # Update metadata to reflect 1D conversion
    metadata_1d = dict(metadata)
    metadata_1d["converted_from_2d"] = True
    metadata_1d["original_obs_shape"] = list(obs_2d.shape)

    out_data = {
        "parameters": params,
        "observations": obs_1d,
        "metadata": metadata_1d,
    }

    out_path = OUTPUT_DIR / f"{model_name}_training_data_1d.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(out_data, f)

    print(f"  Saved to: {out_path}")
    print(f"  Verified shape: {obs_1d.shape}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, source_path in SOURCES.items():
        if not Path(source_path).exists():
            print(f"WARNING: {source_path} not found, skipping {model_name}")
            continue
        convert(model_name, source_path)

    print(f"\nAll conversions complete. Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
