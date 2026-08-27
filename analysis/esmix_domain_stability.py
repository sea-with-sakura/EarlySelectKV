from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


ANCHORS = {"lookahead_qin": "in", "lookahead_qmid": "mid"}


def load_profile(path: Path, tau: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"layer_idx", "method", "mass"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[frame["method"].isin(ANCHORS)].copy()
    profile = frame.groupby(["layer_idx", "method"], observed=True)["mass"].mean().unstack("method")
    missing_methods = set(ANCHORS) - set(profile.columns)
    if missing_methods:
        raise ValueError(f"{path} is missing anchor methods: {sorted(missing_methods)}")
    profile = profile.sort_index().reset_index()
    profile["gap"] = profile["lookahead_qmid"] - profile["lookahead_qin"]
    profile["assignment"] = np.where(profile["gap"] <= tau, "ES-In", "ES-Mid")
    return profile


def compare_profiles(model: str, profiles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left, right in itertools.combinations(sorted(profiles), 2):
        merged = profiles[left][["layer_idx", "gap", "assignment"]].merge(
            profiles[right][["layer_idx", "gap", "assignment"]],
            on="layer_idx",
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        rows.append(
            {
                "model": model,
                "domain_left": left,
                "domain_right": right,
                "routed_layers": len(merged),
                "pearson_r": merged["gap_left"].corr(merged["gap_right"]),
                "assignment_agreement": (merged["assignment_left"] == merged["assignment_right"]).mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ES-Mix calibration profiles across domains.")
    parser.add_argument("--root", default="result/esmix_domain_stability")
    parser.add_argument("--models", nargs="+", default=["mistral-7b-instruct-v0.2", "llama3-8b-instruct", "llama3.1-8b-instruct", "qwen2.5-7b-instruct"])
    parser.add_argument("--domains", nargs="+", default=["alpaca12k_decode_calib", "qasper", "qmsum"])
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir) if args.output_dir else root / "summary"
    out.mkdir(parents=True, exist_ok=True)
    all_pairs: list[pd.DataFrame] = []
    profile_rows: list[pd.DataFrame] = []
    for model in args.models:
        profiles: dict[str, pd.DataFrame] = {}
        for domain in args.domains:
            path = root / "longbench" / "anchor_quality_stage2" / str(args.budget) / model / domain / "stage2_probe_layer_detail_summary.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing profile input: {path}")
            profile = load_profile(path, args.tau)
            profile.insert(0, "domain", domain)
            profile.insert(0, "model", model)
            profile_rows.append(profile)
            profiles[domain] = profile
        all_pairs.append(compare_profiles(model, profiles))

    pairs = pd.concat(all_pairs, ignore_index=True)
    profiles_out = pd.concat(profile_rows, ignore_index=True)
    ranges = pairs.groupby("model", observed=True).agg(
        routed_layers=("routed_layers", "first"),
        pearson_min=("pearson_r", "min"),
        pearson_max=("pearson_r", "max"),
        agreement_min=("assignment_agreement", "min"),
        agreement_max=("assignment_agreement", "max"),
    ).reset_index()
    pairs.to_csv(out / "esmix_domain_pairwise.csv", index=False)
    profiles_out.to_csv(out / "esmix_layer_profiles.csv", index=False)
    ranges.to_csv(out / "esmix_domain_stability.csv", index=False)
    print(ranges.to_string(index=False))


if __name__ == "__main__":
    main()
