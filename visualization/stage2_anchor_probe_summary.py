from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


METHOD_ORDER = ["dense_stage2", "topk_stage2", "qout_stage2", "qmid_stage2", "qin_stage2", "qinwithmid_stage2"]
METHOD_LABELS = {
    "dense_stage2": "dense_stage2",
    "topk_stage2": "topk_stage2",
    "qout_stage2": "qout_stage2",
    "qmid_stage2": "qmid_stage2",
    "qin_stage2": "qin_stage2",
    "qinwithmid_stage2": "qinwithmid_stage2",
}

TASK_ORDER = ["qasper", "passage_retrieval_en"]
ANCHOR_METHODS = ["qout_stage2", "qmid_stage2", "qin_stage2", "qinwithmid_stage2"]


def _load_score(method_dir: Path, dataset: str) -> float | None:
    path = method_dir / "longbench_result_summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    value = data.get("individual_dataset_result", {}).get(dataset)
    return None if value in {None, "N/A"} else float(value)


def collect(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_rows = []
    route_rows = []
    longbench_root = root / "longbench"
    for method in METHOD_ORDER:
        method_root = longbench_root / method
        if not method_root.exists():
            continue
        for budget_dir in sorted(p for p in method_root.iterdir() if p.is_dir() and p.name.isdigit()):
            budget = int(budget_dir.name)
            for model_dir in sorted(p for p in budget_dir.iterdir() if p.is_dir()):
                for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                    dataset = dataset_dir.name
                    score = _load_score(model_dir, dataset)
                    if score is not None:
                        task_rows.append(
                            {
                                "method": method,
                                "method_label": METHOD_LABELS[method],
                                "budget": budget,
                                "model": model_dir.name,
                                "task": dataset,
                                "score": score,
                            }
                        )
                    route_path = dataset_dir / "stage2_probe_summary.csv"
                    if route_path.exists() and route_path.stat().st_size > 0:
                        df = pd.read_csv(route_path)
                        for _, row in df.iterrows():
                            item = row.to_dict()
                            item.update(
                                {
                                    "method": method,
                                    "method_label": METHOD_LABELS[method],
                                    "budget": budget,
                                    "model": model_dir.name,
                                    "task": dataset,
                                }
                            )
                            route_rows.append(item)
    return pd.DataFrame(task_rows), pd.DataFrame(route_rows)


def collect_layer_bucket(root: Path) -> pd.DataFrame:
    rows = []
    longbench_root = root / "longbench"
    for method in METHOD_ORDER:
        method_root = longbench_root / method
        if not method_root.exists():
            continue
        for layer_path in method_root.glob("*/*/*/stage2_probe_layer_summary.csv"):
            parts = layer_path.parts
            try:
                budget = int(parts[-4])
                model = parts[-3]
                task = parts[-2]
            except (ValueError, IndexError):
                continue
            if not layer_path.exists() or layer_path.stat().st_size == 0:
                continue
            df = pd.read_csv(layer_path)
            for _, row in df.iterrows():
                item = row.to_dict()
                item.update(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "budget": budget,
                        "model": model,
                        "task": task,
                    }
                )
                rows.append(item)
    return pd.DataFrame(rows)


def collect_layer_details(root: Path) -> pd.DataFrame:
    rows = []
    longbench_root = root / "longbench"
    for method in METHOD_ORDER:
        method_root = longbench_root / method
        if not method_root.exists():
            continue
        for metrics_path in method_root.glob("*/*/*/stage2_probe_metrics.jsonl"):
            parts = metrics_path.parts
            # .../longbench/<method>/<budget>/<model>/<task>/stage2_probe_metrics.jsonl
            try:
                budget = int(parts[-4])
                model = parts[-3]
                task = parts[-2]
            except (ValueError, IndexError):
                continue
            with metrics_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    item["method"] = method
                    item["method_label"] = METHOD_LABELS[method]
                    item["budget"] = budget
                    item["model"] = model
                    item["task"] = task
                    rows.append(item)
    return pd.DataFrame(rows)


def add_anchor_alignment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "layer_idx" not in df.columns:
        return df
    out = df.copy()
    out["target_layer_idx"] = out["layer_idx"].astype(int)
    out["anchor_layer_idx"] = out["target_layer_idx"]
    out["anchor_available"] = True

    prev_anchor = out["method"].isin(["qin_stage2", "qmid_stage2", "qinwithmid_stage2"])
    out.loc[prev_anchor, "anchor_layer_idx"] = out.loc[prev_anchor, "target_layer_idx"] - 1
    out.loc[prev_anchor, "anchor_available"] = out.loc[prev_anchor, "anchor_layer_idx"] >= 0
    out.loc[out["method"].isin(["dense_stage2", "topk_stage2"]), "anchor_layer_idx"] = pd.NA
    return out


def build_task_gaps(task_df: pd.DataFrame) -> pd.DataFrame:
    if task_df.empty:
        return task_df
    pivot = task_df.pivot_table(index=["model", "budget", "task"], columns="method", values="score", aggfunc="mean")
    rows = []
    for (model, budget, task), values in pivot.iterrows():
        for method in METHOD_ORDER:
            if method not in values or pd.isna(values[method]):
                continue
            rows.append(
                {
                    "model": model,
                    "budget": budget,
                    "task": task,
                    "method": method,
                    "score": float(values[method]),
                    "gap_to_dense": float(values[method] - values.get("dense_stage2", float("nan"))),
                    "gap_to_topk": float(values[method] - values.get("topk_stage2", float("nan"))),
                    "gap_to_qout": float(values[method] - values.get("qout_stage2", float("nan"))),
                }
            )
    gap_df = pd.DataFrame(rows)
    avg = (
        gap_df.groupby(["model", "budget", "method"], as_index=False)[["score", "gap_to_dense", "gap_to_topk", "gap_to_qout"]]
        .mean()
        .assign(task="avg")
    )
    return pd.concat([gap_df, avg], ignore_index=True)


def plot_task_gaps(gap_df: pd.DataFrame, plots_dir: Path) -> None:
    if gap_df.empty:
        return
    sns.set_theme(style="whitegrid", context="talk")
    sub = gap_df[gap_df["task"] == "avg"].copy()
    if sub.empty:
        return
    sub["method"] = pd.Categorical(sub["method"], categories=METHOD_ORDER, ordered=True)
    sub = sub.sort_values("method")
    models = sorted(sub["model"].unique())
    fig, axes = plt.subplots(len(models), 1, figsize=(10, max(4, 3.2 * len(models))), sharex=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models, strict=False):
        model_df = sub[sub["model"] == model].copy()
        sns.barplot(data=model_df, x="method", y="gap_to_topk", ax=ax, color="#2563eb")
        ax.axhline(0.0, color="#9ca3af", linestyle="--", linewidth=1.2)
        ax.set_xlabel("")
        ax.set_ylabel("Gap to topk")
        ax.set_title(model)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"task_gap_to_topk_stage2.{ext}", bbox_inches="tight")
    plt.close(fig)


def _plot_layer_bucket(layer_df: pd.DataFrame, plots_dir: Path, metric: str, ylabel: str) -> None:
    if layer_df.empty:
        return
    sns.set_theme(style="whitegrid", context="talk")
    order = ["early", "middle", "late"]
    sub = layer_df[layer_df["method"].isin(ANCHOR_METHODS)].copy()
    sub["layer_bucket"] = pd.Categorical(sub["layer_bucket"], categories=order, ordered=True)
    sub["method"] = pd.Categorical(sub["method"], categories=ANCHOR_METHODS, ordered=True)
    models = sorted(sub["model"].unique())
    fig, axes = plt.subplots(len(models), len(TASK_ORDER), figsize=(14, max(4.8, 3.6 * len(models))), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for row_axes, model in zip(axes, models, strict=False):
        for ax, task in zip(row_axes, TASK_ORDER, strict=False):
            task_df = sub[(sub["model"] == model) & (sub["task"] == task)].copy()
            sns.lineplot(data=task_df, x="layer_bucket", y=metric, hue="method", marker="o", linewidth=2.2, ax=ax)
            ax.set_title(f"{model} / {task}")
            ax.set_xlabel("Layer bucket")
            ax.set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for row_axes in axes:
        for ax in row_axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"layer_bucket_{metric}.{ext}", bbox_inches="tight")
    plt.close(fig)


def _plot_layer_idx(layer_detail_df: pd.DataFrame, plots_dir: Path, metric: str, ylabel: str) -> None:
    if layer_detail_df.empty:
        return
    sns.set_theme(style="whitegrid", context="talk")
    sub = layer_detail_df[layer_detail_df["method"].isin(ANCHOR_METHODS)].copy()
    grouped = (
        sub.groupby(["model", "task", "method", "layer_idx"], as_index=False)[metric]
        .mean()
        .sort_values(["model", "task", "method", "layer_idx"])
    )
    models = sorted(grouped["model"].unique())
    fig, axes = plt.subplots(len(models), len(TASK_ORDER), figsize=(15, max(4.8, 3.6 * len(models))), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for row_axes, model in zip(axes, models, strict=False):
        for ax, task in zip(row_axes, TASK_ORDER, strict=False):
            task_df = grouped[(grouped["model"] == model) & (grouped["task"] == task)].copy()
            sns.lineplot(data=task_df, x="layer_idx", y=metric, hue="method", linewidth=1.9, ax=ax)
            ax.set_title(f"{model} / {task}")
            ax.set_xlabel("Target layer")
            ax.set_ylabel(ylabel)
    handles, labels = axes[0][0].get_legend_handles_labels()
    for row_axes in axes:
        for ax in row_axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(plots_dir / f"layer_idx_{metric}.{ext}", bbox_inches="tight")
    plt.close(fig)


def build_qin_qmid_delta(layer_detail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if layer_detail_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    sub = layer_detail_df[layer_detail_df["method"].isin(["qin_stage2", "qmid_stage2"])].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    grouped = (
        sub.groupby(["model", "task", "method", "layer_idx"], as_index=False)[
            ["topk_recall", "mass_gap_to_topk", "page_recall"]
        ]
        .mean()
        .sort_values(["model", "task", "method", "layer_idx"])
    )
    wide = grouped.pivot_table(index=["model", "task", "layer_idx"], columns="method")
    rows = []
    for (model, task, layer_idx), values in wide.iterrows():
        try:
            qin_mass = float(values[("mass_gap_to_topk", "qin_stage2")])
            qmid_mass = float(values[("mass_gap_to_topk", "qmid_stage2")])
            qin_recall = float(values[("topk_recall", "qin_stage2")])
            qmid_recall = float(values[("topk_recall", "qmid_stage2")])
            qin_page = float(values[("page_recall", "qin_stage2")])
            qmid_page = float(values[("page_recall", "qmid_stage2")])
        except (KeyError, TypeError, ValueError):
            continue
        mass_delta = qin_mass - qmid_mass
        recall_delta = qin_recall - qmid_recall
        page_delta = qin_page - qmid_page
        rows.append(
            {
                "model": model,
                "task": task,
                "layer_idx": int(layer_idx),
                "qin_mass_gap_to_topk": qin_mass,
                "qmid_mass_gap_to_topk": qmid_mass,
                "qin_minus_qmid_mass_gap": mass_delta,
                "qin_topk_recall": qin_recall,
                "qmid_topk_recall": qmid_recall,
                "qin_minus_qmid_topk_recall": recall_delta,
                "qin_page_recall": qin_page,
                "qmid_page_recall": qmid_page,
                "qin_minus_qmid_page_recall": page_delta,
                "preferred_by_mass": "qin" if mass_delta >= 0 else "qmid",
                "qin_safe_mass_eps_0.005": mass_delta >= -0.005,
                "qin_safe_recall_eps_0.01": recall_delta >= -0.01,
            }
        )
    delta = pd.DataFrame(rows)
    if delta.empty:
        return delta, pd.DataFrame()
    rec_rows = []
    for (model, task), part in delta.groupby(["model", "task"]):
        qin_layers = part.loc[part["preferred_by_mass"] == "qin", "layer_idx"].astype(int).tolist()
        qmid_layers = part.loc[part["preferred_by_mass"] == "qmid", "layer_idx"].astype(int).tolist()
        qin_safe = part.loc[part["qin_safe_mass_eps_0.005"], "layer_idx"].astype(int).tolist()
        rec_rows.append(
            {
                "model": model,
                "task": task,
                "qin_better_layers_by_mass": " ".join(map(str, qin_layers)),
                "qmid_better_layers_by_mass": " ".join(map(str, qmid_layers)),
                "qin_safe_layers_mass_eps_0.005": " ".join(map(str, qin_safe)),
                "qin_better_layer_count": len(qin_layers),
                "qmid_better_layer_count": len(qmid_layers),
            }
        )
    return delta, pd.DataFrame(rec_rows)


def plot_qin_qmid_delta(delta_df: pd.DataFrame, plots_dir: Path) -> None:
    if delta_df.empty:
        return
    sns.set_theme(style="white", context="talk")
    metrics = [
        ("qin_minus_qmid_mass_gap", "Qin - Qmid mass gap to topk", "layer_idx_qin_minus_qmid_mass_gap"),
        ("qin_minus_qmid_topk_recall", "Qin - Qmid topK recall", "layer_idx_qin_minus_qmid_topk_recall"),
    ]
    data = delta_df.copy()
    data["row"] = data["model"] + " / " + data["task"]
    row_order = [
        f"{model} / {task}"
        for model in sorted(data["model"].unique())
        for task in TASK_ORDER
        if ((data["model"] == model) & (data["task"] == task)).any()
    ]
    for metric, title, filename in metrics:
        pivot = data.pivot_table(index="row", columns="layer_idx", values=metric, aggfunc="mean").reindex(row_order)
        vmax = float(pivot.abs().max().max())
        if not vmax or pd.isna(vmax):
            vmax = 1.0
        fig, ax = plt.subplots(figsize=(16, max(4, 0.55 * len(pivot) + 2)))
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdBu_r",
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.2,
            linecolor="#f3f4f6",
            cbar_kws={"label": "positive = qin better"},
        )
        ax.set_title(title)
        ax.set_xlabel("Target layer")
        ax.set_ylabel("")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(plots_dir / f"{filename}.{ext}", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="result/stage2_anchor_probe")
    args = parser.parse_args()

    root = Path(args.root)
    out = root / "summary"
    plots = root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    task_df, route_df = collect(root)
    layer_bucket_df = collect_layer_bucket(root)
    layer_detail_df = add_anchor_alignment(collect_layer_details(root))
    task_df.to_csv(out / "stage2_anchor_task_scores.csv", index=False)
    route_df.to_csv(out / "stage2_anchor_route_summary.csv", index=False)
    layer_bucket_df.to_csv(out / "stage2_anchor_layer_bucket_summary.csv", index=False)
    layer_detail_df.to_csv(out / "stage2_anchor_layer_detail.csv", index=False)
    if not layer_detail_df.empty:
        alignment_cols = [
            "method",
            "task",
            "target_layer_idx",
            "anchor_layer_idx",
            "anchor_available",
            "topk_recall",
            "mass_gap_to_topk",
            "page_recall",
        ]
        layer_detail_df[alignment_cols].to_csv(out / "stage2_anchor_layer_alignment.csv", index=False)
    gap_df = build_task_gaps(task_df)
    gap_df.to_csv(out / "stage2_anchor_task_gaps.csv", index=False)
    plot_task_gaps(gap_df, plots)

    _plot_layer_bucket(layer_bucket_df, plots, "topk_recall", "TopK recall vs topk_stage2")
    _plot_layer_bucket(layer_bucket_df, plots, "mass_gap_to_topk", "Mass gap to topk_stage2")
    _plot_layer_bucket(layer_bucket_df, plots, "page_recall", "Page recall vs topk_stage2")
    _plot_layer_idx(layer_detail_df, plots, "topk_recall", "TopK recall vs topk_stage2")
    _plot_layer_idx(layer_detail_df, plots, "mass_gap_to_topk", "Mass gap to topk_stage2")
    _plot_layer_idx(layer_detail_df, plots, "page_recall", "Page recall vs topk_stage2")
    delta_df, rec_df = build_qin_qmid_delta(layer_detail_df)
    delta_df.to_csv(out / "stage2_anchor_qin_qmid_layer_delta.csv", index=False)
    rec_df.to_csv(out / "stage2_anchor_qin_qmid_layer_recommendation.csv", index=False)
    plot_qin_qmid_delta(delta_df, plots)


if __name__ == "__main__":
    main()
