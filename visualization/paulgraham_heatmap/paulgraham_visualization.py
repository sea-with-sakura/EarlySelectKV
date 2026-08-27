import argparse
import csv
import json
import os
from pathlib import Path

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default='output/paulgraham_passkey')
args = parser.parse_args()

def find_result_dirs(output_dir):
    root = Path(output_dir)
    if not root.exists():
        print(f"Output directory does not exist: {root}")
        return []
    if (root / "output_config.json").is_file():
        return [root]
    return sorted(path.parent for path in root.rglob("output_config.json"))


def summarize_result(result_dir):
    config_path = result_dir / "output_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    pipeline_params = config.get("pipeline_params", {})
    eval_params = config.get("eval_params", {})
    processed_results = config.get("eval_results", {}).get("processed_results", {})
    overall = processed_results.get("overall_results", {})
    return {
        "dataset_config": result_dir.name,
        "result_dir": str(result_dir),
        "method": pipeline_params.get("method"),
        "model_name": pipeline_params.get("model_name"),
        "token_budget": pipeline_params.get("token_budget"),
        "exact_match": overall.get("exact_match"),
        "partial_match": overall.get("partial_match"),
        "background_len_min": eval_params.get("background_len_min"),
        "background_len_max": eval_params.get("background_len_max"),
        "n_background_lens": eval_params.get("n_background_lens"),
        "n_depths": eval_params.get("n_depths"),
        "depth_num_iterations": eval_params.get("depth_num_iterations"),
        "retrieval_target_len": eval_params.get("retrieval_target_len"),
    }


def write_summary(output_dir, rows):
    if not rows:
        return
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "paulgraham_result_summary.json"
    csv_path = root / "paulgraham_result_summary.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    if pd is not None:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    else:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Summary saved to: {json_path}")
    print(f"Summary saved to: {csv_path}")


def generate_plot(output_dir):
    result_dirs = find_result_dirs(output_dir)
    if not result_dirs:
        print(f"No completed PaulGraham passkey result found under: {output_dir}")
        return

    rows = []
    for result_dir in result_dirs:
        rows.append(summarize_result(result_dir))
        generate_one_plot(result_dir)
    write_summary(output_dir, rows)


def generate_one_plot(output_dir):
    if pd is None:
        print("Heatmap skipped: pandas is not installed.")
        return
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from mpl_toolkits.axes_grid1 import make_axes_locatable
    except ModuleNotFoundError as exc:
        print(f"Heatmap skipped: {exc.name} is not installed.")
        return

    file_path = os.path.join(output_dir, 'output_config.json')
    print(f"\nGenerating plot for file: {file_path}")
    with open(file_path, 'r') as f:
        total_file = json.load(f)

    if 'eval_results' not in total_file:
        print("No eval_results in file")
        return
    if 'processed_results' not in total_file['eval_results']:
        print("No processed_results in eval_results")
        return

    file = total_file['eval_results']['processed_results']
    plot_filename = f'result_heatmap.pdf'
    plot_path = os.path.join(output_dir, plot_filename)
    print(f"Plot will be saved as: {plot_path}")

    if not file:
        print("Empty processed_results")
        return

    data = []
    for context_length, depth_results in file.items():
        if context_length in {"background_len_wise_results", "overall_results"}:
            continue
        for depth_lvl, results in depth_results.items():
            if int(context_length) > 1000:
                ctx_length = str(int(context_length) // 1000) + 'K'
            else:
                ctx_length = str(round(int(context_length) / 1000, 1)) + 'K'

            if 'exact_match' not in results:
                continue

            data.append({
                "Context Length": ctx_length,
                'Ctx_Length_Value': int(context_length),
                "Document Depth": round(float(depth_lvl), 2),
                "Exact Match": results['exact_match']
            })

    if not data:
        return

    df = pd.DataFrame(data)
    df = df.sort_values(by=['Ctx_Length_Value', "Document Depth"])

    pivot_table = pd.pivot_table(
        df,
        values='Exact Match',
        index=['Document Depth', 'Ctx_Length_Value'],
        aggfunc='mean'
    ).reset_index()

    pivot_table = pivot_table.pivot(
        index="Document Depth",
        columns="Ctx_Length_Value",
        values="Exact Match"
    )

    if pivot_table.empty:
        return

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(17.5, 8))

    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"])

    NoWords = sorted({i['Context Length'] for i in data}, key=lambda x: float(x[:-1]))
    NoDepths = sorted({i['Document Depth'] for i in data})

    heatmap = sns.heatmap(
        pivot_table,
        fmt="g",
        cbar=False,
        cmap=cmap,
        vmin=0,
        vmax=1,
        ax=ax
    )

    ax.set_xlabel('Word Count', fontsize=30)
    ax.set_ylabel('Depth', fontsize=30)

    ax.set_xticks([i + 0.5 for i in range(len(NoWords))])
    ax.set_xticklabels(NoWords, rotation=45, fontsize=30)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=30)
    ax.set_aspect('equal')

    # Add grid lines
    for i in range(len(NoWords) - 1):
        ax.axvline(i + 1, color='white', linewidth=2)
    for i in range(len(NoDepths) - 1):
        ax.axhline(i + 1, color='white', linewidth=2)

    # Add colorbar with minimal gap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(heatmap.collections[0], cax=cax)

    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == "__main__":
    generate_plot(args.output_dir)
