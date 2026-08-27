import os
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="output/longbench/")
args = parser.parse_args()

DATA_PATH = args.output_dir
DATASETS = [
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "gov_report",
    "qmsum",
    "multi_news",
    "trec",
    "triviaqa",
    "samsum",
    "passage_retrieval_en",
    "lcc",
    "repobench-p",
    "passage_count",
]
TASK_DATASETS = {
    "single_doc_qa": ["narrativeqa", "qasper", "multifieldqa_en"],
    "multi_doc_qa": ["hotpotqa", "2wikimqa", "musique"],
    "summarization": ["gov_report", "qmsum", "multi_news"],
    "few_shots": ["trec", "triviaqa", "samsum"],
    "synthetic": ["passage_count", "passage_retrieval_en"],
    "code": ["lcc", "repobench-p"],
}


def get_task_results():
    results = {}
    ind_dataset_result = {}
    task_ave_result = {}

    NA_flag = False
    # Get individual dataset result
    for dataset in DATASETS:
        file_name = os.path.join(DATA_PATH, dataset, "output_config.json")
        if os.path.isfile(file_name):
            with open(file_name, "r") as f:
                result = json.load(f)
                result = result["eval_results"]["processed_results"]
                key = list(result.keys())[0]
                val = result[key]
                ind_dataset_result[dataset] = val
        else:
            ind_dataset_result[dataset] = "N/A"
            NA_flag = True

    results["individual_dataset_result"] = ind_dataset_result

    # Tab-separated row for pasting into spreadsheets (order matches DATASETS)
    tsv_cells = []
    for dataset in DATASETS:
        v = ind_dataset_result[dataset]
        if v == "N/A":
            tsv_cells.append("N/A")
        else:
            tsv_cells.append(str(np.round(float(v), decimals=2)))
    results["individual_result"] = " ".join(tsv_cells)

    # Get task-average dataset result
    for task, datasets in TASK_DATASETS.items():
        task_NA_flag = False
        task_ave_result[task] = 0
        for dataset in datasets:
            if ind_dataset_result[dataset] != "N/A":
                task_ave_result[task] += ind_dataset_result[dataset]
            else:
                task_NA_flag = True
        if task_NA_flag:
            task_ave_result[task] = "N/A"
        else:
            task_ave_result[task] = np.round(task_ave_result[task] / len(datasets), decimals=2)

    results["task_average_result"] = task_ave_result

    # Tab-separated task averages (order matches TASK_DATASETS keys)
    task_tsv_cells = []
    for task in TASK_DATASETS:
        v = task_ave_result[task]
        if v == "N/A":
            task_tsv_cells.append("N/A")
        else:
            task_tsv_cells.append(str(np.round(float(v), decimals=2)))
    results["task_average_result_row"] = " ".join(task_tsv_cells)

    # Get overall average result
    if NA_flag:
        results["LB_average_result"] = "N/A"
    else:
        average_result = 0
        for dataset in DATASETS:
            if dataset != "passage_count":
                average_result += ind_dataset_result[dataset]
        results["LB_average_result"] = np.round(average_result / (len(DATASETS) - 1), decimals=2)

    # Save result
    output_result_path = os.path.join(DATA_PATH, "longbench_result_summary.json")
    with open(output_result_path, "w+") as output_file:
        json.dump(results, output_file, indent=4)
        print(f"Complete writing task summary to {output_result_path}")


get_task_results()
