import os
import json

from datasets import load_dataset, load_from_disk
from .metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def load_data(config):
    dataset_path = config["dataset_path"]
    dataset_name = config["dataset"]

    # LongBench data in this repo is stored as json/jsonl files under dataset/longbench.
    candidate_files = [
        os.path.join(dataset_path, f"{dataset_name}.json"),
        os.path.join(dataset_path, f"{dataset_name}.jsonl"),
    ]

    for file_path in candidate_files:
        if os.path.isfile(file_path):
            return load_dataset("json", data_files=file_path, split="train")

    # Keep backward compatibility for users that still keep Arrow dataset folders.
    dataset_dir = os.path.join(dataset_path, dataset_name)
    if os.path.isdir(dataset_dir):
        return load_from_disk(dataset_dir)

    raise FileNotFoundError(
        f"LongBench dataset not found. Checked files: {candidate_files} and directory: {dataset_dir}"
    )


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    raw_results = []
    for prediction, ground_truths in zip(predictions, answers):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        total_score += score
        raw_results.append({"answers": prediction, "score": score})

    return round(100 * total_score / len(predictions), 2), raw_results


def eval(pred_dir, model, eval_params):
    scores = dict()
    path = os.path.join(pred_dir, "pred", f"{model}/")
    dataset = eval_params["dataset"]
    all_files = sorted(filename for filename in os.listdir(path) if filename.endswith("jsonl"))
    all_files = [filename for filename in all_files if filename.startswith(f"{dataset}_")]
    if len(all_files) != 1:
        raise ValueError(f"Expected exactly one prediction file for {dataset} in {path}, found: {all_files}")
    print("Evaluating on:", all_files)
    for filename in all_files:
        predictions, answers, lengths = [], [], []
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])

        score, raw_results = scorer(dataset, predictions, answers, all_classes)
        scores[dataset2metric[dataset].__name__] = score

    out_path = f"{pred_dir}/pred/{model}/result.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    return scores, raw_results
