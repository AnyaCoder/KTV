import argparse
import json


def map_prediction_to_option(pred):
    pred_option = "none"
    if isinstance(pred, str):
        prediction_letter = pred[:1]
        if prediction_letter in "abcdefABCDEF":
            pred_option = prediction_letter.lower()
        if "A:" in pred or "A)" in pred:
            pred_option = "a"
        elif "B:" in pred or "B)" in pred:
            pred_option = "b"
        elif "C:" in pred or "C)" in pred:
            pred_option = "c"
        elif "D:" in pred or "D)" in pred:
            pred_option = "d"
        elif "E:" in pred or "E)" in pred:
            pred_option = "e"
        elif "F:" in pred or "F)" in pred:
            pred_option = "f"
    return pred_option


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", required=True)
    args = parser.parse_args()

    with open(args.pred_path, "r") as f:
        preds = [json.loads(line) for line in f if line.strip()]

    task_accuracy = {}
    for item in preds:
        task_name = item["task_name"]
        task_accuracy.setdefault(task_name, {"yes_count": 0, "no_count": 0})
        gt = chr(ord("a") + int(item["answer_number"]))
        pred = map_prediction_to_option(item["pred"])
        if pred == gt:
            task_accuracy[task_name]["yes_count"] += 1
        else:
            task_accuracy[task_name]["no_count"] += 1

    for task_name, stat in task_accuracy.items():
        total = stat["yes_count"] + stat["no_count"]
        acc = stat["yes_count"] / total if total else 0.0
        print(task_name)
        print(f"  yes: {stat['yes_count']}")
        print(f"  no: {stat['no_count']}")
        print(f"  acc: {acc:.4f}")


if __name__ == "__main__":
    main()
