import json
import argparse
from typing import List, Dict, Any, Tuple

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Example structure for ground truth and predictions
# ground_truth.json: [{"id": 1, "relevant": [0, 2, 5]}, ...]
# predictions.json: [{"id": 1, "predicted": [0, 3, 5]}, ...]


def load_annotations(path: str) -> Dict[int, List[int]]:
    """
    Load annotation file mapping document IDs to list of relevant chunk IDs.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return {item['id']: item['relevant'] for item in data}


def load_predictions(path: str) -> Dict[int, List[int]]:
    """
    Load predictions file mapping document IDs to list of predicted chunk IDs.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    return {item['id']: item['predicted'] for item in data}


def compute_metrics(
    ground_truth: Dict[int, List[int]], predictions: Dict[int, List[int]]
) -> Tuple[float, float, float]:
    """
    Compute macro-averaged precision, recall, and F1 score over the dataset.
    """
    y_true = []
    y_pred = []

    for doc_id, relevant in ground_truth.items():
        pred = predictions.get(doc_id, [])
        # Convert to binary relevance vectors
        all_ids = sorted(set(relevant + pred))
        true_vec = [1 if idx in relevant else 0 for idx in all_ids]
        pred_vec = [1 if idx in pred else 0 for idx in all_ids]

        y_true.extend(true_vec)
        y_pred.extend(pred_vec)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance.")
    parser.add_argument(
        "--ground-truth", type=str, required=True,
        help="Path to ground truth JSON file."
    )
    parser.add_argument(
        "--predictions", type=str, required=True,
        help="Path to predictions JSON file."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to write metrics as JSON."
    )
    args = parser.parse_args()

    gt = load_annotations(args.ground_truth)
    pred = load_predictions(args.predictions)
    precision, recall, f1 = compute_metrics(gt, pred)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    print(json.dumps(metrics, indent=2))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metrics, f)
        print(f"Metrics written to {args.output}")


if __name__ == '__main__':
    main()
