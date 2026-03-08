from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
import numpy as np

def multilabel_metrics(y_true, y_pred, average_methods=["micro", "macro", "weighted", "samples"]):
    """
    Compute multilabel precision, recall, f1-score for different averaging methods.
    Returns a dict of dicts: metrics[metric][average] = value
    """
    metrics = {"precision": {}, "recall": {}, "f1": {}}
    for avg in average_methods:
        metrics["precision"][avg] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        metrics["recall"][avg] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        metrics["f1"][avg] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    return metrics

def hamming_loss_metric(y_true, y_pred):
    """
    Compute the Hamming loss for multilabel classification.
    """
    return hamming_loss(y_true, y_pred)

def per_class_f1(y_true, y_pred):
    """
    Return the F1-score for each class/tag (no averaging).
    """
    return f1_score(y_true, y_pred, average=None)

def print_metrics_report(y_true, y_pred, tag_names=None):
    metrics = multilabel_metrics(y_true, y_pred)
    print("Multilabel metrics:")
    for metric, scores in metrics.items():
        for avg, val in scores.items():
            print(f"{metric} ({avg}): {val:.4f}")
    print(f"Hamming loss: {hamming_loss_metric(y_true, y_pred):.4f}")
    f1s = per_class_f1(y_true, y_pred)
    if tag_names is not None:
        for i, f1_val in enumerate(f1s):
            print(f"F1-score for {tag_names[i]}: {f1_val:.4f}")
    else:
        print("F1-score per class:", f1s)
