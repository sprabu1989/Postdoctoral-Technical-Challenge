# task1_classification/metrics.py

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve


def compute_classification_metrics(labels, preds):
    return classification_report(labels, preds, digits=4)


def compute_confusion_matrix(labels, preds):
    return confusion_matrix(labels, preds)


def compute_auc(labels, probs):
    return roc_auc_score(labels, probs)


def compute_roc(labels, probs):
    return roc_curve(labels, probs)


def compute_pr(labels, probs):
    return precision_recall_curve(labels, probs)
