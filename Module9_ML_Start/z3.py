def calculate_metrics(y_true, y_pred):
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        elif yt == 0 and yp == 0:
            tn += 1
    metrics = dict()

    metrics["accuracy"] = ((tp + tn) / (tp + tn + fp + fn))
    metrics["precision"] = (tp / (tp + fn))
    metrics["recall"] = (tp / (tp + fp))
    return metrics