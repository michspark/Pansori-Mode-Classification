from sklearn.metrics import f1_score as sklearn_f1

def masked_acc(preds, targets, no_label_class=0):
    mask = targets != no_label_class
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == targets[mask]).float().mean().item()

def masked_f1(preds, targets, no_label_class=0):
    """Per-class and macro F1 for 우조계열/계면조/아니리/창조, excluding no-label frames."""
    mask = targets != no_label_class
    if mask.sum() == 0:
        return {'f1_ujoh': 0.0, 'f1_gyemyeon': 0.0, 'f1_aniri': 0.0, 'f1_changjo': 0.0, 'f1_macro': 0.0}

    p = preds[mask].numpy()
    t = targets[mask].numpy()

    labels = [1, 2, 3, 4]
    f1_scores = sklearn_f1(t, p, labels=labels, average=None, zero_division=0)
    macro_f1  = sklearn_f1(t, p, labels=labels, average='macro', zero_division=0)

    return {
        'f1_ujoh':     float(f1_scores[0]),
        'f1_gyemyeon': float(f1_scores[1]),
        'f1_aniri':    float(f1_scores[2]),
        'f1_changjo':  float(f1_scores[3]),
        'f1_macro':    float(macro_f1),
    }



