def masked_acc(preds, targets, no_label_class=0):
    mask = targets != no_label_class
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == targets[mask]).float().mean().item()

def masked_f1(preds, targets, no_label_class=0):
    """Per-class and macro F1 for 우조계열/계면조/아니리계열, excluding no-label frames."""
    mask = targets != no_label_class
    if mask.sum() == 0:
        return {'f1_ujoh': 0.0, 'f1_gyemyeon': 0.0, 'f1_aniri': 0.0, 'f1_macro': 0.0}

    p = preds[mask]
    t = targets[mask]

    results = {}
    for cls, name in [(1, 'f1_ujoh'), (2, 'f1_gyemyeon'), (3, 'f1_aniri')]:
        tp = ((p == cls) & (t == cls)).sum().float()
        fp = ((p == cls) & (t != cls)).sum().float()
        fn = ((p != cls) & (t == cls)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        results[name] = (2 * precision * recall / (precision + recall + 1e-8)).item()

    results['f1_macro'] = (results['f1_ujoh'] + results['f1_gyemyeon'] + results['f1_aniri']) / 3
    return results



