def masked_acc(preds, targets, other_class=2):
    mask = targets != other_class
    if mask.sum() == 0:
        return 0.0
    return (preds[mask] == targets[mask]).float().mean().item()

def masked_f1(preds, targets, other_class=2):
    """Per-class and macro F1 for 우조/계면조, excluding others."""
    mask = targets != other_class
    if mask.sum() == 0:
        return {'f1_ujoh': 0.0, 'f1_gyemyeon': 0.0, 'f1_macro': 0.0}

    p = preds[mask]
    t = targets[mask]

    results = {}
    for cls, name in [(0, 'f1_ujoh'), (1, 'f1_gyemyeon')]:
        tp = ((p == cls) & (t == cls)).sum().float()
        fp = ((p == cls) & (t != cls)).sum().float()
        fn = ((p != cls) & (t == cls)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)
        results[name] = (2 * precision * recall / (precision + recall + 1e-8)).item()

    results['f1_macro'] = (results['f1_ujoh'] + results['f1_gyemyeon']) / 2
    return results



