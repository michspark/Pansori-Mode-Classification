import torch
from tqdm import tqdm
from .metrics import masked_acc, masked_f1

def run_test_epoch(loader, model, criterion, device, fs=100, window_size=3000):
    """run_epoch(train=False) but also returns per-song GT, softmax probs, and per-segment metrics."""
    model.eval()
    total_loss = 0.0
    all_preds, all_tgts = [], []
    song_data = {}  # song_name -> {'gt': [Tensor(T,3)], 'pred_probs': [Tensor(T,3)]}
    segment_results = []  # per-segment: song_name, time range, loss, acc, f1s

    with torch.no_grad():
        pbar = tqdm(loader, desc='Test', leave=False)
        for song_name, start_frame, piano, label in pbar:
            piano = piano.to(device)
            label = label.to(device)
            out = model(piano)          # (1, T, C)
            tgt = label.argmax(dim=-1)  # (1, T)

            loss = criterion(out.permute(0, 2, 1), tgt)
            total_loss += loss.item()

            pred_probs = torch.softmax(out, dim=-1)  # (1, T, C)
            preds = out.detach().argmax(dim=-1).view(-1)
            all_preds.append(preds)
            all_tgts.append(tgt.view(-1))

            name = song_name[0] if isinstance(song_name, (list, tuple)) else song_name
            if name not in song_data:
                song_data[name] = {'gt': [], 'pred_probs': [], 'segments': []}
            song_data[name]['gt'].append(label[0].cpu())
            song_data[name]['pred_probs'].append(pred_probs[0].cpu())

            # Per-segment metrics
            seg_start = start_frame.item() if hasattr(start_frame, 'item') else int(start_frame)
            seg_start_sec = seg_start / fs
            seg_end_sec = seg_start_sec + window_size / fs
            song_data[name]['segments'].append({
                'gt': label[0].cpu().numpy(),
                'pred_probs': pred_probs[0].cpu().numpy(),
                'start_sec': seg_start_sec,
                'end_sec': seg_end_sec,
            })
            seg_f1 = masked_f1(preds.cpu(), tgt.view(-1).cpu())
            segment_results.append({
                'song_name': name,
                'start_sec': f"{seg_start_sec:.1f}",
                'end_sec': f"{seg_end_sec:.1f}",
                'time_range': f"{seg_start_sec:.0f}-{seg_end_sec:.0f}s",
                'loss': round(loss.item(), 6),
                'acc': round(masked_acc(preds.cpu(), tgt.view(-1).cpu()), 6),
                'f1_ujoh': round(seg_f1['f1_ujoh'], 6),
                'f1_gyemyeon': round(seg_f1['f1_gyemyeon'], 6),
                'f1_aniri': round(seg_f1['f1_aniri'], 6),
                'f1_macro': round(seg_f1['f1_macro'], 6),
            })

    for name in song_data:
        song_data[name]['gt'] = torch.cat(song_data[name]['gt'], dim=0).numpy()
        song_data[name]['pred_probs'] = torch.cat(song_data[name]['pred_probs'], dim=0).numpy()

    all_preds = torch.cat(all_preds)
    all_tgts  = torch.cat(all_tgts)
    avg_loss  = total_loss / len(loader)
    avg_acc   = masked_acc(all_preds, all_tgts)
    f1        = masked_f1(all_preds, all_tgts)

    return avg_loss, avg_acc, f1, song_data, segment_results


def run_epoch(loader, model, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_tgts = [], []
    song_data = {} if not train else None

    ctx = torch.enable_grad() if train else torch.no_grad()
    desc = 'Train' if train else 'Val'

    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False)
        for song_name, start_frame, piano, label in pbar:
            piano = piano.to(device)
            label = label.to(device)

            if train:
                optimizer.zero_grad()

            out = model(piano)
            tgt = label.argmax(dim=-1)

            loss = criterion(out.permute(0,2,1), tgt)

            if train:
                loss.backward()
                optimizer.step()

            preds = out.detach().argmax(dim=-1).view(-1)
            total_loss += loss.item()
            all_preds.append(preds)
            all_tgts.append(tgt.view(-1))

            if not train:
                pred_probs = torch.softmax(out, dim=-1)
                name = song_name[0] if isinstance(song_name, (list, tuple)) else song_name
                if name not in song_data:
                    song_data[name] = {'gt': [], 'pred_probs': []}
                song_data[name]['gt'].append(label[0].cpu())
                song_data[name]['pred_probs'].append(pred_probs[0].cpu())

    if not train:
        for name in song_data:
            song_data[name]['gt'] = torch.cat(song_data[name]['gt'], dim=0).numpy()
            song_data[name]['pred_probs'] = torch.cat(song_data[name]['pred_probs'], dim=0).numpy()

    all_preds = torch.cat(all_preds)
    all_tgts  = torch.cat(all_tgts)

    avg_loss = total_loss / len(loader)
    avg_acc  = masked_acc(all_preds, all_tgts)
    f1       = masked_f1(all_preds, all_tgts)

    return avg_loss, avg_acc, f1, song_data





