import torch
from tqdm import tqdm
from .metrics import masked_acc, masked_f1

def run_test_epoch(loader, model, criterion, device):
    """Rrun_epoch(train=False) but also returns per-song GT and softmax probs for visualization."""
    model.eval()
    total_loss = 0.0
    all_preds, all_tgts = [], []
    song_data = {}  # song_name -> {'gt': [Tensor(T,3)], 'pred_probs': [Tensor(T,3)]}

    with torch.no_grad():
        pbar = tqdm(loader, desc='Test', leave=False)
        for song_name, piano, label in pbar:
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
                song_data[name] = {'gt': [], 'pred_probs': []}
            song_data[name]['gt'].append(label[0].cpu())           # (T, 3)
            song_data[name]['pred_probs'].append(pred_probs[0].cpu())  # (T, 3)

    for name in song_data:
        song_data[name]['gt'] = torch.cat(song_data[name]['gt'], dim=0).numpy()
        song_data[name]['pred_probs'] = torch.cat(song_data[name]['pred_probs'], dim=0).numpy()

    all_preds = torch.cat(all_preds)
    all_tgts  = torch.cat(all_tgts)
    avg_loss  = total_loss / len(loader)
    avg_acc   = masked_acc(all_preds, all_tgts)
    f1        = masked_f1(all_preds, all_tgts)

    return avg_loss, avg_acc, f1, song_data


def run_epoch(loader, model, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_preds, all_tgts = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    desc = 'Train' if train else 'Val'

    with ctx:
        pbar = tqdm(loader, desc=desc, leave=False)
        for _, piano, label in pbar:
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

    all_preds = torch.cat(all_preds)
    all_tgts  = torch.cat(all_tgts)

    avg_loss = total_loss / len(loader)
    avg_acc  = masked_acc(all_preds, all_tgts)
    f1       = masked_f1(all_preds, all_tgts)

    return avg_loss, avg_acc, f1





