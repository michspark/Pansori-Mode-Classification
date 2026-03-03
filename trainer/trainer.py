import torch
from tqdm import tqdm
from .metrics import masked_acc, masked_f1

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
            out = model(piano)
            tgt = label.argmax(dim=-1)

            loss = criterion(out.permute(0,2,1), tgt)

            if train:
                optimizer.zero_grad()
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





