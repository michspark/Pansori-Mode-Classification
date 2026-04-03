import torch
import wandb
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .metrics import masked_acc, masked_f1
from datasets.dataset import BaseDataset
from utils import plot_confusion_matrix, save_test_csv, plot_posteriorgram

OUTPUT_DIR = Path("/home/sangheon/Desktop/PansoriMIDIDetection/outputs")

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

            preds = out.detach().argmax(dim=-1).view(-1).cpu()
            total_loss += loss.item()
            all_preds.append(preds)
            all_tgts.append(tgt.view(-1).cpu())

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
            preds = out.detach().argmax(dim=-1).view(-1).cpu()
            all_preds.append(preds)
            all_tgts.append(tgt.view(-1).cpu())

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
                'f1_changjo': round(seg_f1['f1_changjo'], 6),
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

def train_step(batch, model, optimizer, criterion, device):
    model.train()
    _, start_frame, piano, label = batch
    piano = piano.to(device)
    label = label.to(device)
    optimizer.zero_grad()
    out = model(piano)
    tgt = label.argmax(dim=-1)
    loss = criterion(out.permute(0, 2, 1), tgt)
    loss.backward()
    optimizer.step()
    preds = out.detach().argmax(dim=-1).view(-1).cpu()
    acc = masked_acc(preds, tgt.view(-1).cpu())
    f1 = masked_f1(preds, tgt.view(-1).cpu())

    return loss.item(), acc, f1


class Trainer:
    def __init__(self, model, optimizer, criterion, device, cfg, fold_idx, T):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.cfg = cfg
        self.fold_idx = fold_idx
        self.T = T
        self.save_path = f"best_model_fold{fold_idx + 1}.pt"
        self.fold_out_dir = OUTPUT_DIR / f"fold{fold_idx + 1}_{T}"

    def run(self, fold):
        fs = self.cfg.data.fs
        window_size = self.cfg.data.window_size
        midi_dir = self.cfg.data.dir.midi_dir
        label_path = self.cfg.data.dir.label_path

        train_dataset = BaseDataset(midi_dir, label_path, song_list=fold['train'], fs=fs, window_size=window_size, is_train=True)
        val_dataset   = BaseDataset(midi_dir, label_path, song_list=fold['val'],   fs=fs, window_size=window_size, is_train=False)
        test_dataset  = BaseDataset(midi_dir, label_path, song_list=fold['test'],  fs=fs, window_size=window_size, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=self.cfg.train.batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False)
        test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=False)

        run = wandb.init(
            project=self.cfg.project_name,
            name=f"fold{self.fold_idx + 1}_{self.T}",
            config=OmegaConf.to_container(self.cfg, resolve=True),
            reinit=True,
        )

        self._fit(train_loader, val_loader)
        self._evaluate(test_loader)
        run.finish()

    def _fit(self, train_loader, val_loader):
        train_mode = self.cfg.train.get('train_mode', 'iteration')
        if train_mode == 'epoch':
            self._fit_epoch(train_loader, val_loader)
        else:
            self._fit_iteration(train_loader, val_loader)

    def _fit_iteration(self, train_loader, val_loader):
        num_iterations          = self.cfg.train.num_iterations
        eval_interval           = self.cfg.train.get('iter_eval_interval', self.cfg.train.get('eval_interval', 200))
        save_interval           = self.cfg.train.get('save_interval', 1000)
        early_stopping_patience = self.cfg.train.get('early_stopping_patience', None)

        best_val_f1      = 0.0
        best_step        = 0
        patience_counter = 0
        global_step      = 0
        train_iter       = iter(train_loader)

        pbar = tqdm(total=num_iterations, desc=f"Fold {self.fold_idx + 1}")

        while global_step < num_iterations:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            train_loss, train_acc, train_f1 = train_step(batch, self.model, self.optimizer, self.criterion, self.device)
            wandb.log({
                'train/loss':        train_loss,
                'train/acc':         train_acc,
                'train/f1_macro':    train_f1['f1_macro'],
                'train/f1_ujoh':     train_f1['f1_ujoh'],
                'train/f1_gyemyeon': train_f1['f1_gyemyeon'],
                'train/f1_aniri':    train_f1['f1_aniri'],
                'train/f1_changjo':  train_f1['f1_changjo'],
            }, step=global_step)

            global_step += 1
            pbar.update(1)

            if global_step % eval_interval == 0:
                val_loss, val_acc, val_f1, val_song_data = run_epoch(
                    val_loader, self.model, self.optimizer, self.criterion, self.device, train=False)
                val_cm = plot_confusion_matrix(val_song_data)
                wandb.log({
                    'val/loss':             val_loss,
                    'val/acc':              val_acc,
                    'val/f1_macro':         val_f1['f1_macro'],
                    'val/f1_ujoh':          val_f1['f1_ujoh'],
                    'val/f1_gyemyeon':      val_f1['f1_gyemyeon'],
                    'val/f1_aniri':         val_f1['f1_aniri'],
                    'val/f1_changjo':       val_f1['f1_changjo'],
                    'val/confusion_matrix': wandb.Image(Image.fromarray(val_cm)),
                }, step=global_step)

                marker = ''
                if val_f1['f1_macro'] > best_val_f1:
                    best_val_f1      = val_f1['f1_macro']
                    best_step        = global_step
                    patience_counter = 0
                    torch.save(self.model.state_dict(), self.save_path)
                    marker = '  ← best'
                else:
                    patience_counter += 1
                    if early_stopping_patience and patience_counter >= early_stopping_patience:
                        print(f"Early stopping at step {global_step} (no improvement for {patience_counter} evals)")
                        break

                print(f"Step {global_step}/{num_iterations} | "
                      f"Train loss {train_loss:.4f}  acc {train_acc:.3f}  f1 {train_f1['f1_macro']:.3f} | "
                      f"Val loss {val_loss:.4f}  acc {val_acc:.3f}  f1 {val_f1['f1_macro']:.3f} "
                      f"[우조 {val_f1['f1_ujoh']:.3f} / 계면조 {val_f1['f1_gyemyeon']:.3f} / 아니리 {val_f1['f1_aniri']:.3f} / 창조 {val_f1['f1_changjo']:.3f}]{marker}")
                pbar.set_description(f"Fold {self.fold_idx + 1} | best f1 {best_val_f1:.3f}")

            if global_step % save_interval == 0:
                ckpt_path = (str(self.fold_out_dir / f"step{global_step}.pt")
                             if self.fold_out_dir.exists()
                             else f"fold{self.fold_idx + 1}_step{global_step}.pt")
                torch.save(self.model.state_dict(), ckpt_path)

        pbar.close()
        print(f"Fold {self.fold_idx + 1} best val f1: {best_val_f1:.4f} at step {best_step}")

    def _fit_epoch(self, train_loader, val_loader):
        num_epochs              = self.cfg.train.num_epoch
        eval_interval           = self.cfg.train.get('epoch_eval_interval', self.cfg.train.get('eval_interval', 1))
        early_stopping_patience = self.cfg.train.get('early_stopping_patience', None)

        best_val_f1      = 0.0
        best_epoch       = 0
        patience_counter = 0

        pbar = tqdm(total=num_epochs, desc=f"Fold {self.fold_idx + 1}")

        for epoch in range(num_epochs):
            train_loss, train_acc, train_f1, _ = run_epoch(
                train_loader, self.model, self.optimizer, self.criterion, self.device, train=True)
            wandb.log({
                'train/loss':        train_loss,
                'train/acc':         train_acc,
                'train/f1_macro':    train_f1['f1_macro'],
                'train/f1_ujoh':     train_f1['f1_ujoh'],
                'train/f1_gyemyeon': train_f1['f1_gyemyeon'],
                'train/f1_aniri':    train_f1['f1_aniri'],
                'train/f1_changjo':  train_f1['f1_changjo'],
            }, step=epoch)

            pbar.update(1)

            val_loss, val_acc, val_f1, val_song_data = run_epoch(
                val_loader, self.model, self.optimizer, self.criterion, self.device, train=False)
            val_cm = plot_confusion_matrix(val_song_data)
            wandb.log({
                'val/loss':             val_loss,
                'val/acc':              val_acc,
                'val/f1_macro':         val_f1['f1_macro'],
                'val/f1_ujoh':          val_f1['f1_ujoh'],
                'val/f1_gyemyeon':      val_f1['f1_gyemyeon'],
                'val/f1_aniri':         val_f1['f1_aniri'],
                'val/f1_changjo':       val_f1['f1_changjo'],
                'val/confusion_matrix': wandb.Image(Image.fromarray(val_cm)),
            }, step=epoch)

            marker = ''
            if val_f1['f1_macro'] > best_val_f1:
                best_val_f1      = val_f1['f1_macro']
                best_epoch       = epoch + 1
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                marker = '  ← best'
            else:
                patience_counter += 1
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1} (no improvement for {patience_counter} evals)")
                    break

            print(f"Epoch {epoch + 1}/{num_epochs} | "
                    f"Train loss {train_loss:.4f}  acc {train_acc:.3f}  f1 {train_f1['f1_macro']:.3f} | "
                    f"Val loss {val_loss:.4f}  acc {val_acc:.3f}  f1 {val_f1['f1_macro']:.3f} "
                    f"[우조 {val_f1['f1_ujoh']:.3f} / 계면조 {val_f1['f1_gyemyeon']:.3f} / 아니리 {val_f1['f1_aniri']:.3f} / 창조 {val_f1['f1_changjo']:.3f}]{marker}")
            pbar.set_description(f"Fold {self.fold_idx + 1} | best f1 {best_val_f1:.3f}")

        pbar.close()
        print(f"Fold {self.fold_idx + 1} best val f1: {best_val_f1:.4f} at epoch {best_epoch}")

    def _evaluate(self, test_loader):
        state_dict = torch.load(self.save_path, map_location='cpu')
        self.model.load_state_dict(state_dict)

        test_loss, test_acc, test_f1, song_data, segment_results = run_test_epoch(
            test_loader, self.model, self.criterion, self.device,
            fs=self.cfg.data.fs, window_size=int(self.cfg.data.window_size * self.cfg.data.fs))

        print(f"\nFold {self.fold_idx + 1} Test | acc {test_acc:.4f}  f1_macro {test_f1['f1_macro']:.4f} "
              f"[우조 {test_f1['f1_ujoh']:.4f} / 계면조 {test_f1['f1_gyemyeon']:.4f} / 아니리 {test_f1['f1_aniri']:.4f} / 창조 {test_f1['f1_changjo']:.4f}]")
        print(f"총 곡 수: {len(song_data)}")
        print(f"총 segment 수: {sum(len(d['segments']) for d in song_data.values())}")

        test_cm = plot_confusion_matrix(song_data)
        wandb.log({
            'test/loss':             test_loss,
            'test/acc':              test_acc,
            'test/f1_macro':         test_f1['f1_macro'],
            'test/f1_ujoh':          test_f1['f1_ujoh'],
            'test/f1_gyemyeon':      test_f1['f1_gyemyeon'],
            'test/f1_aniri':         test_f1['f1_aniri'],
            'test/f1_changjo':       test_f1['f1_changjo'],
            'test/confusion_matrix': wandb.Image(Image.fromarray(test_cm)),
        })

        self.fold_out_dir.mkdir(parents=True, exist_ok=True)
        save_test_csv(segment_results, self.fold_out_dir / "test_results.csv")
        self._save_plots(song_data)

    def _save_plots(self, song_data):
        for sname, data in song_data.items():
            stem = Path(sname).stem

            for seg in data['segments']:
                seg_label = f"{seg['start_sec']:.0f}-{seg['end_sec']:.0f}s"
                fig = plot_posteriorgram(f"{stem} [{seg_label}]", seg['gt'], seg['pred_probs'])
                fig.savefig(self.fold_out_dir / f"{stem}_{seg_label}.png", dpi=120, bbox_inches='tight')
                fig.clf()
                plt.close(fig)

            fig = plot_posteriorgram(sname, data['gt'], data['pred_probs'])
            fig.savefig(self.fold_out_dir / f"{stem}_full.png", dpi=120, bbox_inches='tight')
            fig.clf()
            plt.close(fig)

        plt.close('all')
