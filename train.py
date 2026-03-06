from trainer.trainer import run_epoch, run_test_epoch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from models.model_zoo import Conv2DGRU
from datasets.dataset import BaseDataset
from utils import get_all_song_names, create_kfold_splits, plot_posteriorgram
from torch.utils.data import DataLoader
import torch
import wandb
import hydra
from omegaconf import OmegaConf
from datetime import datetime
from losses import FocalLoss


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    midi_dir = cfg.data.dir.midi_dir
    label_path = cfg.data.dir.label_path
    fs = cfg.data.fs
    window_size = cfg.data.window_size

    T = datetime.now().strftime('%m%d_%H%M%S')

    all_songs = get_all_song_names(midi_dir, label_path)
    folds = create_kfold_splits(all_songs, k=cfg.train.k_folds, seed=cfg.random_seed)

    for fold_idx, fold in enumerate(folds):
        run = wandb.init(
            project=cfg.project_name,
            name=f"fold{fold_idx + 1}_{T}",
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
        )

        train_dataset = BaseDataset(midi_dir, label_path, song_list = fold['train'], fs=fs, window_size=window_size, is_train=True)
        val_dataset = BaseDataset(midi_dir, label_path, song_list = fold['val'], fs=fs, window_size=window_size, is_train=False)
        test_dataset = BaseDataset(midi_dir, label_path, song_list = fold['test'], fs=fs, window_size=window_size, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = Conv2DGRU(cfg.model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
        #criterion = torch.nn.CrossEntropyLoss()
        criterion = FocalLoss()

        best_val_f1 = 0.0
        save_path = f"best_model_fold{fold_idx + 1}.pt"

        for epoch in range(1, cfg.train.num_epochs + 1):
            train_loss, train_acc, train_f1 = run_epoch(train_loader, model, optimizer, criterion, device, train=True)
            val_loss, val_acc, val_f1 = run_epoch(val_loader, model, optimizer, criterion, device, train=False)

            if val_f1['f1_macro'] > best_val_f1:
                best_val_f1 = val_f1['f1_macro']
                torch.save(model.state_dict(), save_path)
                marker = '  ← best'
            else:
                marker = ''

            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/acc': train_acc,
                'train/f1_macro': train_f1['f1_macro'],
                'train/f1_ujoh': train_f1['f1_ujoh'],
                'train/f1_gyemyeon': train_f1['f1_gyemyeon'],
                'val/loss': val_loss,
                'val/acc': val_acc,
                'val/f1_macro': val_f1['f1_macro'],
                'val/f1_ujoh': val_f1['f1_ujoh'],
                'val/f1_gyemyeon': val_f1['f1_gyemyeon'],
            })

            print(f"Epoch {epoch:3d}/{cfg.train.num_epochs} | "
                f"Train loss {train_loss:.4f}  acc {train_acc:.3f}  f1 {train_f1['f1_macro']:.3f} | "
                f"Val loss {val_loss:.4f}  acc {val_acc:.3f}  f1 {val_f1['f1_macro']:.3f} "
                f"[우조 {val_f1['f1_ujoh']:.3f} / 계면조 {val_f1['f1_gyemyeon']:.3f}]{marker}")

        # Evaluate on test set with posteriorgram visualization
        model.load_state_dict(torch.load(save_path))
        test_loss, test_acc, test_f1, song_data = run_test_epoch(
            test_loader, model, criterion, device)
        print(f"\nFold {fold_idx + 1} Test | acc {test_acc:.4f}  f1_macro {test_f1['f1_macro']:.4f} "
              f"[우조 {test_f1['f1_ujoh']:.4f} / 계면조 {test_f1['f1_gyemyeon']:.4f}]")

        log_dict = {
            'test/loss': test_loss,
            'test/acc': test_acc,
            'test/f1_macro': test_f1['f1_macro'],
            'test/f1_ujoh': test_f1['f1_ujoh'],
            'test/f1_gyemyeon': test_f1['f1_gyemyeon'],
        }

        # Log posteriorgram for each test song
        for song_name, data in song_data.items():
            fig = plot_posteriorgram(song_name, data['gt'], data['pred_probs'])
            log_dict[f'test/posteriorgram/{song_name}'] = wandb.Image(fig)
            plt.close(fig)

        wandb.log(log_dict)
        run.finish()

if __name__ == '__main__':
    main()
