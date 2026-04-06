from trainer.trainer import Trainer
from models.model_zoo import Conv2DGRU
from utils import get_all_song_names, create_kfold_splits, load_folds_from_files, load_stratified_folds_from_genre_files, load_half_stratified_folds
from losses import FocalLoss
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import torch
import hydra

_korean_font = fm.FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['font.family'] = _korean_font.get_name()


@hydra.main(config_path="./configs", config_name="config")
def main(cfg):
    device = cfg.device if torch.cuda.is_available() else 'cpu'
    midi_dir = cfg.data.dir.midi_dir
    label_path = cfg.data.dir.label_path

    T = datetime.now().strftime('%m%d_%H%M%S')

    if cfg.data.split == 'stratified':
        folds = load_stratified_folds_from_genre_files(
            cfg.data.dir.stratified_fold_dir, midi_dir, label_path,
            k=cfg.train.k_folds, seed=cfg.random_seed)
    elif cfg.data.split == 'stratified_half':
        folds = load_half_stratified_folds(
            cfg.data.dir.stratified_half_fold_dir, midi_dir, label_path)
    else:
        folds = load_folds_from_files(cfg.data.dir.random_fold_dir, midi_dir, label_path, k=cfg.train.k_folds)

    target_fold = cfg.train.get('fold', None)
    for fold_idx, fold in enumerate(folds):
        if target_fold is not None and (fold_idx + 1) != int(target_fold):
            continue

        model = Conv2DGRU(cfg.model).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

        loss_cfg = cfg.train.loss
        if loss_cfg.name == 'FocalLoss':
            alpha = torch.tensor(loss_cfg.weights).to(device) if loss_cfg.weights else 1
            criterion = FocalLoss(alpha=alpha, gamma=loss_cfg.gamma,
                                  ignore_index=loss_cfg.ignore_index, reduction='mean')
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=loss_cfg.ignore_index)

        trainer = Trainer(model, optimizer, criterion, device, cfg, fold_idx, T)
        trainer.run(fold)

if __name__ == '__main__':
    main()
