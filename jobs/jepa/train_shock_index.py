import os
import pandas as pd
from physiojepa.bedside import ForecastingDataset, CLIP_INTERPOLATE_RANGES
from pathlib import Path
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch
import yaml
import torch.nn as nn
from lightning.pytorch.tuner import Tuner

from datetime import datetime
from physiojepa.train import PatchTFTSingleOutcomeLightning

from physiojepa.jepa import JEPASimpleLightning
from physiojepa.heads import AttentiveClassifier
from sklearn.model_selection import StratifiedGroupKFold
from functools import partial
from physiojepa.augmentations import MixupCallbackClassification, TransformsCallback, channel_masking, jitter_augmentation
from lightning.pytorch.callbacks import ModelCheckpoint

from torch.utils.data import WeightedRandomSampler

import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import AUROC, AveragePrecision

torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)

config_path = 'train_shock_index.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

random_state = config['run_config']['random_state']

encoder_dir = config['paths']['encoder_dir']
pretrained_encoder_path = os.path.join(encoder_dir, config['paths']['pretrained_encoder_path'])

encoder_model = JEPASimpleLightning.load_from_checkpoint(pretrained_encoder_path, map_location='cpu')

d_model = encoder_model.encoder.d_model 
num_heads = encoder_model.encoder.predictor_blocks[0].self_attn.num_heads

forecast_window_sec = config['dataset']['forecast_window_sec']
if isinstance(forecast_window_sec, int):
    forecast_window_sec = [forecast_window_sec]
infer_forecast_windows = config['dataset']['infer_forecast_windows']

n_classes = config['lp_head']['num_classes']
n_labels = len(forecast_window_sec)
if n_labels > 1:
    metrics = {"auroc":AUROC(task='multilabel', average='none', num_labels=n_labels),
            "auprc":AveragePrecision(task='multilabel', average='none', num_labels=n_labels),
            }
else:
    metrics = {"auroc":AUROC(task='binary'),
               "auprc":AveragePrecision(task='binary'),
            }
use_transforms = config['dataset']['use_transforms']

if use_transforms:
    transforms_callback = TransformsCallback(
        transforms=[
            partial(jitter_augmentation, mask_ratio=0.05, jitter_ratio=0.05, p=0.5), # jitter and 0 value timepoint masking
            partial(channel_masking, dim=1, p=0.1, specific_channels=None), # mask out channels with 0s
        ]
    )
else:
    transforms_callback = None

if config['training']['mixup']:
    mixup_callback = MixupCallbackClassification(num_classes=n_labels, mixup_alpha=config['training']['mixup_alpha'], ignore_index=2)
else:
    mixup_callback = None

BATCHSIZE = config['training']['batch_size']
num_workers = config['training']['num_workers']
data_dir = config['paths']['data_dir']
models_dir = config['paths']['models_dir']
y_outcome = config['dataset']['y_outcome']

max_lr = config['scheduler']['max_lr']
div_factor = config['scheduler']['div_factor']
final_div_factor = config['scheduler']['final_div_factor']
pct_start=config['scheduler']['pct_start']
anneal_strategy=config['scheduler']['anneal_strategy']
scheduler_kwargs = dict(max_lr=max_lr, div_factor=div_factor, final_div_factor=final_div_factor, pct_start=pct_start, anneal_strategy=anneal_strategy)

perform_cv=config['training']['perform_cv']

outcome_df_path = config['paths']['outcome_df_path']
outcome_df = pd.read_csv(outcome_df_path)
outcome_df['Time Stamp (seconds)'] = outcome_df['Time Stamp (seconds)'].round()

zarr_files = outcome_df.file_path.tolist()

groups = [Path(i).stem.split('-')[0] for i in zarr_files]
labels = outcome_df[config['dataset']['y_outcome']].values.tolist()

random_state = config['run_config']['random_state']
splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_state)
train_idxs, test_idxs = next(splitter.split(X=zarr_files,  y=labels, groups=groups))

train_zarrs = [zarr_files[i] for i in train_idxs]
test_zarrs = [zarr_files[i] for i in test_idxs]

train_labels = [labels[i] for i in train_idxs]

groups = [Path(i).stem.split('-')[0] for i in train_zarrs]

splitter2 = StratifiedGroupKFold(n_splits=10,  shuffle=True, random_state=random_state)
train_idxs, val_idxs = next(splitter2.split(X=train_zarrs, y=train_labels, groups=groups))

val_zarrs = [train_zarrs[i] for i in val_idxs]
train_zarrs = [train_zarrs[i] for i in train_idxs]

outcome_df_train = outcome_df.loc[outcome_df.file_path.isin(train_zarrs)].copy()
outcome_df_val = outcome_df.loc[outcome_df.file_path.isin(val_zarrs)].copy()
outcome_df_test = outcome_df.loc[outcome_df.file_path.isin(test_zarrs)].copy()

channels = config['dataset']['channels']
c_in = len(channels)


frequency = config['dataset']['frequency']

sample_seq_len_seconds = config['dataset']['sample_seq_len_seconds']
win_length = config['dataset']['win_length'] 
overlap = config['dataset']['overlap']
hop_length=win_length - int(overlap*win_length)
max_seq_len = sample_seq_len_seconds*frequency

dataset_filename = config['paths']['dataset_filename']
n_patches = (max(max_seq_len, win_length)-win_length) // hop_length + 1
if ((max_seq_len-win_length) % hop_length != 0):
    n_patches += 1
n_patches = int(n_patches)

if Path(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz')).exists():
    sample_df_train = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-train_samples.csv.gz'))
else:
    sample_df_train = None
if Path(os.path.join(models_dir, f'{dataset_filename}-val_samples.csv.gz')).exists():
    sample_df_val = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-val_samples.csv.gz'))
else:
    sample_df_val = None
if Path(os.path.join(models_dir, f'{dataset_filename}-test_samples.csv.gz')).exists():
    sample_df_test = pd.read_csv(os.path.join(models_dir, f'{dataset_filename}-test_samples.csv.gz'))
else:
    sample_df_test = None

constant_nan_tolerance = config['dataset']['constant_nan_tolerance']
y_padding_mask = config['dataset']['y_padding_mask']

lp_head = dict(embed_dim=d_model,
        num_heads=num_heads,
        mlp_ratio=config['lp_head']['mlp_ratio'],
        depth=config['lp_head']['depth'],
        c_in=c_in,
        norm_layer=nn.LayerNorm,
        init_std=config['lp_head']['init_std'],
        qkv_bias=config['lp_head']['qkv_bias'],
        num_classes=n_labels,
        complete_block=config['lp_head']['complete_block'],
        affine=config['lp_head']['affine'],
    )


lp_model = AttentiveClassifier(**lp_head)


name = config['run_config']['name']
filename = f"{name}" + "{epoch:02d}-AUC-{val_auroc:.5f}-PRC-{val_auprc:.5f}"
date_str = datetime.now().strftime("%Y-%m-%d")
sub_dir = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}"
wandb_project = config['run_config']['wandb_project']
wandb_name = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}-{config['run_config']['name']}"
os.makedirs(os.path.join(models_dir, sub_dir), exist_ok=True)


checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="val_loss", mode='min', filename=filename)
checkpoint_callback2 = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="train_loss", mode='min', filename=filename, save_last=True)
checkpoint_callback3 = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="val_auprc_0" if n_labels > 1 else "val_auprc", mode='max', filename=filename)
checkpoints = [checkpoint_callback, checkpoint_callback2, checkpoint_callback3]
wandb_logger = WandbLogger(project=wandb_project, offline=False, name=wandb_name, save_dir=os.path.join(models_dir, sub_dir))
wandb_logger.log_hyperparams({**dict(encoder_model.hparams), **config})


if __name__ == "__main__":
    pl.seed_everything(random_state)
    train_ds = ForecastingDataset(
                 channels=channels, 
                 forecast_window_sec = forecast_window_sec,
                 outcome_df = outcome_df_train,
                 outcome_df_outcome_col=y_outcome,
                 file_col='file_path',
                 y_date_column='date', # column indicating date of sample collection
                 outcome_df_seconds_since_column='Time Stamp (seconds)', # column indicating how many seconds since beginning of waveform
                 outcome_df_duration_column='event_length',
                 sample_df = sample_df_train,
                 sample_seq_len_sec=sample_seq_len_seconds, 
                 frequency=frequency, 
                 butterworth_filters=None,
                 median_filter_kernel_size=None,
                 clip_interpolations=CLIP_INTERPOLATE_RANGES,
                 constant_nan_tolerance=constant_nan_tolerance,
                 require_all_channels=True,
                 infer_forecast_windows=infer_forecast_windows,
                 normalize_signals=config['dataset']['normalize_signals']
    )
    
    label_weights = 1 / train_ds.sample_df[f'outcome_val_{forecast_window_sec[0]}sec'].value_counts(dropna=False, normalize=False)
    sample_weights = [label_weights[i] for i in train_ds.sample_df[f'outcome_val_{forecast_window_sec[0]}sec'].values.tolist()]
    weighted_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_ds), replacement=True)

    val_ds = ForecastingDataset(
                 channels=channels, 
                 forecast_window_sec = forecast_window_sec,
                 outcome_df = outcome_df_val,
                 outcome_df_outcome_col=y_outcome,
                 file_col='file_path',
                 y_date_column='date', # column indicating date of sample collection
                 outcome_df_seconds_since_column='Time Stamp (seconds)', # column indicating how many seconds since beginning of waveform
                 outcome_df_duration_column='event_length',
                 sample_df = sample_df_val,
                 sample_seq_len_sec=sample_seq_len_seconds, 
                 frequency=frequency, 
                 butterworth_filters=None,
                 median_filter_kernel_size=None,
                 clip_interpolations=CLIP_INTERPOLATE_RANGES,
                 constant_nan_tolerance=constant_nan_tolerance,
                 require_all_channels=True,
                 infer_forecast_windows=infer_forecast_windows,
                 normalize_signals=config['dataset']['normalize_signals']
    )

    test_ds = ForecastingDataset(
                    channels=channels, 
                    forecast_window_sec = forecast_window_sec,
                    outcome_df = outcome_df_test,
                    outcome_df_outcome_col=y_outcome,
                    file_col='file_path',
                    y_date_column='date', # column indicating date of sample collection
                    outcome_df_seconds_since_column='Time Stamp (seconds)', # column indicating how many seconds since beginning of waveform
                    outcome_df_duration_column='event_length',
                    sample_df=sample_df_test,
                    sample_seq_len_sec=sample_seq_len_seconds, 
                    frequency=frequency, 
                    butterworth_filters=None,
                    median_filter_kernel_size=None,
                    clip_interpolations=CLIP_INTERPOLATE_RANGES,
                    constant_nan_tolerance=constant_nan_tolerance,
                    require_all_channels=True,
                    infer_forecast_windows=infer_forecast_windows,
                    normalize_signals=config['dataset']['normalize_signals']
    )

    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, sampler=weighted_sampler, drop_last=True, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)


    if not Path(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz")).exists():
        train_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz"), compression='gzip', index=True)
    if not Path(os.path.join(models_dir, f"{dataset_filename}-val_samples.csv.gz")).exists():
        val_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-val_samples.csv.gz"), compression='gzip', index=True)
    if not Path(os.path.join(models_dir, f"{dataset_filename}-test_samples.csv.gz")).exists():
        test_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-test_samples.csv.gz"), compression='gzip', index=True)
    
    if config['training']['use_class_weights']:
        class_weights = []
        for forecast_sec in forecast_window_sec:
            class0 = train_ds.sample_df[f'outcome_val_{forecast_sec}sec'].value_counts(dropna=False, normalize=False).sort_index().values[0]
            class1 = train_ds.sample_df[f'outcome_val_{forecast_sec}sec'].value_counts(dropna=False, normalize=False).sort_index().values[1]
            pos_weight = class0 / class1
            class_weights.append(pos_weight)
        class_weights = torch.tensor(class_weights)
    else:
        class_weights = None
    
    patchmeupe2e_model = PatchTFTSingleOutcomeLightning(learning_rate=config['optimizer']['learning_rate'], 
                                    train_size=len(train_ds), 
                                    n_gpus=config['training']['n_gpus'],
                                    batch_size=BATCHSIZE,
                                    linear_probing_head=lp_model,
                                    preloaded_model=encoder_model, 
                                    metrics=metrics, 
                                    class_weights=class_weights,
                                    fine_tune=config['training']['fine_tune'],
                                    epochs=config['training']['epochs'],
                                    scheduler_type=config['scheduler']['scheduler_type'],
                                    optimizer_type=config['optimizer']['optimizer_type'],
                                    weight_decay=config['optimizer']['weight_decay'],
                                    use_weight_decay_scheduler=config['optimizer']['use_weight_decay_scheduler'],
                                    final_weight_decay=config['optimizer']['final_weight_decay'],
                                    scheduler_kwargs=scheduler_kwargs,
                                    mixup_callback=mixup_callback,
                                    transforms=transforms_callback)
    

    trainer = pl.Trainer(precision=config['run_config']['precision'],
                        enable_checkpointing=True, 
                        enable_progress_bar=True, 
                        enable_model_summary=True,
                        logger=wandb_logger,
                        val_check_interval=config['training']['val_check_interval'],
                        log_every_n_steps=50,
                        num_sanity_val_steps=2,
                        detect_anomaly=False,
                        profiler=None,
                        strategy="ddp",
                        gradient_clip_val=config['training']['gradient_clip_val'],
                        gradient_clip_algorithm='norm' if config['training']['use_gradient_clipping'] else None,
                        accelerator="gpu", 
                        devices=config['training']['n_gpus'], 
                        default_root_dir=models_dir, 
                        max_epochs=config['training']['epochs'], 
                        fast_dev_run=False,
                        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
                        sync_batchnorm=True, # added just in case we switch to use more GPUs
                        callbacks=checkpoints
                        )

    if config['training']['use_lr_finder']:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(patchmeupe2e_model, train_dataloaders=train_loader, update_attr=False, attr_name="max_lr") # this actually doesnt reassing... need to do below manyally
        new_lr = lr_finder.suggestion()
        patchmeupe2e_model.scheduler_kwargs['max_lr'] = new_lr
        print(f"Using max lr: {new_lr}")
    trainer.fit(model=patchmeupe2e_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=None)
    # Predictions and targets
    best_model_path = checkpoint_callback3.best_model_path
    val_preds,val_targets = list(zip(*trainer.predict(model=patchmeupe2e_model, dataloaders=val_loader, return_predictions=True, ckpt_path=best_model_path)))
    test_preds,test_targets = list(zip(*trainer.predict(model=patchmeupe2e_model, dataloaders=test_loader, return_predictions=True, ckpt_path=best_model_path)))


    val_preds_cat = torch.cat(val_preds)
    val_targets_cat = torch.cat(val_targets)
    test_preds_cat = torch.cat(test_preds)
    test_targets_cat = torch.cat(test_targets)

    tensor_dict = {'val_targets': val_targets_cat, 'val_preds': val_preds_cat,
                   'test_targets': test_targets_cat, 'test_preds': test_preds_cat,
                   }
    torch.save(tensor_dict, os.path.join(models_dir, sub_dir, f'{config["dataset"]["y_outcome"]}-{config["run_config"]["name"]}-predictions.pt'))