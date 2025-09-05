import os
import yaml
import pandas as pd
from physiojepa.bedside import ForecastingDataset, CLIP_INTERPOLATE_RANGES
from pathlib import Path
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch
from datetime import datetime

from torch.utils.data import WeightedRandomSampler

from physiojepa.baselines import FCN, GeneralTimeSupervised
from sklearn.model_selection import StratifiedGroupKFold
from physiojepa.augmentations import MixupCallbackClassification, TransformsCallback, channel_masking, jitter_augmentation

from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import AUROC, Accuracy, AveragePrecision
from functools import partial

torch.set_float32_matmul_precision('high')

config_path = 'fcn_baseline_hypotension.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model = FCN(**config['arch'])

forecast_window_sec = config['dataset']['forecast_window_sec']
if isinstance(forecast_window_sec, int):
    forecast_window_sec = [forecast_window_sec]

n_classes = config['arch']['n_classes']
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
win_length = config['dataset']['win_length'] # every ten seconds for patches
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

y_padding_mask = config['dataset']['y_padding_mask']
constant_nan_tolerance = config['dataset']['constant_nan_tolerance']
require_all_channels = config['dataset']['require_all_channels']
infer_forecast_windows = config['dataset']['infer_forecast_windows']
 

name = config['run_config']['name']
filename = f"{name}" + "{epoch:02d}-Focal:{val_loss:.5f}-CE:{val_ce_loss:.5f}"
date_str = datetime.now().strftime("%Y-%m-%d")
sub_dir = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}"
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="val_loss", mode='min', filename=filename)
checkpoint_callback2 = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="train_loss", mode='min', filename=filename, save_last=True)
checkpoint_callback3 = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=1, monitor="val_auprc_0" if n_labels > 1 else "val_auprc", mode='max', filename=filename)
checkpoints = [checkpoint_callback, checkpoint_callback2, checkpoint_callback3]
wandb_project = config['run_config']['wandb_project']
wandb_name = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}-{config['run_config']['name']}"
os.makedirs(os.path.join(models_dir, sub_dir), exist_ok=True)

wandb_logger = WandbLogger(project=f"{wandb_project}", offline=False, name=wandb_name, save_dir=os.path.join(models_dir, sub_dir))
wandb_logger.log_hyperparams(config)

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
   
    patchmeupe2e_model = GeneralTimeSupervised(supervised_model=model,
                 learning_rate=config['optimizer']['learning_rate'],
                 train_size=len(train_ds),
                 batch_size=BATCHSIZE,
                 n_gpus=config['training']['n_gpus'],
                 n_classes=1, # this should be 1 for binary classification / multilabel classification
                 n_labels=1 if n_classes != 1 else n_labels, # n labels is the number of forecast windows, which corresponds to levels in the hierarchical inception time architecture
                 metrics=metrics, # name:function for metrics to log
                 loss_fxn=config['training']['loss_fxn'], # loss function to use, can be CrossEntropy or FocalLoss
                 gamma=config['training']['gamma'], # for focal loss
                 class_weights=class_weights, # weights of classes to use in CE loss fxn
                 label_smoothing=config['training']['label_smoothing'], # label smoothing for cross entropy loss
                 y_padding_mask=y_padding_mask, # padded value that was added to target and indice to ignore when computing loss
                 epochs=config['training']['epochs'], # number of epochs for one_cycle_scheduler
                 optimizer_type=config['optimizer']['optimizer_type'],
                 scheduler_type=config['scheduler']['scheduler_type'],
                 weight_decay=config['optimizer']['weight_decay'],
                 use_weight_decay_scheduler=config['optimizer']['use_weight_decay_scheduler'],
                 final_weight_decay=config['optimizer']['final_weight_decay'],
                 transforms=transforms_callback,
                 mixup_callback=mixup_callback,
                 scheduler_kwargs=scheduler_kwargs,
                 )
    
    trainer = pl.Trainer(precision=config['run_config']['precision'],
                        enable_checkpointing=True, 
                        enable_progress_bar=True, 
                        enable_model_summary=True, 
                        logger=wandb_logger,
                        val_check_interval=config['training']['val_check_interval'],
                        log_every_n_steps=50,
                        num_sanity_val_steps=2, # speed up
                        detect_anomaly=False, # speed up, though defualt
                        profiler=None, # this is the default, not me
                        strategy="ddp",
                        gradient_clip_val=config['training']['gradient_clip_val'],
                        gradient_clip_algorithm='norm' if config['training']['use_gradient_clipping'] else None,
                        accelerator="gpu", 
                        devices=config['training']['n_gpus'], 
                        default_root_dir=os.path.join(models_dir, sub_dir), 
                        max_epochs=config['training']['epochs'], 
                        fast_dev_run=False,
                        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
                        sync_batchnorm=True, # added just in case we switch to use more GPUs
                        callbacks=checkpoints
                        )


    trainer.fit(model=patchmeupe2e_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=None)
    del trainer
    #Predictions and targets
    trainer2 = pl.Trainer(precision=config['run_config']['precision'],
                        enable_checkpointing=False, 
                        enable_progress_bar=True, 
                        enable_model_summary=False, 
                        logger=None,
                        val_check_interval=config['training']['val_check_interval'],
                        log_every_n_steps=None,
                        num_sanity_val_steps=0, # speed up
                        detect_anomaly=False, # speed up, though defualt
                        profiler=None, # this is the default, not me
                        strategy="ddp",
                        gradient_clip_val=config['training']['gradient_clip_val'],
                        gradient_clip_algorithm='norm' if config['training']['use_gradient_clipping'] else None,
                        accelerator="gpu", 
                        devices=1, # only 1 device for inference
                        default_root_dir=os.path.join(models_dir, sub_dir), 
                        max_epochs=config['training']['epochs'], 
                        fast_dev_run=False,
                        sync_batchnorm=True, # added just in case we switch to use more GPUs
                        )
    
    best_model_path = checkpoint_callback3.best_model_path
    val_preds, val_targets = list(zip(*trainer2.predict(model=patchmeupe2e_model, dataloaders=val_loader, return_predictions=True, ckpt_path=best_model_path)))
    test_preds, test_targets = list(zip(*trainer2.predict(model=patchmeupe2e_model, dataloaders=test_loader, return_predictions=True, ckpt_path=best_model_path)))
    
    test_targets_cat = torch.cat(test_targets).cpu()
    test_preds_cat = torch.cat(test_preds).cpu()

    val_preds_cat = torch.cat(val_preds).cpu()
    val_targets_cat = torch.cat(val_targets).cpu()

    
    tensor_dict = {'val_targets': val_targets_cat, 'val_preds': val_preds_cat,
                   'test_targets': test_targets_cat, 'test_preds': test_preds_cat,
                   }
    torch.save(tensor_dict, os.path.join(models_dir, sub_dir, f'{config["dataset"]["y_outcome"]}-{config["run_config"]["name"]}-predictions.pt'))
    
    auroc_metric = AUROC(task='binary')
    avg_prec_metric = AveragePrecision(task='binary')
    accuracy_metric = Accuracy(task='binary')

    # print(auroc_scores)
    print("Val AUC", auroc_metric(val_preds_cat, val_targets_cat))
    print("Val AP", avg_prec_metric(val_preds_cat, val_targets_cat))
    print("Val Accuracy", accuracy_metric(val_preds_cat, val_targets_cat))

    print('Test AUC', auroc_metric(test_preds_cat, test_targets_cat))
    print('Test AP', avg_prec_metric(test_preds_cat, test_targets_cat))
    print('Test Accuracy', accuracy_metric(test_preds_cat, test_targets_cat))