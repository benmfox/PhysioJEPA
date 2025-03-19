import os
import pandas as pd
from physio_jepa.bedside import ForecastingDataset, CLIP_INTERPOLATE_RANGES
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch

from physio_jepa.train import PatchTFTSingleOutcomeLightning, PatchTSJEPALightning
from physio_jepa.heads import AvgPatchLogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import AUROC, MulticlassAUROC, AveragePrecision, MulticlassAccuracy, BinaryAUROC, BinaryAccuracy

random_state = 12
torch.set_float32_matmul_precision('medium')

metrics = {"auroc":AUROC(task='binary')}

scheduler_type = 'onecycle'

wandb_offline = False

val_check_interval = 1.0
weight_decay = 1e-3
precision = '16-mixed'
loss_fxn = 'CrossEntropy'
fine_tune = False
label_smoothing = 0. # only works for ce loss
gamma = 2.

y_outcome = 'shock_index_class' #'shock_index_class or "hypotension"

BATCHSIZE = 16
accumulate_grad_batches = 1
EPOCHS = 50
n_gpus = -1
num_workers = 8
learning_rate = 1e-6 # note that this does nothing -- div_factor=25 and max_lr are used instead in onecycle
max_lr = 1e-3

gradient_clip_val = 1
use_gradient_clipping=True


data_dir = ''
models_dir = ""
model_run = ''
name = ''

pretrained_model_name = ''
encoder = PatchTSJEPALightning.load_from_checkpoint(checkpoint_path=os.path.join(models_dir, pretrained_model_name), map_location='cpu')
encoder.pretrain = False
encoder.model.pretrain = False

d_model = 512

outcome_df_path = ''
outcome_df = pd.read_csv(outcome_df_path)
outcome_df['Time Stamp (seconds)'] = outcome_df['Time Stamp (seconds)'].round()

zarr_files_shock_index = outcome_df.loc[outcome_df['shock_index_class'].isin([0,1]), 'file'].unique().tolist()

zarr_files = outcome_df.file.unique().tolist()
groups = [Path(i).stem.split('-')[0] for i in zarr_files]
labels = outcome_df['hypotension'].values.tolist()
outcome_df.drop(columns=['unique_identifier'], inplace=True)
zarr_to_ignore = []
zarr_files = [z for z in zarr_files if z not in zarr_to_ignore]
groups = zarr_files

splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_state)
train_idxs, test_idxs = next(splitter.split(X=zarr_files,  y=labels, groups=groups))

train_zarrs = [zarr_files[i] for i in train_idxs]
test_zarrs = [zarr_files[i] for i in test_idxs]

if y_outcome == 'shock_index_class':
    train_zarrs = [z for z in train_zarrs if z in zarr_files_shock_index]
    test_zarrs = [z for z in test_zarrs if z in zarr_files_shock_index]
    file_label_dict = dict(zip(outcome_df['file'].unique().tolist(), outcome_df['shock_index_class'].tolist()))
    train_labels = [file_label_dict[i] for i in train_zarrs]
else:
    train_labels = [labels[i] for i in train_idxs]
groups = [Path(i).stem.split('-')[0] for i in train_zarrs]

splitter2 = StratifiedGroupKFold(n_splits=10,  shuffle=True, random_state=random_state)
train_idxs, val_idxs = next(splitter2.split(X=train_zarrs, y=train_labels, groups=groups))

val_zarrs = [train_zarrs[i] for i in val_idxs]
train_zarrs = [train_zarrs[i] for i in train_idxs]

channels = ['ABP', 'II', 'V', 'PLETH','RESP']

c_in = len(channels)
c_in_head = c_in
class_weights = None 

frequency = 125
forecast_window_sec = 60*5 # number of seconds to predict in the future or past (-60 predicts 1 minute behind the end of the segment)
sample_seq_len_seconds = int(60*30) # 5 minute segment to predict the current segment
sample_stride_sec = int(60) # every 5 minutes + 1 minute (avoid including labels of the previous segment) 
win_length = int(1*125) # every ten seconds for patches
overlap = 0.
hop_length=win_length - int(overlap*win_length)
max_seq_len = sample_seq_len_seconds*frequency

if y_outcome != 'shock_index_class':
    dataset_filename = f"{frequency}Hz_{''.join(channels)}channels_{sample_seq_len_seconds}sec_segment_{sample_stride_sec}sec_stride_SS"
else:
    dataset_filename = f"{frequency}Hz_{''.join(channels)}channels_{sample_seq_len_seconds}sec_segment_{sample_stride_sec}sec_stride_SS_shock_index_class"


n_patches = (max(max_seq_len, win_length)-win_length) // hop_length + 1
if ((max_seq_len-win_length) % hop_length != 0):
    n_patches += 1
n_patches = int(n_patches)

# read in the sample_df (instead of calculated it)
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

forecast_within = False 
include_labels_in_x = False 
y_padding_mask =  2 

nan_tolerance = 0.2 # maxiumum amount of missing data allowed in a single channel 

lp_arch = dict(c_in=c_in_head, 
                input_size = d_model, 
                dropout=0.1)

lp_model = AvgPatchLogisticRegression(**lp_arch)

wandb_logger = WandbLogger(project=f"{model_run}", offline=wandb_offline, name=name, save_dir=models_dir)
wandb_logger.log_hyperparams({**dict(encoder.hparams), **{f"lp_head_{k}":v for k,v in lp_arch.items()}})
add_params = {'encoder_path':pretrained_model_name}

if __name__ == "__main__":
    pl.seed_everything(random_state)
    CLIP_INTERPOLATE_RANGES = {'ABP': {"phys_range":None, "percentiles":None}, 
                           'II': {"phys_range":None, "percentiles":None}, 
                           'V': {"phys_range":None, "percentiles":None}, 
                           'I': {"phys_range":None, "percentiles":None}, 
                           'III': {"phys_range":None, "percentiles":None},
                           'AVR': {"phys_range":None, "percentiles":None},
                           'AVF': {"phys_range":None, "percentiles":None},
                           'PLETH': {"phys_range":None, "percentiles":None},
                           'RESP': {"phys_range":None, "percentiles":None}
                           }
    train_ds = ForecastingDataset(
                 zarr_files=train_zarrs,
                 channels=channels, 
                 forecast_window_sec = forecast_window_sec,
                 outcome_df = outcome_df,
                 y_outcome=y_outcome,
                 include_labels_in_x=include_labels_in_x, 
                 forecast_within=forecast_within, 
                 sample_df = sample_df_train,
                 max_seq_len_sec=None, 
                 sample_seq_len_sec=sample_seq_len_seconds, 
                 sample_stride_sec=sample_stride_sec,
                 frequency=frequency, 
                 butterworth_filters=None,
                 median_filter_kernel_size=None,
                 clip_interpolations=CLIP_INTERPOLATE_RANGES, 
                 nan_tolerance=nan_tolerance,
                 require_all_channels=True
    )

    val_ds = ForecastingDataset(
                 zarr_files=val_zarrs,
                 channels=channels, 
                 sample_df = sample_df_val,
                 forecast_window_sec = forecast_window_sec,
                 outcome_df = outcome_df,
                 y_outcome=y_outcome,
                 forecast_within=forecast_within, 
                 include_labels_in_x=include_labels_in_x, 
                 max_seq_len_sec=None, 
                 sample_seq_len_sec=sample_seq_len_seconds, 
                 sample_stride_sec=sample_stride_sec,
                 frequency=frequency, 
                 butterworth_filters=None,
                 median_filter_kernel_size=None,
                 clip_interpolations=CLIP_INTERPOLATE_RANGES, 
                 nan_tolerance=nan_tolerance,
                 require_all_channels=True
    )

    test_ds = ForecastingDataset(
                    zarr_files=test_zarrs,
                    channels=channels, 
                    sample_df=sample_df_test,
                    forecast_window_sec = forecast_window_sec,
                    outcome_df = outcome_df,
                    y_outcome=y_outcome,
                    forecast_within=forecast_within, 
                    include_labels_in_x=include_labels_in_x, 
                    max_seq_len_sec=None, 
                    sample_seq_len_sec=sample_seq_len_seconds, 
                    sample_stride_sec=sample_stride_sec,
                    frequency=frequency, 
                    butterworth_filters=None,
                    median_filter_kernel_size=None,
                    clip_interpolations=CLIP_INTERPOLATE_RANGES,
                    nan_tolerance=nan_tolerance,
                    require_all_channels=True
    )
    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, drop_last=True, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    test_loader = DataLoader(test_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    test_targets = [y for _, y in test_loader]
    test_targets_cat = torch.cat(test_targets)

    if not Path(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz")).exists():
        train_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz"), compression='gzip', index=True)
    if not Path(os.path.join(models_dir, f"{dataset_filename}-val_samples.csv.gz")).exists():
        val_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-val_samples.csv.gz"), compression='gzip', index=True)
    if not Path(os.path.join(models_dir, f"{dataset_filename}-test_samples.csv.gz")).exists():
        test_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-test_samples.csv.gz"), compression='gzip', index=True)
    # cross fold validation
    inner_splitter = StratifiedGroupKFold(n_splits=5)
    auroc_scores = []
    avg_prec_scores = []
    accuracy_scores = []
    train_zarrs = train_ds.sample_df.file.unique().tolist() # need to extract exact train zarrs
    label_dict = dict(zip(outcome_df.file.unique().tolist(), outcome_df.hypotension.tolist()))
    train_labels = [label_dict[i] for i in train_zarrs]
    groups = [Path(i).stem.split('-')[0] for i in train_zarrs]
    #train_idxs, test_idxs = next(splitter.split(X=train_zarrs,  y=train_labels, groups=groups)) # enumerate([(train_idxs, test_idxs)])

    for fold, (fold_train_idx, fold_val_idx) in enumerate(inner_splitter.split(X=train_zarrs, groups=groups, y=train_labels)):
        fold_train_ds = torch.utils.data.Subset(train_ds, fold_train_idx)
        fold_val_ds = torch.utils.data.Subset(train_ds, fold_val_idx)
        fold_train_loader = DataLoader(fold_train_ds, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
        fold_val_loader = DataLoader(fold_val_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
        patchmeupe2e_model = PatchTFTSingleOutcomeLightning(learning_rate=learning_rate, 
                                    train_size=len(train_ds), 
                                    batch_size=BATCHSIZE,
                                    linear_probing_head=lp_model,
                                    preloaded_model=encoder, 
                                    label_smoothing=label_smoothing,
                                    metrics=metrics, 
                                    weight_decay=weight_decay,
                                    gamma=gamma,
                                    fine_tune=fine_tune,
                                    loss_fxn=loss_fxn,
                                    class_weights=class_weights,
                                    y_padding_mask=y_padding_mask,
                                    torch_model_name='model', 
                                    remove_pretrain_layers=['head', 'mask'],
                                    scheduler_type='OneCycle',
                                    optimizer_type='adamw',
                                    max_lr=max_lr, 
                                    epochs=EPOCHS,
                                    create_zero_channel_mask=False)
    # 2025-02-09 CHIL Probing Models-Final JEPA - w ABP, same splitsepoch=38-Focal:val_loss=0.31850-CE:val_ce_loss=0.61177.ckpt
        fold_filename = f"{model_run}-{fold}-{name}" + "{epoch:02d}-Focal:{val_loss:.5f}-CE:{val_ce_loss:.5f}"
        fold_callbacks = []
        fold_callback = ModelCheckpoint(dirpath=models_dir, save_top_k=1, monitor="val_auroc", mode='max', filename=fold_filename)
        fold_callbacks.append(fold_callback)
        trainer = pl.Trainer(precision=precision,
                        enable_checkpointing=True, # not me
                        enable_progress_bar=True, # not me
                        enable_model_summary=True, # not me
                        logger=wandb_logger,
                        val_check_interval=val_check_interval,
                        log_every_n_steps=50,
                        num_sanity_val_steps=0, # speed up
                        detect_anomaly=False, # speed up, though defualt
                        profiler=None, # this is the default, not me
                        strategy="ddp",
                        gradient_clip_val=gradient_clip_val,
                        gradient_clip_algorithm='norm' if use_gradient_clipping else None,
                        accelerator="gpu", 
                        devices=n_gpus, 
                        default_root_dir=models_dir, 
                        max_epochs=EPOCHS, 
                        fast_dev_run=False,
                        accumulate_grad_batches=accumulate_grad_batches,
                        sync_batchnorm=True, # added just in case we switch to use more GPUs
                        callbacks=fold_callbacks
                        )

        trainer.fit(model=patchmeupe2e_model, train_dataloaders=fold_train_loader, val_dataloaders=fold_val_loader, ckpt_path=None)
        best_model_path = fold_callback.best_model_path
        fold_val_preds = trainer.predict(model=patchmeupe2e_model, dataloaders=fold_val_loader, return_predictions=True, ckpt_path=best_model_path)
        fold_val_targets = [y for _, y in fold_val_loader]
        fold_val_preds_cat = torch.cat(fold_val_preds)
        fold_val_targets_cat = torch.cat(fold_val_targets)
        auroc_metric = MulticlassAUROC(num_classes=2,  average='none', ignore_index=y_padding_mask)
        avg_prec_metric = AveragePrecision(task='multiclass', num_classes=2, average='none', ignore_index=y_padding_mask)
        accuracy_metric = MulticlassAccuracy(num_classes=2, average='none', ignore_index=y_padding_mask)
        auroc_scores.append(auroc_metric(fold_val_preds_cat, fold_val_targets_cat)[1])
        avg_prec_scores.append(avg_prec_metric(fold_val_preds_cat, fold_val_targets_cat)[1])
        accuracy_scores.append(accuracy_metric(fold_val_preds_cat, fold_val_targets_cat)[1])
    # fit final model to all train data
    patchmeupe2e_model = PatchTFTSingleOutcomeLightning(learning_rate=learning_rate, 
                                    train_size=len(train_ds), 
                                    batch_size=BATCHSIZE,
                                    linear_probing_head=lp_model,
                                    preloaded_model=encoder, 
                                    label_smoothing=label_smoothing,
                                    metrics=metrics, 
                                    weight_decay=weight_decay,
                                    gamma=gamma,
                                    fine_tune=fine_tune,
                                    loss_fxn=loss_fxn,
                                    class_weights=class_weights,
                                    y_padding_mask=y_padding_mask,
                                    torch_model_name='model', 
                                    remove_pretrain_layers=['head', 'mask'],
                                    scheduler_type='OneCycle',
                                    optimizer_type='adamw',
                                    max_lr=max_lr, 
                                    epochs=EPOCHS,
                                    create_zero_channel_mask=False)
    filename = f"{model_run}-{name}" + "{epoch:02d}-Focal:{val_loss:.5f}-CE:{val_ce_loss:.5f}"
    checkpoint_callback = ModelCheckpoint(dirpath=models_dir, save_top_k=1, monitor="val_loss", mode='min', filename=filename)
    checkpoint_callback2 = ModelCheckpoint(dirpath=models_dir, save_top_k=1, monitor="val_auroc", mode='max', filename=filename)
    checkpoints = [checkpoint_callback, checkpoint_callback2]

    trainer = pl.Trainer(precision=precision,
                        enable_checkpointing=True, # not me
                        enable_progress_bar=True, # not me
                        enable_model_summary=True, # not me
                        logger=wandb_logger,
                        val_check_interval=val_check_interval,
                        log_every_n_steps=50,
                        num_sanity_val_steps=0, # speed up
                        detect_anomaly=False, # speed up, though defualt
                        profiler=None, # this is the default, not me
                        strategy="ddp",
                        gradient_clip_val=gradient_clip_val,
                        gradient_clip_algorithm='norm' if use_gradient_clipping else None,
                        accelerator="gpu", 
                        devices=n_gpus, 
                        default_root_dir=models_dir, 
                        max_epochs=EPOCHS, 
                        fast_dev_run=False,
                        accumulate_grad_batches=accumulate_grad_batches,
                        sync_batchnorm=True, # added just in case we switch to use more GPUs
                        callbacks=checkpoints
                        )


    trainer.fit(model=patchmeupe2e_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=None)

    #Predictions and targets
    best_model_path = checkpoint_callback2.best_model_path
    val_preds = trainer.predict(model=patchmeupe2e_model, dataloaders=val_loader, return_predictions=True, ckpt_path=best_model_path)
    test_preds = trainer.predict(model=patchmeupe2e_model, dataloaders=test_loader, return_predictions=True, ckpt_path=best_model_path)
    val_targets = [y for _, y in val_loader]

    val_preds_cat = torch.cat(val_preds)
    val_targets_cat = torch.cat(val_targets)
    test_preds_cat = torch.cat(test_preds)
    

    auroc_metric = MulticlassAUROC(num_classes=2,  average='none', ignore_index=y_padding_mask)
    avg_prec_metric = AveragePrecision(task='multiclass', num_classes=2, average='none', ignore_index=y_padding_mask)
    accuracy_metric = MulticlassAccuracy(num_classes=2, average='none', ignore_index=y_padding_mask)

    print("Val AUC", auroc_metric(val_preds_cat, val_targets_cat)[1])
    print("Test AUC", auroc_metric(test_preds_cat, test_targets_cat)[1])

    print('Validation set metrics')
    print(auroc_scores)
    print(avg_prec_scores)
    print(accuracy_scores)

    print('Validation set metrics with 95% confidence intervals:')
    
    def calc_confidence_interval(scores, confidence=0.95):
        n = len(scores)
        mean = np.mean(scores)
        std_err = np.std(scores, ddof=1) / np.sqrt(n)  # Standard error
        # Using t-distribution for small sample sizes
        from scipy import stats
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_of_error = t_val * std_err
        return mean, margin_of_error

    # Calculate and print CIs for each metric
    for metric_name, scores in [
        ('AUROC', auroc_scores),
        ('Average Precision', avg_prec_scores),
        ('Accuracy', accuracy_scores)
    ]:
        mean, margin = calc_confidence_interval(scores)
        print(f'{metric_name}: {mean:.3f} ± {margin:.3f} ({mean-margin:.3f} to {mean+margin:.3f})')

    test_preds_cat = torch.cat(test_preds)[:,1,0].cpu()
    test_targets_cat = torch.cat(test_targets)[:,0].cpu()

    auroc_metric = BinaryAUROC(ignore_index=y_padding_mask)
    avg_prec_metric = AveragePrecision(task='binary', ignore_index=y_padding_mask)
    accuracy_metric = BinaryAccuracy(ignore_index=y_padding_mask)

    print('AUC', auroc_metric(test_preds_cat, test_targets_cat))
    print('AP', avg_prec_metric(test_preds_cat, test_targets_cat))
    print('Accuracy', accuracy_metric(test_preds_cat, test_targets_cat))
    
    torch.save(val_preds_cat, f'{y_outcome}-{name}-val-preds.pt')
    torch.save(val_targets_cat, f'{y_outcome}-{name}-val-targets.pt')
    torch.save(test_preds_cat, f'{y_outcome}-{name}-test-preds.pt')
    torch.save(test_targets_cat, f'{y_outcome}-{name}-test-targets.pt')



