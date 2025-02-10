import os, optuna
import pandas as pd
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.samplers import TPESampler
from timeflies.bedside import ForecastingDataset, CLIP_INTERPOLATE_RANGES
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl


from physiojepa.train import PatchTFTSupervised
from physiojepa.heads import AvgPatchLogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from physiojepa.augmentations import MixupCallback

import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import MulticlassAUROC, AveragePrecision, MulticlassAccuracy, BinaryAUROC

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

study_name = 'hypertune_hypotension_model'
wandb_kwargs = {"project": "Hypertune Hypotension Model", "reinit": True}
wandbc = WeightsAndBiasesCallback(metric_name="val_loss", wandb_kwargs=wandb_kwargs, as_multirun=True)

random_state = 12
torch.set_float32_matmul_precision('medium')


scheduler_type = 'onecycle'

wandb_offline = False

save_datasets = True # save the sample_df to csv for faster training later

val_check_interval = 1.0
precision = '32'
loss_fxn = 'CrossEntropy'
label_smoothing = 0. # only works for ce loss
gamma = 2.

accumulate_grad_batches = 1
EPOCHS = 100
n_gpus = -1
num_workers = 8

gradient_clip_val = 1
use_gradient_clipping=True

data_dir = ''
models_dir = ""
model_run = ''
name = ''

outcome_df_path = ''

outcome_df = pd.read_csv(outcome_df_path)
outcome_df['Time Stamp (seconds)'] = outcome_df['Time Stamp (seconds)'].round()

zarr_files = outcome_df.file.unique().tolist()#glob.glob(os.path.join(data_dir, '*.zarr/'))
groups = [Path(i).stem.split('-')[0] for i in zarr_files]
labels = outcome_df['hypotension'].values.tolist()
outcome_df.drop(columns=['file', 'unique_identifier'], inplace=True)
zarr_to_ignore = []
zarr_files = [z for z in zarr_files if z not in zarr_to_ignore]
groups = zarr_files

TRAIN_SIZE = 0.9
# Split the filtered Zarr files into training and validation sets
splitter = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=random_state)
train_idxs, test_idxs = next(splitter.split(X=zarr_files, groups=groups))

train_zarrs = [zarr_files[i] for i in train_idxs]
groups = [Path(i).stem.split('-')[0] for i in train_zarrs]
test_zarrs = [zarr_files[i] for i in test_idxs]

HYPERTUNE_SIZE = 0.7
splitter2 = GroupShuffleSplit(n_splits=1, train_size=HYPERTUNE_SIZE, random_state=random_state)
train_idxs, hypertune_idxs = next(splitter2.split(X=train_zarrs, groups=groups))

hypertune_zarrs = [train_zarrs[i] for i in hypertune_idxs]
train_zarrs = [train_zarrs[i] for i in train_idxs]

@wandbc.track_in_wandb()
def objective(trial: optuna.trial.Trial) -> float:

    channel_choices = [['ABP','II', 'V', 'PLETH','RESP'],
                       ['ABP','II', 'V', 'PLETH'],
                       ['ABP','II', 'V'],
                       ['ABP','II'],
                       ['ABP']]
    n_channels = trial.suggest_int("n_channels", low=1, high=5, step=1)
    channels = channel_choices[n_channels-1]
    BATCHSIZE = trial.suggest_categorical("batchsize", [8, 16])
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    max_lr = min(learning_rate*100, 1e-2)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-2, log=True)
    
    use_mixup = trial.suggest_categorical("use_mixup", [True, False])
    if use_mixup:   
        mixup_alpha = trial.suggest_float("mixup_alpha", 0.1, 0.5, step=0.1)
        mixup_callback = MixupCallback(num_classes=2, mixup_alpha=mixup_alpha, return_sequence_padding_mask=False, ignore_index=2)
    else:
        mixup_callback = None

    c_in = len(channels) 

    frequency = 125
    forecast_window_sec = 60*5 # number of seconds to predict in the future or past (-60 predicts 1 minute behind the end of the segment)
    
    sample_seq_len_seconds = trial.suggest_categorical("sample_seq_len_seconds", [60*5, 60*15, 60*30, 60*45])
    sample_stride_sec = int(60) # every 5 minutes + 1 minute (avoid including labels of the previous segment) 
    win_length_sec = trial.suggest_categorical("win_length_sec", [1, 3, 5, 10])
    win_length = int(win_length_sec*frequency) # every ten seconds for patches
    overlap = 0.
    hop_length=win_length - int(overlap*win_length)
    max_seq_len = sample_seq_len_seconds*frequency

    dataset_filename = f"{frequency}Hz_{''.join(channels)}channels_{sample_seq_len_seconds}sec_segment_{sample_stride_sec}sec_stride_{forecast_window_sec}sec_forecast_filtered_medFalse"

    n_patches = (max(max_seq_len, win_length)-win_length) // hop_length + 1
    if ((max_seq_len-win_length) % hop_length != 0):
        n_patches += 1
    n_patches = int(n_patches)

    # read in the sample_df (instead of calculated it)
    ## this is only for ABP waveforms, that are 60 seconds long, and have the outcome for the current minute
    if Path(f'{models_dir}/{dataset_filename}-train_samples.csv.gz').exists():
        sample_df_train = pd.read_csv(f'{models_dir}/{dataset_filename}-train_samples.csv.gz')
    else:
        sample_df_train = None
    if Path(f'{models_dir}/{dataset_filename}-hypertune_samples.csv.gz').exists():
        sample_df_val = pd.read_csv(f'{models_dir}/{dataset_filename}-hypertune_samples.csv.gz')
    else:
        sample_df_val = None

    forecast_within = False # we want to forecast exactly every forecast_window_sec
    include_labels_in_x = False # if you set this to true, then c_in should be c_in + 1. This adds the current hypotension labels as a variable for forecasting
    y_padding_mask = 2 # -100 is dummy 2 is real # this is the label that we are going to ignore in cross entropy

    nan_tolerance = 0.2 # maxiumum amount of missing data allowed in a single channel 
## I adjusted the encoder kwargs
    encoder_kwargs = dict(c_in=c_in,
                win_length=win_length,
                hop_length=hop_length,
                max_seq_len=max_seq_len,
                #time_domain=True,
                pos_encoding_type=trial.suggest_categorical("pos_encoding_type", ['learned', 'tAPE']), # options include learned or tAPE
                relative_attn_type=trial.suggest_categorical("relative_attn_type", ['vanilla', 'eRPE']), # options include vanilla or eRPE
                use_revin=True,
                dim1reduce=False, # indicator to normalize by timepoint in revin or by channel (likely use False here)
                use_flash_attn=False,
                affine=trial.suggest_categorical("affine", [True, False]),
                mask_ratio=0., # if pretrain head is set to false, this doesnt run
                augmentations=[], # if pretrain head is set to false, this doesnt run (patch_mask, jitter_zero_mask, )
                n_layers=trial.suggest_int("n_layers", 1, 3, step=1), 
                d_model=trial.suggest_categorical("d_model", [64, 128, 256, 512]),
                n_heads=trial.suggest_categorical("n_heads", [1,2,4]),
                shared_embedding=trial.suggest_categorical("shared_embedding", [True, False]),
                d_ff=trial.suggest_categorical("d_ff", [128, 256, 512, 1024]),
                norm='LayerNorm',
                attn_dropout=trial.suggest_float("attn_dropout", 0.0, 0.5, step=0.1),
                dropout=trial.suggest_float("dropout", 0.0, 0.5, step=0.1),
                act=trial.suggest_categorical("act", ["gelu", "relu"]), 
                res_attention=trial.suggest_categorical("res_attention", [True, False]),
                pre_norm=trial.suggest_categorical("pre_norm", [True, False]),
                store_attn=False,
                pretrain_head=False, # this must be set to tfalse
                pretrain_head_n_layers=0,
                pretrain_head_dropout=0.
                )


    lp_arch = dict(c_in=c_in, # avg patch logistic regression
                    input_size = encoder_kwargs['d_model'], 
                    dropout=trial.suggest_float("dropout_head", 0.0, 0.5, step=0.1)
                    )


    lp_model = AvgPatchLogisticRegression(**lp_arch)

    train_ds = ForecastingDataset(
                 zarr_files=train_zarrs,
                 channels=channels, 
                 sample_df = sample_df_train,
                 forecast_window_sec = forecast_window_sec,
                 outcome_df = outcome_df,
                 forecast_within=forecast_within, # set to false because we are predicting every minute exactly
                 include_labels_in_x=include_labels_in_x, # # indicator to include y labels in sample data time frame (resampled to the frequency)
                 max_seq_len_sec=None, 
                 sample_seq_len_sec=sample_seq_len_seconds, 
                 sample_stride_sec=sample_stride_sec,
                 frequency=frequency, 
                 butterworth_filters=None,#ALL_FREQUENCY_FILTERS, # adding butterworth filters
                 median_filter_kernel_size=None,#3,
                 clip_interpolations=CLIP_INTERPOLATE_RANGES, # adding fill nan and clip interpolations, note that to fill nan (even if you dont want to clip, the channel must be included here, just with None for the values in the dictionary. 
                 nan_tolerance=nan_tolerance
    )

    val_ds = ForecastingDataset(
                    zarr_files=hypertune_zarrs,
                    channels=channels, 
                    sample_df=sample_df_val,
                    forecast_window_sec = forecast_window_sec,
                    outcome_df = outcome_df,
                    forecast_within=forecast_within, # set to false because we are predicting every minute exactly
                    include_labels_in_x=include_labels_in_x, # # indicator to include y labels in sample data time frame (resampled to the frequency)
                    max_seq_len_sec=None, 
                    sample_seq_len_sec=sample_seq_len_seconds, 
                    sample_stride_sec=sample_stride_sec,
                    frequency=frequency, 
                    butterworth_filters=None,#ALL_FREQUENCY_FILTERS,
                    median_filter_kernel_size=None,#3,
                    clip_interpolations=CLIP_INTERPOLATE_RANGES,
                    nan_tolerance=nan_tolerance
    )

    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    #test_loader = DataLoader(test_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    if not Path(f'{models_dir}/{dataset_filename}-train_samples.csv.gz').exists():
        train_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz"), compression='gzip', index=True)
    if not Path(f'{models_dir}/{dataset_filename}-hypertune_samples.csv.gz').exists():
        val_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-hypertune_samples.csv.gz"), compression='gzip', index=True)
        #test_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-test_samples.csv.gz"), compression='gzip', index=True)
    

    patchmeupe2e_model = PatchTFTSupervised(learning_rate=learning_rate, 
                                      class_weights=None,
                                      loss_fxn=loss_fxn, # can be CrossEntropy or FocalLoss, I would try with FocalLoss if imbalanced
                                      gamma=gamma, # gamma for focal loss, ignored in CrossEntropy
                                      label_smoothing=0., # i would potentially try without label smoothing
                                      use_sequence_padding_mask=False,
                                      y_padding_mask=y_padding_mask,
                                      train_loader_size=len(train_loader), 
                                      max_lr = max_lr, 
                                      metrics=[], 
                                      epochs=EPOCHS, 
                                      optimizer_type='adamw',
                                      weight_decay=weight_decay,
                                      scheduler_type=scheduler_type,
                                      encoder_kwargs=encoder_kwargs,
                                      linear_probing_head=lp_model
                                      )
    #prune_callback = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    if mixup_callback is not None:
        callbacks = [mixup_callback]
    else:
        callbacks = []
    trainer = pl.Trainer(precision=precision, #may need to change if on minerva
                    enable_checkpointing=True, # not me
                    enable_progress_bar=True, # not me
                    enable_model_summary=True, # not me
                    logger=None,
                    val_check_interval=val_check_interval,
                    sync_batchnorm=True,
                    strategy="ddp",
                    log_every_n_steps=50,
                    gradient_clip_val=gradient_clip_val,
                    gradient_clip_algorithm='norm' if use_gradient_clipping else None,
                    num_sanity_val_steps=0, # speed up
                    detect_anomaly=False, # speed up, though defualt
                    profiler=None, # this is the default, not me
                    accelerator="gpu", 
                    accumulate_grad_batches=accumulate_grad_batches,
                    devices=n_gpus, #switch to n_gpus
                    default_root_dir=models_dir, 
                    max_epochs=50, 
                    fast_dev_run=False,
                    callbacks=callbacks)
    trainer.fit(model=patchmeupe2e_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=None)
    preds = trainer.predict(model=patchmeupe2e_model, dataloaders=val_loader, return_predictions=True, ckpt_path='best')
    targets = [y for _, y in val_loader]
    preds_cat = torch.cat(preds)
    targets_cat = torch.cat(targets)

    auroc_metric = MulticlassAUROC(num_classes=2,  average='none', ignore_index=y_padding_mask)
    #auroc_metric = BinaryAUROC(ignore_index=y_padding_mask)
    avg_prec_metric = AveragePrecision(task='multiclass', num_classes=2, average='none', ignore_index=y_padding_mask)
    accuracy_metric = MulticlassAccuracy(num_classes=2, average='none', ignore_index=y_padding_mask)

    auroc_val = auroc_metric(preds_cat, targets_cat)
    avg_prec_val = avg_prec_metric(preds_cat, targets_cat)
    accuracy_val = accuracy_metric(preds_cat, targets_cat)

    return auroc_val[1].item()

if __name__ == "__main__":
    pl.seed_everything(random_state)

    storage_name = "sqlite:///{}.db".format(study_name)
    
    sampler = TPESampler(n_startup_trials=50, multivariate=False)
    study = optuna.create_study(direction="maximize", 
                                sampler=sampler,
                                study_name=study_name, 
                                pruner=optuna.pruners.MedianPruner(), 
                                storage=storage_name, 
                                load_if_exists=True)
    
    study.optimize(objective, n_trials=500, gc_after_trial=True, callbacks=[wandbc])
