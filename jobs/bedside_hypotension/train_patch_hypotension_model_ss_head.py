import os
import pandas as pd
from physiojepa.bedside import ForecastingDataset
import glob
from pathlib import Path
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch

from physiojepa.train import PatchTFTSingleOutcomeLightning, PatchTSJEPALightning
from physiojepa.heads import AvgPatchLogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from physiojepa.augmentations import MixupCallback, VariableChannelInput

from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.loggers import WandbLogger
from torchmetrics.classification import MulticlassAUROC, AveragePrecision, MulticlassAccuracy

random_state = 12
torch.set_float32_matmul_precision('medium')

mixup_callback = MixupCallback(num_classes=2, mixup_alpha=0.4, return_sequence_padding_mask=False, ignore_index=2)

wandb_offline = False

val_check_interval = 1.0
weight_decay = 1e-3
precision = '32'
loss_fxn = 'FocalLoss'
fine_tune = False
label_smoothing = 0. # only works for ce loss
gamma = 2.

BATCHSIZE = 16
accumulate_grad_batches = 1
EPOCHS = 50
n_gpus = -1
num_workers = 8
learning_rate = 1e-6 # note that this does nothing -- div_factor=25 and max_lr are used instead in onecycle
max_lr = 1e-2

gradient_clip_val = 1
use_gradient_clipping=True


data_dir = ''
models_dir = ''
model_run = ''
name = ''

pretrained_model_name = ''
encoder = PatchTSJEPALightning.load_from_checkpoint(checkpoint_path=os.path.join(models_dir, pretrained_model_name), map_location='cpu')
encoder.pretrain = False
encoder.model.pretrain = False

d_model = 256

filter_medications = False
if filter_medications:
    outcome_df_path = ''
else:
    outcome_df_path = ''

outcome_df = pd.read_csv(outcome_df_path)
outcome_df['Time Stamp (seconds)'] = outcome_df['Time Stamp (seconds)'].round()
outcome_zarrs = outcome_df.file.unique().tolist()

zarr_files = glob.glob(os.path.join(data_dir, '*.zarr/'))
zarr_to_ignore = []
zarr_files = [z for z in zarr_files if z not in zarr_to_ignore]
groups = [Path(i).stem.split('-')[0] for i in zarr_files]


TRAIN_SIZE = 0.9
# Split the filtered Zarr files into training and validation sets
splitter = GroupShuffleSplit(n_splits=1, train_size=TRAIN_SIZE, random_state=random_state)
pretrain_idxs, test_idxs = next(splitter.split(X=zarr_files, groups=groups))

pretrain_zarrs = [zarr_files[i] for i in pretrain_idxs]
pretrain_groups = [groups[i] for i in pretrain_idxs]

PRETRAIN_SIZE = 0.9
splitter2 = GroupShuffleSplit(n_splits=1, train_size=PRETRAIN_SIZE, random_state=random_state)
train_idxs, val_idxs = next(splitter2.split(X=pretrain_zarrs, groups=pretrain_groups))

train_zarrs = [pretrain_zarrs[i] for i in train_idxs if pretrain_zarrs[i] in outcome_zarrs]
val_zarrs = [pretrain_zarrs[i] for i in val_idxs if pretrain_zarrs[i] in outcome_zarrs]
test_zarrs = [zarr_files[i] for i in test_idxs if zarr_files[i] in outcome_zarrs]

ss_channels = ['ABP', 'II', 'V', 'PLETH','RESP']
channels = ['II', 'V', 'PLETH','RESP'] # this is the order of the SS model: ['ABP', 'II', 'V', 'PLETH','RESP']
drop_channel_indexes = [0]
indexes_to_add_channels = [ss_channels.index(i) for i in ss_channels if i not in channels]
variable_channel_cb = VariableChannelInput(indexes_to_add_channels=indexes_to_add_channels, n_channels_expected=len(ss_channels), channel_dim=1, return_sequence_padding_mask=False)

c_in = len(ss_channels)
c_in_head = c_in if drop_channel_indexes is None else c_in - len(drop_channel_indexes)

frequency = 125
forecast_window_sec = 60*5 # number of seconds to predict in the future or past (-60 predicts 1 minute behind the end of the segment)
sample_seq_len_seconds = int(60*45) # 5 minute segment to predict the current segment
sample_stride_sec = int(60) # every 5 minutes + 1 minute (avoid including labels of the previous segment) 
win_length = int(1*125) # every ten seconds for patches
overlap = 0.
hop_length=win_length - int(overlap*win_length)
max_seq_len = sample_seq_len_seconds*frequency

dataset_filename = f"{frequency}Hz_{''.join(channels)}channels_{sample_seq_len_seconds}sec_segment_{sample_stride_sec}sec_stride_{forecast_window_sec}sec_forecast_filtered_med{filter_medications}"

n_patches = (max(max_seq_len, win_length)-win_length) // hop_length + 1
if ((max_seq_len-win_length) % hop_length != 0):
    n_patches += 1
n_patches = int(n_patches)

# read in the sample_df (instead of calculated it)
if Path(f'{models_dir}/{dataset_filename}-train_samples.csv.gz').exists():
    sample_df_train = pd.read_csv(f'/sc/arion/projects/EHR_ML/bfox/models_hypotension_chil/{dataset_filename}-train_samples.csv.gz')
else:
    sample_df_train = None
if Path(f'{models_dir}/{dataset_filename}-val_samples.csv.gz').exists():
    sample_df_val = pd.read_csv(f'{models_dir}/{dataset_filename}-val_samples.csv.gz')
else:
    sample_df_val = None
if Path(f'{models_dir}/{dataset_filename}-test_samples.csv.gz').exists():
    sample_df_test = pd.read_csv(f'{models_dir}/{dataset_filename}-test_samples.csv.gz')
else:
    sample_df_test = None

forecast_within = False # we want to forecast exactly every forecast_window_sec
include_labels_in_x = False # if you set this to true, then c_in should be c_in + 1. This adds the current hypotension labels as a variable for forecasting
y_padding_mask = -100 if mixup_callback is not None else 2 # -100 is dummy 2 is real # this is the label that we are going to ignore in cross entropy

nan_tolerance = 0.2 # maxiumum amount of missing data allowed in a single channel 

lp_arch = dict(c_in=c_in_head, # avg patch logistic regression
                input_size = d_model, 
                dropout=0.1)

lp_model = AvgPatchLogisticRegression(**lp_arch)


wandb_logger = WandbLogger(project=f"{model_run}", offline=wandb_offline, name=name, save_dir=models_dir)
wandb_logger.log_hyperparams({**dict(encoder.hparams), **{f"lp_head_{k}":v for k,v in lp_arch.items()}})
add_params = {'encoder_path':pretrained_model_name, 'mixup_callback':True if mixup_callback is not None else False}

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
                 zarr_files=val_zarrs,
                 channels=channels, 
                 sample_df = sample_df_val,
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

    test_ds = ForecastingDataset(
                    zarr_files=test_zarrs,
                    channels=channels, 
                    sample_df=sample_df_test,
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
    test_loader = DataLoader(test_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    test_targets = [y for _, y in test_loader]
    test_targets_cat = torch.cat(test_targets)

    if not Path(f'/sc/arion/projects/EHR_ML/bfox/models_hypotension_chil/{dataset_filename}-train_samples.csv.gz').exists():
        train_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz"), compression='gzip', index=True)
    if not Path(f'/sc/arion/projects/EHR_ML/bfox/models_hypotension_chil/{dataset_filename}-val_samples.csv.gz').exists():
        val_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-val_samples.csv.gz"), compression='gzip', index=True)
    if not Path(f'/sc/arion/projects/EHR_ML/bfox/models_hypotension_chil/{dataset_filename}-test_samples.csv.gz').exists():
        test_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-test_samples.csv.gz"), compression='gzip', index=True)
    
    patchmeupe2e_model = PatchTFTSingleOutcomeLightning(learning_rate=learning_rate, 
                                label_smoothing=label_smoothing,
                                weight_decay=weight_decay,
                                gamma=gamma,
                                drop_channel_indexes=drop_channel_indexes,
                                linear_probing_head=lp_model,
                                fine_tune=fine_tune,
                                loss_fxn=loss_fxn,
                                class_weights=None,
                                use_sequence_padding_mask=False,
                                y_padding_mask=y_padding_mask,
                                pretrained_encoder_path=None,
                                preloaded_model=encoder, 
                                torch_model_name='model', 
                                remove_pretrain_layers=['head', 'mask'],
                                train_size=len(train_ds), 
                                scheduler_type='OneCycle',
                                optimizer_type='adamw',
                                max_lr=max_lr, 
                                metrics={}, 
                                epochs=EPOCHS, 
                                batch_size=BATCHSIZE)
    
    filename = f"{model_run}-{name}" + "{epoch:02d}-Focal:{val_loss:.5f}-CE:{val_ce_loss:.5f}"
    checkpoint_callback = ModelCheckpoint(dirpath=models_dir, save_top_k=1, monitor="val_loss", mode='min', filename=filename)
    checkpoints = [checkpoint_callback]
    if variable_channel_cb is not None:
        checkpoints.append(variable_channel_cb)
    if mixup_callback is not None:
        checkpoints.append(mixup_callback)
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

    preds = trainer.predict(model=patchmeupe2e_model, dataloaders=val_loader, return_predictions=True, ckpt_path='best')
    test_preds = trainer.predict(model=patchmeupe2e_model, dataloaders=test_loader, return_predictions=True, ckpt_path='best')
    targets = [y for _, y in val_loader]

    preds_cat = torch.cat(preds)
    targets_cat = torch.cat(targets)
    test_preds_cat = torch.cat(test_preds)
    

    auroc_metric = MulticlassAUROC(num_classes=2,  average='none', ignore_index=y_padding_mask)
    avg_prec_metric = AveragePrecision(task='multiclass', num_classes=2, average='none', ignore_index=y_padding_mask)
    accuracy_metric = MulticlassAccuracy(num_classes=2, average='none', ignore_index=y_padding_mask)

    print("Val AUC", auroc_metric(preds_cat, targets_cat)[1])
    print("Test AUC", auroc_metric(test_preds_cat, test_targets_cat)[1])

    torch.save(test_preds_cat, f'{model_run}-{name}-test_preds.pt')
    torch.save(test_targets_cat, f'{model_run}-{name}-test_targets.pt')
    torch.save(preds_cat, f'{model_run}-{name}-val_preds.pt')
    torch.save(targets_cat, f'{model_run}-{name}-val_targets.pt')



