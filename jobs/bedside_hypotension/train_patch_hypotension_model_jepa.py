import os
import pandas as pd
from physio_jepa.bedside import SelfSupervisedDataset, CLIP_INTERPOLATE_RANGES
from pathlib import Path
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch

from physio_jepa.train import PatchTSJEPALightning
from sklearn.model_selection import StratifiedGroupKFold

from physio_jepa.loss import mse_loss

from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.loggers import WandbLogger

random_state = 12
torch.set_float32_matmul_precision('medium')

scheduler_type = 'onecycle'
wandb_offline = False

loss_func = mse_loss
val_check_interval = 1.0
weight_decay = 1e-4
precision = '16-mixed'

BATCHSIZE = 16
accumulate_grad_batches = 1
EPOCHS = 100
n_gpus = -1
num_workers = 8
learning_rate = 1e-6
max_lr = 0.001 

gradient_clip_val = 1
use_gradient_clipping=True

data_dir = ''
models_dir = ""
model_run = ''
name = ''

outcome_df_path = ''
outcome_df = pd.read_csv(outcome_df_path)
outcome_df['Time Stamp (seconds)'] = outcome_df['Time Stamp (seconds)'].round()

zarr_files = outcome_df.file.unique().tolist()#
groups = [Path(i).stem.split('-')[0] for i in zarr_files]
labels = outcome_df['hypotension'].values.tolist()
outcome_df.drop(columns=['file', 'unique_identifier'], inplace=True)
zarr_to_ignore = []
zarr_files = [z for z in zarr_files if z not in zarr_to_ignore]
groups = zarr_files

splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=random_state)
train_idxs, test_idxs = next(splitter.split(X=zarr_files,  y=labels, groups=groups))

train_zarrs = [zarr_files[i] for i in train_idxs]
groups = [Path(i).stem.split('-')[0] for i in train_zarrs]

train_labels = [labels[i] for i in train_idxs]

splitter2 = StratifiedGroupKFold(n_splits=10,  shuffle=True, random_state=random_state)
train_idxs, val_idxs = next(splitter2.split(X=train_zarrs, y=train_labels, groups=groups))

val_zarrs = [train_zarrs[i] for i in val_idxs]
train_zarrs = [train_zarrs[i] for i in train_idxs]


channels = ['ABP', 'II', 'V', 'PLETH', 'RESP']
c_in = len(channels) 


frequency = 125
sample_seq_len_seconds = int(60*30) # 30 minute segment to predict the current segment
sample_stride_sec = int(60*30) # every 30 minutes
win_length = int(1*125) 
overlap = 0.
hop_length=win_length - int(overlap*win_length)
max_seq_len = sample_seq_len_seconds*frequency

nan_tolerance = 0.2 # maxiumum amount of missing data allowed in a single channel 
dataset_filename = f"{frequency}Hz_{''.join(channels)}channels_{sample_seq_len_seconds}sec_segment_{sample_stride_sec}sec_stride_SS"

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

y_padding_mask = 2 

encoder_arch = dict(c_in=c_in,
                  win_length=win_length, # the length of the patch of time/interval or short time ft windown length (when time_domain=False)
                  hop_length=hop_length, # the length of the distance between each patch/fft
                  max_seq_len=max_seq_len, # maximum sequence len
                  pos_encoding_type='tAPE',
                  use_revin=True, # if time_domain is true, whether or not to instance normalize time data
                  affine=True, # if time_domain is true, whether or not to learn revin normalization parameters 
                  n_layers=4, # the number of transformer encoder layers to use
                  d_model=512, # the dimension of the input to the transofmrer encoder
                  n_heads=8, # the number of heads in each layer
                  shared_embedding=False, # indicator for whether or not each channel should be projected with its own set of linear weights to the encoder dimension
                  d_ff=2048, # the feedforward layer size in the transformer
                  attn_dropout=0., # dropout in attention
                  dropout=0.1, # dropout for linear layers
                  act="gelu", # activation function
                  pre_norm=True)

predictor_arch = dict(c_in=c_in,
                      num_patches=n_patches,         # number of patches from encoder
                      d_model=encoder_arch['d_model'],        # encoder embedding dimension
                      predictor_dim=256,  # predictor embedding dimension (typically smaller)
                      n_heads=4,
                      n_layers=2,
                      d_ff=1024,
                      pos_encoding_type='tAPE',
                      dropout=0.1,
                      attn_dropout=0.,
                      act="gelu",
                      pre_norm=True
                      )

wandb_logger = WandbLogger(project=f"{model_run}", offline=wandb_offline, name=name, save_dir=models_dir)

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

    train_ds = SelfSupervisedDataset(
                 zarr_files=train_zarrs,
                 channels=channels, 
                 sample_df = sample_df_train,
                 max_seq_len_sec=None, 
                 sample_seq_len_sec=sample_seq_len_seconds, 
                 sample_stride_sec=sample_stride_sec,
                 frequency=frequency, 
                 butterworth_filters=None,#ALL_FREQUENCY_FILTERS, # adding butterworth filters
                 median_filter_kernel_size=None,#3,
                 clip_interpolations=CLIP_INTERPOLATE_RANGES, # adding fill nan and clip interpolations, note that to fill nan (even if you dont want to clip, the channel must be included here, just with None for the values in the dictionary. 
                 nan_tolerance=nan_tolerance
    )

    val_ds = SelfSupervisedDataset(
                    zarr_files=val_zarrs,
                    channels=channels, 
                    sample_df=sample_df_val,
                    max_seq_len_sec=None, 
                    sample_seq_len_sec=sample_seq_len_seconds, 
                    sample_stride_sec=sample_stride_sec,
                    frequency=frequency, 
                    butterworth_filters=None,#ALL_FREQUENCY_FILTERS,
                    median_filter_kernel_size=None,#3,
                    clip_interpolations=CLIP_INTERPOLATE_RANGES,
                    nan_tolerance=nan_tolerance
    )

    if not Path(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz")).exists():
        train_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz"), compression='gzip', index=True)
    if not Path(os.path.join(models_dir, f"{dataset_filename}-val_samples.csv.gz")).exists():
        val_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-val_samples.csv.gz"), compression='gzip', index=True)

    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)


    patchfreq_model = PatchTSJEPALightning(learning_rate=learning_rate,
                                            train_size=len(train_ds),
                                            batch_size=BATCHSIZE,
                                            channels=channels,
                                            patchtsjepa_encoder_kwargs=encoder_arch,
                                            patchtsjepa_predictor_kwargs=predictor_arch,
                                            loss_func=loss_func,
                                            max_lr=max_lr,
                                            weight_decay=weight_decay,
                                            epochs=EPOCHS,
                                            optimizer_type='adamw',
                                            scheduler_type='OneCycle',
                                            target_mask_range=(0.1, 0.5),
                                            context_mask_range=(0.2, 0.8),
                                            pretrain=True,
                                            )
    
    filename = f"{model_run}-{name}" + "{epoch:02d}-{val_loss:.5f}"
    checkpoint_callback = ModelCheckpoint(dirpath=models_dir, save_top_k=1, monitor="val_loss", mode='min', filename=filename)
    checkpoint_callback2 = ModelCheckpoint(dirpath=models_dir, save_top_k=1, monitor="train_loss", mode='min', filename=filename)
    checkpoints = [checkpoint_callback, checkpoint_callback2]

    trainer = pl.Trainer(precision=precision,
                        enable_checkpointing=True, 
                        enable_progress_bar=True, 
                        enable_model_summary=True, 
                        logger=wandb_logger,
                        val_check_interval=val_check_interval,
                        log_every_n_steps=50,
                        num_sanity_val_steps=0, 
                        detect_anomaly=False, 
                        profiler=None, 
                        strategy="ddp",
                        gradient_clip_val=gradient_clip_val,
                        gradient_clip_algorithm='norm' if use_gradient_clipping else None,
                        accelerator="gpu", 
                        devices=n_gpus, 
                        default_root_dir=models_dir, 
                        max_epochs=EPOCHS,
                        fast_dev_run=False,
                        accumulate_grad_batches=accumulate_grad_batches,
                        sync_batchnorm=True, 
                        callbacks=checkpoints
                        )


    trainer.fit(model=patchfreq_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=None)

