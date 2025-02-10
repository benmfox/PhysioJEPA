import os
import pandas as pd
from physiojepa.bedside import SelfSupervisedDataset
import glob
from pathlib import Path
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch

from physiojepa.train import PatchTSJEPALightning
from sklearn.model_selection import GroupShuffleSplit

from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.loggers import WandbLogger

random_state = 12
torch.set_float32_matmul_precision('medium')

scheduler_type = 'onecycle'
wandb_offline = False
use_flash_attn = False

val_check_interval = 1.0
weight_decay = 1e-4
precision = '32'

BATCHSIZE = 8
accumulate_grad_batches = 1
EPOCHS = 100
n_gpus = -1
num_workers = 8
learning_rate = 1.5e-6
max_lr = 1e-3 # this determines the learning rate

loss_func = 'mse'

gradient_clip_val = 1
use_gradient_clipping=True

data_dir = ''
models_dir = ""
model_run = ''
name = ''

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

train_zarrs = [pretrain_zarrs[i] for i in train_idxs]
val_zarrs = [pretrain_zarrs[i] for i in val_idxs]

#test_zarrs = [zarr_files[i] for i in test_idxs]

channels = ['ABP', 'II', 'V', 'PLETH', 'RESP']

c_in = len(channels) 


frequency = 125
sample_seq_len_seconds = int(60*45) # 45 minute segment to predict the current segment
sample_stride_sec = int(60*45) # every 45 minutes
win_length = int(1*125) # every ten seconds for patches
overlap = 0.
hop_length=win_length - int(overlap*win_length)
max_seq_len = sample_seq_len_seconds*frequency

dataset_filename = f"{frequency}Hz_{''.join(channels)}channels_{sample_seq_len_seconds}sec_segment_{sample_stride_sec}sec_stride_SS"

n_patches = (max(max_seq_len, win_length)-win_length) // hop_length + 1
if ((max_seq_len-win_length) % hop_length != 0):
    n_patches += 1
n_patches = int(n_patches)

# read in the sample_df (instead of calculated it)
## this is only for ABP waveforms, that are 60 seconds long, and have the outcome for the current minute
#sample_df = pd.read_csv('/sc/arion/projects/EHR_ML/bfox/mimic3_samples_60_seconds_125Hz_ABP.csv.gz')
if Path(f'{models_dir}/{dataset_filename}-train_samples.csv.gz').exists():
    sample_df_train = pd.read_csv(f'{models_dir}/{dataset_filename}-train_samples.csv.gz')
else:
    sample_df_train = None
if Path(f'{models_dir}/{dataset_filename}-val_samples.csv.gz').exists():
    sample_df_val = pd.read_csv(f'{models_dir}/{dataset_filename}-val_samples.csv.gz')
else:
    sample_df_val = None

y_padding_mask = -100

nan_tolerance = 0.2 # maxiumum amount of missing data allowed in a single channel 

encoder_arch = dict(c_in=c_in,
                  win_length=win_length, # the length of the patch of time/interval or short time ft windown length (when time_domain=False)
                  hop_length=hop_length, # the length of the distance between each patch/fft
                  max_seq_len=max_seq_len, # maximum sequence len
                  time_domain=True,
                  pos_encoding_type='tAPE',
                  use_flash_attn=use_flash_attn, # indicator to use flash attention
                  patch_encoder_type='linear',
                  use_revin=True, # if time_domain is true, whether or not to instance normalize time data
                  dim1reduce=False, # indicator to normalize by timepoint in revin
                  affine=True, # if time_domain is true, whether or not to learn revin normalization parameters 
                  n_layers=3, # the number of transformer encoder layers to use
                  d_model=256, # the dimension of the input to the transofmrer encoder
                  n_heads=4, # the number of heads in each layer
                  shared_embedding=False, # indicator for whether or not each channel should be projected with its own set of linear weights to the encoder dimension
                  d_ff=512, # the feedforward layer size in the transformer
                  norm='LayerNorm', # BatchNorm or LayerNorm during trianing
                  attn_dropout=0., # dropout in attention
                  dropout=0.1, # dropout for linear layers
                  act="gelu", # activation function
                  res_attention=True, # whether to use residual attention
                  pre_norm=False)

predictor_arch = dict(c_in=c_in,
                      num_patches=n_patches,         # number of patches from encoder
                      d_model=256,        # encoder embedding dimension
                      predictor_dim=128,  # predictor embedding dimension (typically smaller)
                      n_heads=2,
                      n_layers=1,
                      d_ff=256,
                      pos_encoding_type='learned',
                      norm='LayerNorm',
                      dropout=0.1,
                      attn_dropout=0.0,
                      act="gelu",
                      pre_norm=False,
                      use_flash_attn=use_flash_attn)

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

    if not Path(f'{models_dir}/{dataset_filename}-train_samples.csv.gz').exists():
        train_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz"), compression='gzip', index=True)
    if not Path(f'{models_dir}/{dataset_filename}-val_samples.csv.gz').exists():
        val_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-val_samples.csv.gz"), compression='gzip', index=True)

    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)


    patchfreq_model = PatchTSJEPALightning(learning_rate=learning_rate,
                                            train_size=len(train_ds),
                                            batch_size=BATCHSIZE,
                                            channels=channels,
                                            patchtsjepa_encoder_kwargs=encoder_arch,
                                            patchtsjepa_predictor_kwargs=predictor_arch,
                                            use_sequence_padding_mask=False,
                                            loss_func=loss_func,
                                            max_lr=max_lr,
                                            weight_decay=weight_decay,
                                            epochs=EPOCHS,
                                            optimizer_type='adamw',
                                            scheduler_type='OneCycle',
                                            target_mask_range=(0.1, 0.5),
                                            context_mask_range=(0.2, 0.8),
                                            mask_all_channels=True,
                                            pretrain=True,
                                            )
    
    filename = f"{model_run}-{name}" + "{epoch:02d}-{val_loss:.5f}"
    checkpoint_callback = ModelCheckpoint(dirpath=models_dir, save_top_k=1, monitor="val_loss", mode='min', filename=filename)
    checkpoints = [checkpoint_callback]

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


    trainer.fit(model=patchfreq_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=None)

