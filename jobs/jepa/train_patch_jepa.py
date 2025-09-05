import os
import pandas as pd
from physiojepa.bedside import SelfSupervisedDataset, CLIP_INTERPOLATE_RANGES
from pathlib import Path
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import torch
import torch.nn as nn
from physiojepa.jepa import JEPASimpleLightning, mse_variance_loss, loss_pred
from sklearn.model_selection import  GroupShuffleSplit
from datetime import datetime
from functools import partial

from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.loggers import WandbLogger
import yaml, glob

torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)

config_path = 'train_patch_jepa.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

random_state = config['run_config']['random_state']
loss_fxn = mse_variance_loss if config['training']['loss_fxn'] == "variance" else loss_pred

max_lr = config['scheduler']['max_lr']
div_factor = config['scheduler']['div_factor']
final_div_factor = config['scheduler']['final_div_factor']
pct_start=config['scheduler']['pct_start']
anneal_strategy=config['scheduler']['anneal_strategy']
scheduler_kwargs = dict(max_lr=max_lr, div_factor=div_factor, final_div_factor=final_div_factor, pct_start=pct_start, anneal_strategy=anneal_strategy)

BATCHSIZE = config['training']['batch_size']
accumulate_grad_batches = config['training']['accumulate_grad_batches']
EPOCHS = config['training']['epochs']
n_gpus = config['training']['n_gpus']
num_workers = config['training']['num_workers']
use_gradient_clipping=config['training']['use_gradient_clipping']
gradient_clip_val = config['training']['gradient_clip_val']
precision = config['run_config']['precision']

data_dir = config['paths']['data_dir']
models_dir = config['paths']['models_dir']
model_run = config['run_config']['model_run']
name = config['run_config']['name']

zarr_files = glob.glob(os.path.join(data_dir, '*.zarr/'))
groups = [Path(i).stem.split('-')[0] for i in zarr_files]

splitter = GroupShuffleSplit(n_splits=1, train_size=0.95, random_state=random_state)
train_idxs, val_idxs = next(splitter.split(X=zarr_files, groups=groups))

val_zarrs = [zarr_files[i] for i in val_idxs]
train_zarrs = [zarr_files[i] for i in train_idxs]

channels = config['dataset']['channels']
c_in = len(channels) 

frequency = config['dataset']['frequency']
sample_seq_len_seconds = config['dataset']['seq_len_sec']
sample_stride_sec = config['dataset']['sample_stride_sec']

patch_seconds = config['dataset']['patch_seconds']
win_length = int(patch_seconds*frequency) 
overlap = config['dataset']['overlap']
hop_length=win_length - int(overlap*win_length)
max_seq_len = frequency*sample_seq_len_seconds

constant_nan_tolerance = config['dataset']['constant_nan_tolerance']
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

encoder_arch = dict(c_in=c_in,
            num_patches=n_patches,
            patch_size=win_length,
            patch_stride=hop_length,
            d_model=config['encoder']['d_model'],
            nhead=config['encoder']['nhead'],
            use_tst_block=config['encoder']['use_tst_block'],
            shared_embedding=config['encoder']['shared_embedding'],
            num_layers=config['encoder']['num_layers'],
            pe_type=config['encoder']['pe_type'],
            mlp_ratio=config['encoder']['mlp_ratio'],
            qkv_bias=config['encoder']['qkv_bias'],
            qk_scale=config['encoder']['qk_scale'],
            drop_rate=config['encoder']['drop_rate'],
            attn_drop_rate=config['encoder']['attn_drop_rate'],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            jepa=config['encoder']['jepa'],
            tokenizer_type=config['encoder']['tokenizer_type'],
            tokenizer_kwargs=config['encoder']['tokenizer_kwargs'],
            embed_activation=nn.GELU())

predictor_arch = dict(num_patches=n_patches,
            encoder_embed_dim=encoder_arch['d_model'],
            predictor_embed_dim=config['predictor']['predictor_embed_dim'], #384
            nhead=config['predictor']['nhead'], #12
            num_layers=config['predictor']['num_layers'], #6
            pe_type=config['predictor']['pe_type'],
            mlp_ratio=config['predictor']['mlp_ratio'],
            qkv_bias=config['predictor']['qkv_bias'],
            qk_scale=config['predictor']['qk_scale'],
            drop_rate=config['predictor']['drop_rate'],
            attn_drop_rate=config['predictor']['attn_drop_rate'],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            use_tst_block=config['predictor']['use_tst_block'],
            c_in_mask_tokens=config['predictor']['c_in_mask_tokens'],
            embed_activation=nn.GELU())

name = config['run_config']['name']
filename = f"{name}" + "{epoch:02d}-{val_loss:.5f}"
date_str = datetime.now().strftime("%Y-%m-%d")
sub_dir = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}"
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=2, monitor="val_loss", mode='min', filename=filename)
checkpoint_callback2 = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=2, monitor="train_loss", mode='min', filename=filename, save_last=True)
wandb_project = config['run_config']['wandb_project']
wandb_name = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}-{config['run_config']['name']}"
os.makedirs(os.path.join(models_dir, sub_dir), exist_ok=True)
wandb_logger = WandbLogger(project=f"{wandb_project}", offline=False, name=name, save_dir=os.path.join(models_dir, sub_dir))

callbacks = [checkpoint_callback, checkpoint_callback2]

if __name__ == "__main__":
    pl.seed_everything(random_state)

    train_ds = SelfSupervisedDataset(
                 zarr_files=train_zarrs,
                 channels=channels, 
                 sample_df = sample_df_train,
                 max_seq_len_sec=None, 
                 sample_seq_len_sec=sample_seq_len_seconds, 
                 sample_stride_sec=sample_stride_sec,
                 frequency=frequency, 
                 butterworth_filters=None,
                 median_filter_kernel_size=None,#3,
                 normalize_signals=config['dataset']['normalize_signals'],
                 require_all_channels=config['dataset']['require_all_channels'],
                 clip_interpolations=CLIP_INTERPOLATE_RANGES, # adding fill nan and clip interpolations, note that to fill nan (even if you dont want to clip, the channel must be included here, just with None for the values in the dictionary. 
                 constant_nan_tolerance=constant_nan_tolerance
    )

    val_ds = SelfSupervisedDataset(
                    zarr_files=val_zarrs,
                    channels=channels, 
                    sample_df=sample_df_val,
                    max_seq_len_sec=None, 
                    sample_seq_len_sec=sample_seq_len_seconds, 
                    sample_stride_sec=sample_stride_sec,
                    frequency=frequency, 
                    butterworth_filters=None,
                    median_filter_kernel_size=None,#3,
                    normalize_signals=config['dataset']['normalize_signals'],
                    require_all_channels=config['dataset']['require_all_channels'],
                    clip_interpolations=CLIP_INTERPOLATE_RANGES,
                    constant_nan_tolerance=constant_nan_tolerance
    )

    if not Path(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz")).exists():
        train_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz"), compression='gzip', index=True)
    if not Path(os.path.join(models_dir, f"{dataset_filename}-val_samples.csv.gz")).exists():
        val_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-val_samples.csv.gz"), compression='gzip', index=True)

    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False, drop_last=False, num_workers=num_workers, persistent_workers=True, pin_memory=False)


    patchfreq_model = JEPASimpleLightning(learning_rate=config['optimizer']['learning_rate'],
                                        train_size=len(train_ds),
                                        batch_size=BATCHSIZE,
                                        n_gpus=n_gpus,
                                        patchtsjepa_encoder_kwargs=encoder_arch,
                                        patchtsjepa_predictor_kwargs=predictor_arch,
                                        weight_decay=config['optimizer']['weight_decay'],
                                        use_weight_decay_scheduler=config['optimizer']['use_weight_decay_scheduler'],
                                        final_weight_decay=config['optimizer']['final_weight_decay'],
                                        epochs=EPOCHS,
                                        loss_fn=loss_fxn,
                                        optimizer_type=config['optimizer']['optimizer_type'],
                                        scheduler_type=config['scheduler']['scheduler_type'],
                                        target_mask_range=config['training']['target_mask_range'], # the target can be up to 50% of the original x 
                                        context_mask_range=config['training']['context_mask_range'], # the context can be up to 80% of masked out target x 
                                        mask_block_range=config['training']['mask_block_range'],
                                        ema_decay=config['training']['ema_decay'],
                                        scheduler_kwargs=scheduler_kwargs,
                                        transforms=None)
    
    trainer = pl.Trainer(precision=precision,
                        enable_checkpointing=True, 
                        enable_progress_bar=True, 
                        enable_model_summary=True, 
                        logger=wandb_logger,
                        val_check_interval=config['training']['val_check_interval'],
                        log_every_n_steps=10,
                        num_sanity_val_steps=2, # speed up
                        detect_anomaly=False, # speed up, though defualt
                        profiler=None, # this is the default, not me
                        strategy="ddp",
                        gradient_clip_val=gradient_clip_val,
                        gradient_clip_algorithm='norm' if use_gradient_clipping else None,
                        accelerator="gpu", 
                        devices=n_gpus, 
                        default_root_dir=os.path.join(models_dir, sub_dir), 
                        max_epochs=EPOCHS,
                        fast_dev_run=False,
                        accumulate_grad_batches=accumulate_grad_batches,
                        sync_batchnorm=True, # added just in case we switch to use more GPUs
                        callbacks=callbacks
                        )


    trainer.fit(model=patchfreq_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=None)

