import torch, pandas as pd, wandb, os, glob, lightning.pytorch as pl
import yaml
from physiojepa.patchtst import PatchTFTSimpleLightning
from physiojepa.bedside import SelfSupervisedDataset, CLIP_INTERPOLATE_RANGES
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from datetime import datetime

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner

import wandb
from torch import nn

from pathlib import Path

torch.set_float32_matmul_precision('high')
torch.backends.cuda.enable_flash_sdp(True)

config_path = 'train_patchtst.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
random_state = config['run_config']['random_state']
loss_fxn = config['training']['loss_fxn']

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
            patch_size=win_length,
            patch_stride=hop_length,
            num_patches=n_patches,
            d_model=config['encoder']['d_model'],
            n_heads=config['encoder']['n_heads'],
            d_ff=config['encoder']['d_ff'],
            num_layers=config['encoder']['num_layers'],
            augmentations=config['encoder']['augmentations'],
            mask_ratio=config['encoder']['mask_ratio'],
            shared_embedding=config['encoder']['shared_embedding'],
            pretrain_head=config['encoder']['pretrain_head'],
            dropout=config['encoder']['dropout'],
            attn_dropout=config['encoder']['attn_dropout'],
            act=config['encoder']['act'],
            pre_norm=config['encoder']['pre_norm'],
            pe_type=config['encoder']['pe_type'],
            qkv_bias=config['encoder']['qkv_bias'],
            init_std=config['encoder']['init_std'],
            tokenizer_type=config['encoder']['tokenizer_type'],
            tokenizer_kwargs=config['encoder']['tokenizer_kwargs'])


name = config['run_config']['name']
filename = f"{name}" + "{epoch:02d}-{val_loss:.5f}"
date_str = datetime.now().strftime("%Y-%m-%d")
sub_dir = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}"
checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=2, monitor="val_loss", mode='min', filename=filename)
checkpoint_callback2 = ModelCheckpoint(dirpath=os.path.join(models_dir, sub_dir), save_top_k=2, monitor="train_loss", mode='min', filename=filename, save_last=True)
wandb_project = config['run_config']['wandb_project']
wandb_name = f"{date_str}-{config['run_config']['model_type']}-{config['run_config']['model_run']}-{config['run_config']['name']}"
os.makedirs(os.path.join(models_dir, sub_dir), exist_ok=True)

wandb_logger = WandbLogger(project=f'{wandb_project}', offline=False, name=wandb_name, save_dir=os.path.join(models_dir, sub_dir))
wandb_logger.log_hyperparams(config)
# train model
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
                 median_filter_kernel_size=None,
                 normalize_signals=config['dataset']['normalize_signals'],
                 require_all_channels=config['dataset']['require_all_channels'],
                 clip_interpolations=CLIP_INTERPOLATE_RANGES,
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
                    median_filter_kernel_size=None,
                    normalize_signals=config['dataset']['normalize_signals'],
                    require_all_channels=config['dataset']['require_all_channels'],
                    clip_interpolations=CLIP_INTERPOLATE_RANGES,
                    constant_nan_tolerance=constant_nan_tolerance
    )
    if not Path(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz")).exists():
        train_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-train_samples.csv.gz"), compression='gzip', index=True)
    if not Path(os.path.join(models_dir, f"{dataset_filename}-val_samples.csv.gz")).exists():
        val_ds.sample_df.to_csv(os.path.join(models_dir, f"{dataset_filename}-val_samples.csv.gz"), compression='gzip', index=True)
    train_loader = DataLoader(train_ds, batch_size=BATCHSIZE, shuffle=True, num_workers=num_workers, drop_last=False, persistent_workers=True, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCHSIZE, shuffle=False, num_workers=num_workers, drop_last=False, persistent_workers=True, pin_memory=False)
    patchfreq_model = PatchTFTSimpleLightning(learning_rate=config['optimizer']['learning_rate'],
                                        train_size=len(train_ds),
                                        batch_size=BATCHSIZE,
                                        n_gpus=n_gpus,
                                        metrics={},
                                        loss_func=loss_fxn,
                                        weight_decay=config['optimizer']['weight_decay'],
                                        epochs=EPOCHS,
                                        use_weight_decay_scheduler=config['optimizer']['use_weight_decay_scheduler'],
                                        final_weight_decay=config['optimizer']['final_weight_decay'],
                                        optimizer_type=config['optimizer']['optimizer_type'],
                                        scheduler_type=config['scheduler']['scheduler_type'],
                                        huber_delta=config['training']['huber_delta'],
                                        scheduler_kwargs=scheduler_kwargs,
                                        transforms=None,
                                        **encoder_arch)
    
    trainer = pl.Trainer(precision=precision,
                     enable_checkpointing=True, 
                     enable_progress_bar=True, 
                     enable_model_summary=True, 
                     logger=wandb_logger, 
                     strategy="ddp",
                     sync_batchnorm=True,
                     val_check_interval=config['training']['val_check_interval'],
                     log_every_n_steps=10,
                     gradient_clip_val=gradient_clip_val,
                     gradient_clip_algorithm='norm' if use_gradient_clipping else None,
                     num_sanity_val_steps=2,
                     detect_anomaly=False, 
                     profiler=None, 
                     accelerator="gpu", 
                     accumulate_grad_batches=accumulate_grad_batches,
                     devices=n_gpus, 
                     default_root_dir=os.path.join(models_dir, sub_dir), 
                     max_epochs=EPOCHS, 
                     fast_dev_run=False,
                     callbacks=[checkpoint_callback, checkpoint_callback2])
    
    if config['training']['use_lr_finder']:
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(patchfreq_model, train_dataloaders=train_loader, update_attr=False, attr_name="max_lr") 
        new_lr = lr_finder.suggestion()
        patchfreq_model.scheduler_kwargs['max_lr'] = new_lr
        print(f"Using max lr: {new_lr}")
    
    trainer.fit(model=patchfreq_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=None)
