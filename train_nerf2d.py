from pathlib import Path

import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from nerf2d import NeRF2D_LightningModule
from nerf2d_dataset import NeRF2D_Datamodule

pl.seed_everything(0)
torch.cuda.empty_cache()

dm = NeRF2D_Datamodule(folder=Path('data/cube/'), batch_size=100)
model = NeRF2D_LightningModule(
    lr=1e-4,
    t_near=1.5,
    t_far=4,
    n_steps=100,
    n_freqs_pos=10,
    n_layers=4,
    d_hidden=128,
    chunk_size=30000
)

origins, directions, colors = next(iter(dm.train_dataloader()))

wandb_logger = pl_loggers.WandbLogger(
    project='NeRF2D',
    mode='run',
    job_type='train'
)

checkpoint_callback = ModelCheckpoint('checkpoints', save_last=True)
early_stopping = pl.callbacks.EarlyStopping('val_loss', patience=30)

trainer = pl.Trainer(
    max_epochs=1000,
    logger=wandb_logger,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback, early_stopping]
)
trainer.fit(model, dm)
