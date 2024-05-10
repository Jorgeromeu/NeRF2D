from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from nerf2d import NeRF2D_LightningModule
from nerf2d_dataset import NeRF2D_Datamodule

pl.seed_everything(0)

dm = NeRF2D_Datamodule(folder=Path('data/views/'), batch_size=100)
model = NeRF2D_LightningModule(lr=1e-4, t_near=0, t_far=2, n_steps=100)

logger = pl_loggers.WandbLogger(project='NeRF2D', mode='run')

checkpoint_callback = ModelCheckpoint(
    'checkpoints',
    every_n_epochs=10
)

trainer = pl.Trainer(
    max_epochs=1000,
    logger=logger,
    log_every_n_steps=1,
)
trainer.fit(model, dm)
