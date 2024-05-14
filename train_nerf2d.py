from pathlib import Path

import pytorch_lightning as pl

from nerf2d_dataset import NeRF2D_Datamodule
from nerf2d import NeRF2D_LightningModule

dm = NeRF2D_Datamodule(folder=Path('data/views/'))
model = NeRF2D_LightningModule()

trainer = pl.Trainer(
    max_epochs=10
)

trainer.fit(model, dm)
