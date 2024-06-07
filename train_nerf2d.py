from pathlib import Path

import hydra
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from rootutils import rootutils

import wandb
from nerf2d import NeRF2D_LightningModule
from nerf2d_dataset import NeRF2D_Datamodule

# use project working directory
rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True, cwd=True)

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')

    # ensure reproducibility
    pl.seed_everything(cfg.seed)

    # load dataset
    dm = NeRF2D_Datamodule(
        folder=Path(cfg.data.folder),
        batch_size=cfg.data.batch_size,
        camera_subset=cfg.data.camera_subset,
        camera_subset_n=cfg.data.camera_subset_n,
    )

    # load model
    model = NeRF2D_LightningModule(**cfg.model)

    # setup training loop
    wandb_logger = pl_loggers.WandbLogger(
        project=cfg.wandb.project,
        mode='run',
        job_type=cfg.wandb.get('job_type'),
        name=cfg.wandb.get('run_name'),
        log_model=True,
    )

    # callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val_psnr', mode='max', dirpath='checkpoints', save_last=True)
    early_stopping = pl.callbacks.EarlyStopping('val_psnr', patience=cfg.trainer.patience)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        overfit_batches=cfg.trainer.overfit_batches,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
    )

    # train
    trainer.fit(model, dm)

    wandb.finish()

if __name__ == '__main__':
    main()
