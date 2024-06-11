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

    # initialize run
    wandb.init(project=cfg.wandb.project, job_type=cfg.wandb.job_type)

    # download dataset
    dataset_artifact = wandb.use_artifact(cfg.data.artifact)
    dataset_dir = Path(dataset_artifact.download())

    # load dataset
    dm = NeRF2D_Datamodule(
        folder=dataset_dir,
        batch_size=cfg.data.batch_size,
        camera_subset=cfg.data.camera_subset,
        camera_subset_n=cfg.data.camera_subset_n,
        t_far=cfg.model.t_far,
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
    checkpoint_callback = ModelCheckpoint(monitor='val_psnr', mode='max', dirpath='checkpoints', save_top_k=1)
    early_stopping = pl.callbacks.EarlyStopping('val_loss', patience=cfg.trainer.patience)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        overfit_batches=cfg.trainer.overfit_batches,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping],
        fast_dev_run=cfg.dev_run,
    )

    # train
    trainer.fit(model, dm)
    trainer.test(model, dm)

    wandb.finish()

if __name__ == '__main__':
    main()
