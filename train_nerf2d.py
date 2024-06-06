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
    dm = NeRF2D_Datamodule(folder=Path('./data/cube_depth/'), batch_size=cfg.data.batch_size)

    print("Dataset loaded")

    # Experiment with different depth_loss_weight values to see how it affects the training (from 0.0 to 1.0)
    depth_loss_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for depth_loss_weight in depth_loss_weights:
        cfg.model.depth_loss_weight = depth_loss_weight

        # setup model
        print("The depth_loss_weight is: ", cfg.model.depth_loss_weight)
        model = NeRF2D_LightningModule(**cfg.model)

        print("Model loaded")

        # setup training loop
        wandb_logger = pl_loggers.WandbLogger(
            project=cfg.wandb.project,
            mode='run',
            job_type=cfg.get('job_type'),
            name='nerf2d - depth loss weight: ' + str(cfg.model.depth_loss_weight),
            log_model=True,
        )

        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', dirpath='checkpoints', save_last=True)
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

        wandb.finish()

        print("Training finished")

if __name__ == '__main__':
    main()
