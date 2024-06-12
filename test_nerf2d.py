from pathlib import Path

import hydra
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
import torch
from omegaconf import DictConfig
from rootutils import rootutils

import wandb
from nerf2d import NeRF2D_LightningModule
from nerf2d_dataset import NeRF2D_Datamodule
from wandb_utils import get_model_checkpoint

# use project working directory
rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True, cwd=True)

@hydra.main(version_base=None, config_path='config', config_name='test_config')
def main(cfg: DictConfig):
    torch.set_float32_matmul_precision('high')

    # ensure reproducibility
    pl.seed_everything(cfg.seed)

    # initialize run
    wandb.init(project=cfg.wandb.project, job_type=cfg.wandb.job_type)

    # load model
    model_artifact = wandb.use_artifact(cfg.model.artifact)
    model_dir = Path(model_artifact.download())
    model = NeRF2D_LightningModule.load_from_checkpoint(get_model_checkpoint(model_dir))

    # load dataset
    dataset_artifact = wandb.use_artifact(cfg.data.artifact)
    dataset_dir = Path(dataset_artifact.download())

    # load dataset
    dm = NeRF2D_Datamodule(
        folder=dataset_dir,
        t_far=model.hparams.t_far
    )

    # setup training loop
    wandb_logger = pl_loggers.WandbLogger(
        project=cfg.wandb.project,
        mode='run',
        job_type=cfg.wandb.get('job_type'),
        name=cfg.wandb.get('run_name'),
    )

    trainer = pl.Trainer(
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=wandb_logger,
        fast_dev_run=cfg.dev_run,
    )

    # test
    trainer.test(model, dm)

    wandb.finish()

if __name__ == '__main__':
    main()
