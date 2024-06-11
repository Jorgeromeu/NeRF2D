from rootutils import rootutils
import click

import wandb

# use project working directory
rootutils.setup_root(__file__, indicator=".project-root", dotenv=True, pythonpath=True, cwd=True)

@click.command()
@click.option('--artifact_name', help='Name of the artifact')
@click.option('--path', help='Path to the dataset', type=click.Path(exists=True))
def main(artifact_name, path):

    wandb.init(project='NeRF2D', job_type='log_dataset', name=f'log_{artifact_name}')

    artifact = wandb.Artifact(artifact_name, type='dataset')
    artifact.add_dir(path)

    wandb.log_artifact(artifact)

if __name__ == '__main__':
    main()
