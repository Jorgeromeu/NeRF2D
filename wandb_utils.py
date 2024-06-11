from pathlib import Path

from nerf2d import NeRF2D_LightningModule

def get_artifact_dir_api(api, artifact_id: str):
    artifact = api.artifact(artifact_id)
    artifact_dir = Path(artifact.download())
    return artifact_dir

def get_checkpoint_api(api, artifact_id: str):
    artifact_dir = get_artifact_dir_api(api, artifact_id)
    checkpoint = artifact_dir / 'model.ckpt'
    return checkpoint

def load_from_checkpoint_api(api, artifact_id: str) -> NeRF2D_LightningModule:
    checkpoint = get_checkpoint_api(api, artifact_id)
    model = NeRF2D_LightningModule.load_from_checkpoint(checkpoint)
    return model
