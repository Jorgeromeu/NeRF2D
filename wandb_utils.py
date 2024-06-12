import re
from pathlib import Path
from re import Pattern

from tqdm import tqdm
from wandb.apis.public import Run, File

from nerf2d import NeRF2D_LightningModule
from wandb import CommError, Artifact

def example_train_run(api) -> Run:
    run: Run = api.run("romeu/NeRF2D/5x9fjp84")
    return run

class RunDataManager:
    """
    Utility class for downloading and caching files from wandb runs
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _download_files(self, files: list[File], root: Path, replace=False):
        """
        Download files from wandb run to a specified directory.
        """

        for f in tqdm(files):

            try:
                f.download(replace=replace, root=str(root.absolute()))
            except CommError as e:
                pass

    def _run_dir(self, run: Run):
        path = self.data_dir / run.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def download_run_files(self, run: Run, files: list[File], replace=False) -> list[Path]:
        self._download_files(files, self._run_dir(run), replace=replace)
        return self.read_run_files(run, files)

    def read_run_files(
            self,
            run: Run,
            files: list[File],
            sort_by_step=False
    ) -> list[Path]:

        files = [self._run_dir(run) / f.name for f in files]

        if sort_by_step:
            files = sorted(files, key=lambda f: int(f.stem.split('_')[-2]))

        return files

def get_files_by_regex(run: Run, pattern: Pattern) -> list[File]:
    files = [f for f in run.files() if pattern.match(f.name)]
    return files

def get_files_by_label(run: Run, label: str) -> list[File]:
    pattern = re.compile(r'.*/' + label + r'_(\d+).*\.png')
    return get_files_by_regex(run, pattern)

def get_last_file_by_label(run: Run, label: str) -> list[File]:
    files = get_files_by_label(run, label)
    return [files[-1]]

def sort_by_step(files: list[Path]):
    return sorted(files, key=lambda f: int(f.stem.split('_')[-2]))

def parse_file_name(file: Path):
    label, step, _ = file.stem.split('_')
    return {
        'label': label,
        'step': int(step),
    }

def first_logged_artifact_of_type(run: Run, artifact_type: str) -> Artifact:
    for artifact in run.logged_artifacts():
        if artifact.type == artifact_type:
            return artifact
    return None

# Use-case specific

def get_artifact_dir_api(api, artifact_id: str):
    artifact = api.artifact(artifact_id)
    artifact_dir = Path(artifact.download())
    return artifact_dir

def get_model_checkpoint(artifact_dir: Path):
    return artifact_dir / 'model.ckpt'

def get_checkpoint_api(api, artifact_id: str):
    artifact_dir = get_artifact_dir_api(api, artifact_id)
    checkpoint = get_model_checkpoint(artifact_dir)
    return checkpoint

def load_from_checkpoint_api(api, artifact_id: str) -> NeRF2D_LightningModule:
    checkpoint = get_checkpoint_api(api, artifact_id)
    model = NeRF2D_LightningModule.load_from_checkpoint(checkpoint)
    return model
