import re
from pathlib import Path
from re import Pattern

from tqdm import tqdm
from wandb.apis.public import Run, File

from nerf2d import NeRF2D_LightningModule
from nerf2d_dataset import NeRF2D_Datamodule
from wandb import CommError, Artifact

def example_train_run(api) -> Run:
    run: Run = api.run("romeu/NeRF2D/5x9fjp84")
    return run

class RunWrapper:
    """
    Utility class over runs to provide a better interface
    """

    def __init__(self, run: Run):
        self.run = run
        self.files = [f for f in run.files()]
        self.id = run.id

    def get_files_by_regex(self: Run, pattern: Pattern) -> list[File]:
        files = [f for f in self.files if pattern.match(f.name)]
        return files

    def get_files_by_label(self, label: str) -> list[File]:
        pattern = re.compile(r'.*/' + label + r'_(\d+).*\.png')
        files = self.get_files_by_regex(pattern)
        files = sorted(files, key=lambda f: parse_run_file_name(Path(f.name))['step'])
        return files

    def get_last_file_by_label(self, label: str) -> list[File]:
        files = self.get_files_by_label(label)
        return [files[-1]]

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

    def download_run_files_by_label(self, run: RunWrapper, label: str, replace=False) -> list[Path]:
        files = run.get_files_by_label(label)
        return self.download_run_files(run.run, files, replace=replace)

    def download_last_run_file_by_label(self, run: RunWrapper, label: str, replace=False) -> list[Path]:
        files = run.get_last_file_by_label(label)
        return self.download_run_files(run.run, files, replace=replace)

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

def parse_run_file_name(file: Path):
    label, step, _ = file.stem.rsplit('_', maxsplit=2)
    return {
        'label': label,
        'step': int(step),
    }

def first_logged_artifact_of_type(run: Run, artifact_type: str) -> Artifact:
    for artifact in run.logged_artifacts():
        if artifact.type == artifact_type:
            return artifact
    return None

def first_used_artifact_of_type(run: Run, artifact_type: str) -> Artifact:
    for artifact in run.used_artifacts():
        if artifact.type == artifact_type:
            return artifact
    return None

# Use-case specific

def get_ckpt(artifact: Artifact):
    path = Path(artifact.download())
    return path / 'model.ckpt'

def load_lm_from_artifact(model_artifact: Artifact) -> NeRF2D_LightningModule:
    """
    Load a model from a model artifact
    """
    ckpt = get_ckpt(model_artifact)
    return NeRF2D_LightningModule.load_from_checkpoint(ckpt)

def load_dm_from_artifact(data_artifact: Artifact, model_artifact: Artifact) -> NeRF2D_Datamodule:
    """
    Load a datamodule from a data artifact and model artifact
    """
    path = Path(data_artifact.download())
    ckpt = get_ckpt(model_artifact)
    dm = NeRF2D_Datamodule.load_from_checkpoint(ckpt, folder=path)
    return dm

def load_dm_from_artifact_no_train(data_artifact: Artifact) -> NeRF2D_Datamodule:
    """
    Load a datamodule from a data artifact without a model artifact
    """
    path = Path(data_artifact.download())
    dm = NeRF2D_Datamodule(folder=path)
    return dm
