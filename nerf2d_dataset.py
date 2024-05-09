import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.utils.data import TensorDataset, DataLoader
from torchvision.io import read_image

from nerf2d import get_rays2d

def read_image_folder(path: Path):
    transforms_json = json.load(open(path / 'transforms.json'))

    # read poses
    poses = torch.stack([torch.Tensor(frame_data['transform_matrix']) for frame_data in transforms_json['frames']])

    # read images
    ims = torch.stack([read_image(str(path / f'cam-{i}.png')) for i in range(len(poses))])
    ims = rearrange(ims, 'n c h w -> n c h w').squeeze(-1)

    focal = transforms_json['focal']

    # TODO throw away alpha channel and normalize images

    return ims, poses, focal

class NeRFDataset2D(TensorDataset):
    """
    Each item is a ray, and its corresponding pixel color
    """

    def __init__(self, images: torch.Tensor, poses: torch.Tensor, focal_length: float):
        """
        :param images: Images in dataset N, C, H
        :param poses: poses in dataset N, 3, 3
        :param focal_length: focal_length of cameras
        """

        self.ims = images
        self.poses = poses
        self.focal_length = focal_length

        self.image_resolution = self.ims.shape[2]

        # get a ray for each pixel
        rays = [get_rays2d(self.image_resolution, self.focal_length, c2w) for c2w in self.poses]
        origins = torch.stack([ray[0] for ray in rays])
        dirs = torch.stack([ray[1] for ray in rays])

        # remove alpha channel
        ims = self.ims[:, 0:3, :]

        # flatten all images and rays
        colors_flat = rearrange(ims, 'n c h -> (n h) c')
        origins_flat = rearrange(origins, 'n h d -> (n h) d')
        dirs_flat = rearrange(dirs, 'n h d -> (n h) d')

        # each entry is a tuple of (origin, direction, pixel_color)
        super().__init__(origins_flat, dirs_flat, colors_flat)

class NeRF2D_Datamodule(pl.LightningDataModule):

    def __init__(self, folder: Path, batch_size=100):
        super().__init__()

        self.save_hyperparameters(ignore=['folder'])

        # read images and poses
        self.ims, self.poses, self.focal = read_image_folder(folder)
        self.dataset = NeRFDataset2D(self.ims, self.poses, self.focal)

        # save additional hyperparams
        self.hparams.n_train_images = len(self.dataset)
        self.hparams.image_resolution = self.dataset.image_resolution

    def train_dataloader(self):
        return DataLoader(self.dataset, shuffle=True, batch_size=self.hparams.batch_size)
