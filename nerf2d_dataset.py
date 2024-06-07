import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from einops import rearrange
from torch.utils.data import TensorDataset, DataLoader
from torchvision.io import read_image
import numpy as np

from camera_model_2d import pixel_center_rays

def read_image_folder(path: Path):
    with open(path / 'transforms.json') as f:
        transforms_json = json.load(f)

    # read poses
    poses = torch.stack([torch.Tensor(frame_data['transform_matrix']) for frame_data in transforms_json['frames']])



    # read images
    ims = torch.stack([read_image(str(path / f'cam-{i}.png')) for i in range(len(poses))])

    if path.name == 'train':
        # TODO: remove
        indices = np.arange(0, 50, 10)
        poses = poses[indices]
        # read images
        ims = torch.stack([read_image(str(path / f'cam-{10 * i}.png')) for i in range(len(poses))])
    # drop alpha channel and nromalize to floats
    ims = ims[:, :3, :, :] / 255

    focal = transforms_json['focal']

    return ims, poses, focal

class NeRFDataset2D(TensorDataset):
    """
    Each item is a ray, and its corresponding pixel color
    """

    def __init__(self, images: torch.Tensor, poses: torch.Tensor, focal_length: float):
        """
        :param images: Images in dataset N, C, H, W
        :param poses: poses in dataset N, 3, 3
        :param focal_length: focal_length of cameras
        """

        # save poses, focal length and images
        self.poses = poses
        self.focal_length = focal_length
        self.ims = images

        self.image_resolution = images.shape[2]

        # get a ray for each pixel
        # TODO hardcoded
        rays = [pixel_center_rays(self.image_resolution, self.focal_length, c2w) for c2w in self.poses]
        origins = torch.stack([ray[0] for ray in rays])
        dirs = torch.stack([ray[1] for ray in rays])

        # remove width dimension
        ims_no_w = rearrange(self.ims, 'n c h 1 -> n c h')

        # flatten all images and rays
        colors_flat = rearrange(ims_no_w, 'n c h -> (n h) c')
        origins_flat = rearrange(origins, 'n h d -> (n h) d')
        dirs_flat = rearrange(dirs, 'n h d -> (n h) d')

        # each entry is a tuple of (origin, direction, pixel_color)
        super().__init__(origins_flat, dirs_flat, colors_flat)

class NeRF2D_Datamodule(pl.LightningDataModule):

    def __init__(self, folder: Path, batch_size=100):
        super().__init__()

        self.save_hyperparameters(ignore=['folder'])

        # read training images and poses
        self.train_ims, self.train_poses, self.train_focal = read_image_folder(folder / 'train')
        self.train_height = self.train_ims.shape[2]
        self.train_dataset = NeRFDataset2D(self.train_ims, self.train_poses, self.train_focal)

        # read test images and poses
        self.test_ims, self.test_poses, self.test_focal = read_image_folder(folder / 'test')
        self.test_height = self.test_ims.shape[2]
        self.test_dataset = NeRFDataset2D(self.test_ims, self.test_poses, self.test_focal)

        # save additional hyperparams
        self.hparams.n_train_images = len(self.train_dataset)
        self.hparams.image_resolution = self.train_dataset.image_resolution

    def train_dataloader(self):
        return DataLoader(self.train_dataset, shuffle=True, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_height)
