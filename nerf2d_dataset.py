import json
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision.io import read_image

from camera_model_2d import pixel_center_rays
from transform2d import Transform2D

def read_image_folder(path: Path, t_far=6):
    with open(path / 'transforms.json') as f:
        transforms_json = json.load(f)

    # read poses
    poses = torch.stack([torch.Tensor(frame_data['transform_matrix']) for frame_data in transforms_json['frames']])

    # read images
    ims = torch.stack([read_image(str(path / f'cam-{i}.png')) for i in range(len(poses))])
    # drop alpha channel and normalize to floats
    ims = ims[:, :3, :, :] / 255

    focal = transforms_json['focal']

    depths = np.stack([np.load(path / f'cam-{i}.npz')['depth_map'] for i in range(len(poses))])
    depths = rearrange(depths, 'n h d -> n d h 1')
    depths = Tensor(depths)

    # set -1 in depths to max value
    depths[depths == -1] = t_far

    return ims, poses, focal, depths

def get_n_evenly_spaced_views(views, n):
    return views[::(len(views) // (n + 1))][:-1]

def plot_poses(poses: list[Tensor]):
    poses = [Transform2D.from_matrix(pose) for pose in poses]

    positions = torch.stack([pose.translation() for pose in poses])
    angles = torch.stack([pose.rotation() for pose in poses])

    dx = torch.cos(angles)
    dy = torch.sin(angles)
    d = torch.stack([dx, dy], dim=1)

    fig, ax = plt.subplots()
    plt.quiver(positions[:, 0], positions[:, 1], d[:, 0], d[:, 1])
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    ax.set_aspect('equal', 'box')
    fig.tight_layout()
    return fig

class NeRFDataset2D(TensorDataset):
    """
    Each item is a ray, and its corresponding pixel color
    """

    def __init__(self, images: torch.Tensor, poses: torch.Tensor, focal_length: float, depths: torch.Tensor):
        """
        :param images: Images in dataset N, C, H, W
        :param poses: poses in dataset N, 3, 3
        :param focal_length: focal_length of cameras
        """

        # save poses, focal length, images and depth data
        self.poses = poses
        self.focal_length = focal_length
        self.ims = images
        self.depths = depths

        self.image_resolution = images.shape[2]

        # get a ray for each pixel
        # TODO hardcoded
        rays = [pixel_center_rays(self.image_resolution, self.focal_length, c2w) for c2w in self.poses]
        origins = torch.stack([ray[0] for ray in rays])
        dirs = torch.stack([ray[1] for ray in rays])

        # remove width dimension
        ims_no_w = rearrange(self.ims, 'n c h 1 -> n c h')
        depths_no_w = rearrange(self.depths, 'n d h 1 -> n d h')

        # flatten all images and rays
        colors_flat = rearrange(ims_no_w, 'n c h -> (n h) c')
        origins_flat = rearrange(origins, 'n h d -> (n h) d')
        dirs_flat = rearrange(dirs, 'n h d -> (n h) d')

        # flatten depth data
        # depths_flat = rearrange(self.depths, 'n h 1 -> (n h)')
        depths_flat = rearrange(depths_no_w, 'n c h -> (n h) c')

        # each entry is a tuple of (origin, direction, pixel_color)
        super().__init__(origins_flat, dirs_flat, colors_flat, depths_flat)

class NeRF2D_Datamodule(pl.LightningDataModule):

    def __init__(
            self, folder: Path,
            batch_size=100,
            camera_subset=False,
            camera_subset_n=5,
            t_far=6
    ):
        super().__init__()

        self.save_hyperparameters(ignore=['folder'])

        train_folder = folder / 'train'
        test_folder = folder / 'test'
        val_folder = folder / 'val'

        # read training images and poses
        self.train_ims, self.train_poses, self.train_focal, self.train_depths = read_image_folder(train_folder,
                                                                                                  t_far=t_far)
        self.train_height = self.train_ims.shape[2]

        if camera_subset:
            self.train_ims = get_n_evenly_spaced_views(self.train_ims, camera_subset_n)
            self.train_poses = get_n_evenly_spaced_views(self.train_poses, camera_subset_n)
            self.train_depths = get_n_evenly_spaced_views(self.train_depths, camera_subset_n)

        self.train_dataset = NeRFDataset2D(self.train_ims, self.train_poses, self.train_focal, self.train_depths)

        # read test images and poses
        self.test_ims, self.test_poses, self.test_focal, self.test_depths = read_image_folder(test_folder,
                                                                                              t_far=t_far)
        self.test_height = self.test_ims.shape[2]
        self.test_dataset = NeRFDataset2D(self.test_ims, self.test_poses, self.test_focal, self.test_depths)

        # read val images and poses
        self.val_ims, self.val_poses, self.val_focal, self.val_depths = read_image_folder(val_folder, t_far)
        self.val_height = self.test_ims.shape[2]
        self.val_dataset = NeRFDataset2D(self.val_ims, self.val_poses, self.val_focal, self.val_depths)

        # save additional hyperparams
        self.hparams.n_train_images = len(self.train_dataset)
        self.hparams.image_resolution = self.train_dataset.image_resolution

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, num_workers=15,
                          persistent_workers=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.test_height, num_workers=15, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.test_height, num_workers=15, persistent_workers=True)
