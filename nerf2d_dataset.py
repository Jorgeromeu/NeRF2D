import json
from pathlib import Path

import torch
from einops import rearrange
from torch.utils.data import TensorDataset
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

        self.h = self.ims.shape[2]

        # get a ray for each pixel
        rays = [get_rays2d(self.h, self.focal_length, c2w) for c2w in self.poses]
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
