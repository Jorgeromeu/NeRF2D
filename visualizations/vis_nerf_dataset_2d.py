from pathlib import Path

import rerun as rr
import torch

import rerun_util as ru
from nerf2d_dataset import NeRFDataset2D, read_image_folder
from transform2d import Transform2D

data_folder = Path('../data/')
scene = 'bunny'
split = 'test'

folder = data_folder / scene / split
n_rays = 2000
shuffle = True
sample = True

# Load the dataset
rr.init('vis_nerf_dataset_2d', spawn=True)

# read dataset
ims, poses, focal, depths = read_image_folder(Path(folder))
dataset = NeRFDataset2D(ims, poses, focal, depths)

# get rays, colors and depths
origins = torch.stack([ray[0] for ray in dataset])
dirs = torch.stack([ray[1] for ray in dataset])
colors = torch.stack([ray[2] for ray in dataset])
depths = torch.Tensor([ray[3].item() for ray in dataset]).unsqueeze(-1)

if shuffle:
    # shuffle all arrays
    perm = torch.randperm(len(origins))
    origins = origins[perm]
    dirs = dirs[perm]
    colors = colors[perm]
    depths = depths[perm]

if sample:
    origins = origins[:n_rays]
    dirs = dirs[:n_rays]
    colors = colors[:n_rays]
    depths = depths[:n_rays]

print(depths.max())

rr.log(
    'rays',
    rr.Arrows3D(
        origins=ru.embed_Points2D(origins),
        vectors=ru.embed_Points2D(dirs) * depths,
        colors=colors
    )
)

for i in range(len(ims)):
    c2w = poses[i]

    rr.log(
        f'world/cam{i}',
        rr.Pinhole(height=dataset.image_resolution,
                   width=1,
                   focal_length=focal,
                   camera_xyz=ru.CAM_2D)
    )

    c2w_transform = Transform2D.from_matrix(c2w)
    rr.log(f'world/cam{i}', ru.embed_Transform2D(c2w_transform))
