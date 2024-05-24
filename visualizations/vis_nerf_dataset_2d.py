from pathlib import Path

import rerun as rr
import torch

import rerun_util as ru
from nerf2d_dataset import NeRFDataset2D, read_image_folder
from transform2d import Transform2D

folder = '/home/jorge/repos/NeRF2D/data/cube/train'
image_width = 10
ray_length = 1.6

# Load the dataset
rr.init('vis_dataset_2d', spawn=True)

# read dataset
ims, poses, focal = read_image_folder(Path(folder))
dataset = NeRFDataset2D(ims, poses, focal)

# log all rays in dataset
rays = [dataset[i] for i in range(len(dataset))]
origins = torch.stack([ray[0] for ray in rays])
directions = torch.stack([ray[1] for ray in rays])
colors = torch.stack([ray[2] for ray in rays])

rr.log(f'rays', rr.Arrows3D(
    origins=ru.embed_Points2D(origins),
    vectors=ru.embed_Points2D(directions * ray_length),
    colors=colors
))

for i in range(len(dataset.ims)):
    # read transform
    c2w = torch.Tensor(dataset.poses[i])

    # read image
    im = dataset.ims[i]

    # log camera
    rr.log(
        f'cam{i}',
        rr.Pinhole(
            height=dataset.image_resolution,
            width=image_width,
            focal_length=focal,
            camera_xyz=rr.ViewCoordinates.FUR
        )
    )

    # Transform camera
    c2w_transform = Transform2D.from_matrix(c2w)
    rr.log(f'cam{i}', ru.embed_Transform2D(c2w_transform))

    # Log image
    # im_repeated = repeat(im, 'c h -> h w c', w=image_width)
    # rr.log(f'world/cam{i}', rr.Image(im_repeated))
