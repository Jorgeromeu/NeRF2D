from pathlib import Path

import rerun as rr
import torch
from einops import rearrange, einsum

import rerun_util as ru
from camera_model_2d import pixel_center_rays
from nerf2d import NeRF2D_LightningModule, volume_rendering_weights
from nerf2d_dataset import read_image_folder, NeRFDataset2D
from transform2d import Transform2D

ckpt = '../checkpoints/last-v4.ckpt'
dataset_path = '../data/cube/val'
nerf = NeRF2D_LightningModule.load_from_checkpoint(
    ckpt,
).cpu()
cam_idx = 20

ims, poses, focal, depths = read_image_folder(Path(dataset_path))
height = ims[0].shape[1]
dataset = NeRFDataset2D(ims, poses, focal, depths)

rr.init('vis_rendering', spawn=True)

c2w = Transform2D.from_matrix(poses[cam_idx])

def visualize_rendering(focal, cw2, height):
    # log camera
    rr.log('cam', ru.pinhole_2D(focal, height))
    rr.log('cam', ru.embed_Transform2D(c2w))

    # render rays for view
    origins, directions = pixel_center_rays(height, focal, c2w.as_matrix())

    # rr.log('rays', ru.embed_rays(origins, directions))

    # query points
    query_angles, query_points, ts = nerf.compute_query_points(origins, directions)

    # flatten query points and angles
    points_flat = rearrange(query_points, 'n t d -> (n t) d')
    angles_flat = rearrange(query_angles, 'n t d -> (n t) d')

    # query network
    with torch.no_grad():
        outputs_flat = nerf.model(points_flat, angles_flat)

    # unflatten outputs
    outputs = rearrange(outputs_flat, '(n t) c -> n t c', n=origins.shape[0])

    # split outputs into colors and densities
    colors = outputs[:, :, 0:3]
    densities = outputs[:, :, 3]

    # compute volume rendering weights
    weights = volume_rendering_weights(ts, densities)

    weights_flat = rearrange(weights, 'n t -> (n t)')
    colors_flat = rearrange(colors, 'n t c -> (n t) c')
    densities_flat = rearrange(densities, 'n t -> (n t)')

    rr.log(
        'points',
        rr.Points3D(
            ru.embed_Points2D(points_flat),
            radii=weights_flat.detach().numpy() / 5,
            colors=colors_flat.detach().numpy()
        )
    )

    rendered_colors = einsum(weights, colors, 'n t, n t c -> n c')
    rendered_depths = einsum(weights, ts, 'n t, t -> n')

    rr.log('cam/rendered', rr.Image(rearrange(rendered_colors, 'n c -> n 1 c')))

for cam_idx in range(len(ims)):
    c2w = Transform2D.from_matrix(poses[cam_idx])
    visualize_rendering(focal, c2w, height)
