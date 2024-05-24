import rerun as rr
import torch
from einops import rearrange, einsum

import rerun_util as ru
from dummy_volume import DummyVolume
from nerf2d import NeRF2D_LightningModule, get_rays2d, volume_rendering_weights

volume_res = 500
res = 100
f = 100
query_point_multiplier = 1

def visualize_volume(
        volume_name: str,
        volume: DummyVolume,
        x_lo=-3, x_hi=3,
        y_lo=-3, y_hi=3,
        res=100
):
    # create grid
    xs = torch.linspace(x_lo, x_hi, res)
    ys = torch.linspace(y_lo, y_hi, res)
    x, y = torch.meshgrid(xs, ys, indexing='ij')
    coords = torch.stack([x, y], dim=2)

    # sample volume at grid points
    coords_flat = rearrange(coords, 'h w c -> (h w) c')
    viewdirs = torch.zeros(coords_flat.shape[0], 1)
    outputs_flat = volume(coords_flat, viewdirs).detach().numpy()

    # visualize points
    rr.log(volume_name, rr.Points3D(ru.embed_Points2D(coords_flat), colors=outputs_flat))

def visualize_rendering(res, f, c2w, volume, nerf: NeRF2D_LightningModule):
    # visualize camera
    rr.log('camera', rr.Pinhole(height=res, width=1, focal_length=f, camera_xyz=ru.CAM_2D))
    rr.log('camera', ru.embed_Transform2D(c2w))

    image = nerf.render_view(res, f, c2w.as_matrix())
    image = rearrange(image, 'c h w -> h w c')

    rr.log('camera', rr.Image(image))

    # get camera rays
    o, d = get_rays2d(res, f, c2w.as_matrix())

    # # visualize rays

    # rr.log('rays', rr.Arrows3D(origins=ru.embed_Points2D(o), vectors=ru.embed_Points2D(d)))
    #
    # # get query points
    angles, points, ts = nerf.compute_query_points(o, d)

    # query network
    points_flat = rearrange(points, 'n t d -> (n t) d')
    outputs_flat = volume(points_flat, angles)
    outputs = rearrange(outputs_flat, '(n t) c -> n t c', n=res)
    colors = outputs[:, :, 0:3]
    densities = outputs[:, :, 3]

    # compute volume rendering weights
    weights = volume_rendering_weights(ts, densities)

    # visualize query points: size is weight color is color
    weights_flat = rearrange(weights, 'n t -> (n t)')
    rr.log(
        'points',
        rr.Points3D(
            ru.embed_Points2D(points_flat),
            radii=weights_flat.detach().numpy() * query_point_multiplier,
            colors=outputs_flat[:, 0:3].detach().numpy()
        )
    )

    # render colors
    rendered_colors = einsum(weights, colors, 'n t, n t c -> n c')

    # show image on camera
    image = rearrange(rendered_colors, 'h c -> h 1 c')

rr.init('render_volume', spawn=True)

nerf = NeRF2D_LightningModule.load_from_checkpoint('../checkpoints/last-v16.ckpt', t_far=6).cpu()

visualize_volume('volume', nerf.model, res=volume_res)

# def uniform_spaced_circle(radius, num_points):
#     angles = np.linspace(0, 2 * np.pi, num_points + 1)[:num_points]
#     x = radius * np.cos(angles)
#     y = radius * np.sin(angles)
#
#     pos = np.stack([x, y], axis=1)
#     dirs = -pos / np.linalg.norm(pos)
#
#     return pos, angles + np.pi
#
# pos, angles = uniform_spaced_circle(4, 100)
#
# # volume = DummyVolume(density=10)
# volume = nerf.model
#
# # visualize volume
# # visualize_volume('volume', volume, res=100)
#
# for i in range(len(pos)):
#     translation = torch.tensor(pos[i])
#     rotation = angles[i]
#
#     c2w = Transform2D.from_translation_and_rotation(translation, rotation)
#
#     visualize_rendering(res, f, c2w, volume, nerf)
