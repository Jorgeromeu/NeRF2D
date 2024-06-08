import numpy as np
import rerun as rr
import torch
from einops import rearrange

from nerf2d import NeRF2D_LightningModule

spatial_res = 100
angle_res = 50
x_lo, x_hi = -3, 3
y_lo, y_hi = -3, 3

output = np.zeros((angle_res, spatial_res, spatial_res, 4))

# create grid
xs = torch.linspace(x_lo, x_hi, spatial_res)
ys = torch.linspace(y_lo, y_hi, spatial_res)
x, y = torch.meshgrid(xs, ys, indexing='ij')
coords = torch.stack([x, y], dim=2)
coords_flat = rearrange(coords, 'h w c -> (h w) c')

viewdirs = torch.linspace(0, 2 * np.pi, angle_res)

torch.set_grad_enabled(False)

nerf = NeRF2D_LightningModule.load_from_checkpoint('../checkpoints/last-v1.ckpt', t_far=6).cpu()
volume = nerf.model

rr.init('radiance_field', spawn=True)

for i, angle in enumerate(viewdirs):
    viewdirs_flat = angle.expand(coords_flat.shape[0], 1)
    outputs_flat = volume(coords_flat, viewdirs_flat).detach().numpy()
    output_square = rearrange(outputs_flat, '(h w) c -> h w c', h=spatial_res)

    output_square *= 255
    rr.log('color', rr.Image(output_square[:, :, 0:3]))
    rr.log('density', rr.Image(output_square[:, :, 3]))

    h, w, _ = output_square.shape

    direction = np.array([np.sin(angle), np.cos(angle)])
    rr.log('dir', rr.Arrows2D(origins=np.array([h // 2, w // 2]), vectors=direction * 10))
