import numpy as np
import rerun as rr
import torch

import rerun_util as ru
from camera_model_2d import projection_matrix, unembed_homog, get_rays, embed_homog
from transform2d import Transform2D

def rect_points(center: torch.Tensor, w: int, h: int):
    """
    Return a list of 2D points corresponding to rectangle corners
    """

    half_w = w / 2
    half_h = h / 2
    points = torch.Tensor([[-half_w, half_h],
                           [-half_w, -half_h],
                           [half_w, -half_h],
                           [half_w, half_h]])
    return points + center

def as_mesh(points):
    strip = torch.cat([points, points[0].unsqueeze(0)], dim=0)
    return rr.LineStrips3D([ru.embed_Points2D(strip)])

rr.init('cam2d', spawn=True)

f = 3
h = 3
t = torch.Tensor([0, 2])
r = np.radians(-70)

c2w = Transform2D.from_translation_and_rotation(t, r)
w2c = c2w.inverse()

rr.log('world', rr.Transform3D(translation=[0, 0, 0]))

# log camera
rr.log('world/cam', rr.Pinhole(height=h, focal_length=f, width=0, camera_xyz=ru.CAM_2D))
rr.log('world/cam', ru.embed_Transform2D(c2w))

# generate random points
N = 20
p = rect_points(torch.Tensor([1, 0]), 0.5, 0.5)

rr.log('world/points', as_mesh(p))

# embed to homogeneous space
p_homog = embed_homog(p)

P = projection_matrix(f)
matrix = P @ w2c.as_matrix()

p_projected_homog = p_homog @ matrix.T

# unembed to 1D coordinates
p_projected = unembed_homog(p_projected_homog)

# get corresponding 2d coords on camera plane
o, d = get_rays(p_projected[:, 0], f, c2w.as_matrix())
p_projected_2d = o + 0.1 * d

rr.log('world/d', ru.embed_rays(o, d))
