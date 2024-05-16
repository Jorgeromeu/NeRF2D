import numpy as np
import rerun as rr
import torch
from einops import repeat, rearrange

import rerun_util as ru
from nerf2d import get_rays2d, NeRF2D_LightningModule
from transform2d import Transform2D

rr.init('camera_rays', spawn=True)

height = 20
focal = 20
fake_width = 3
ray_len = 0.1
dir_len = 0.02
near = 0.5
far = 1
n_steps = 10

# Create a 2D transformation
translation = torch.Tensor([0.01, 0.01])
rotation = np.radians(45)
transform_2d = Transform2D.from_translation_and_rotation(translation, rotation)
c2w = transform_2d.as_matrix()

# compute rays
origins, directions = get_rays2d(height, focal, c2w)

# create fake image and ray_colors
colors = torch.linspace(0, 1, height).unsqueeze(1).repeat(1, 3)
im = repeat(colors, 'h c -> h w c', w=fake_width)

# world frame
rr.log('world', rr.Transform3D())

# visualize camera
rr.log('world/cam', rr.Pinhole(height=height, width=fake_width, focal_length=focal, camera_xyz=ru.CAM_2D))
rr.log('world/cam', ru.embed_Transform2D(transform_2d))
rr.log('world/cam', rr.Image(im))

# visualize rays
rr.log('world/rays', rr.Arrows3D(origins=ru.embed_Points2D(origins),
                                 vectors=ru.embed_Points2D(directions) * ray_len,
                                 colors=colors))

nerf = NeRF2D_LightningModule(
    t_near=near,
    t_far=far,
    n_steps=n_steps,
)

query_dirs, query_points, ts = nerf.compute_query_points(origins, directions)

query_angles_flat = rearrange(query_dirs, 'n t d -> (n t) d')
cosines = torch.cos(query_angles_flat)
sines = torch.sin(query_angles_flat)
dirs_flat = torch.cat((cosines, sines), dim=-1)

points_flat = rearrange(query_points, 'n t d -> (n t) d')

# rr.log('world/query_points', rr.Points3D(ru.embed_Points2D(points_flat), radii=0.01))

rr.log(
    'world/query_dirs',
    rr.Arrows3D(
        origins=ru.embed_Points2D(points_flat),
        vectors=ru.embed_Points2D(dirs_flat) * dir_len,
    )
)
