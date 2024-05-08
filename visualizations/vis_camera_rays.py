import numpy as np
import rerun as rr
import torch

import rerun_util as ru
from nerf2d import get_rays2d
from transform2d import Transform2D

rr.init('camera_rays', spawn=True)

height = 100
focal = 100

# Create a 2D transformation
translation = torch.Tensor([0.1, 0.1])
rotation = np.radians(45)
transform_2d = Transform2D.from_translation_and_rotation(translation, rotation)
c2w = transform_2d.as_matrix()

rr.log('world', rr.Transform3D())

rr.log('world/cam', rr.Pinhole(height=height, width=1, focal_length=focal, camera_xyz=rr.ViewCoordinates.FUR))
rr.log('world/cam', ru.embed_Transform2D(transform_2d))

origins, directions = get_rays2d(height, focal, c2w)

rr.log('world/rays', rr.Arrows3D(origins=ru.embed_Points2D(origins),
                                 vectors=ru.embed_Points2D(directions) * 0.1))
