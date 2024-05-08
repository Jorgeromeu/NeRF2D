import numpy as np
import rerun as rr

import rerun_util

rr.init('cameras', spawn=True)

data = np.load('../data/tiny_nerf_data.npz')
ims = data['images']
poses = data['poses']
f = data['focal'].item()
h, w = 100, 100

for i in range(len(ims)):
    pose = poses[i]
    translation = pose[:3, 3]
    rotation = pose[:3, :3]

    print(ims[i].shape)

    rr.log(
        f'world/cam{i}',
        rr.Pinhole(height=h, width=w, focal_length=f, camera_xyz=rerun_util.CAM_3D_BLENDER)
    )
    rr.log(f'world/cam{i}', rr.Transform3D(translation=translation, mat3x3=rotation))
    rr.log(f'world/cam{i}', rr.Image(ims[i]))

# rr.log('world/cam', rr.Pinhole(height=h, width=w, focal_length=f, camera_xyz=rr.ViewCoordinates.RIGHT_HAND_Y_UP))
# rr.log('world/cam', rr.Transform3D())
