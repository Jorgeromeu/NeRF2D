import rerun as rr
import torch
from scipy.spatial.transform import Rotation

from transform2d import Transform2D

# Common camera coordinate systems
CAM_2D = rr.ViewCoordinates.FUR  # camera faces x-axis
CAM_3D_BLENDER = rr.ViewCoordinates.RIGHT_HAND_Y_UP  # camera faces -z axis, y up

def embed_Points2D(points: torch.Tensor):
    """
    Embed a 2D point cloud into 3D by adding a zero z coordinate
    :param points: 2D point cloud Nx2
    :return: 3d point cloud Nx3
    """

    # add zero z coordinate
    return torch.cat([points, torch.zeros(points.shape[0], 1)], dim=-1)

def embed_Transform2D(transform: Transform2D):
    """
    Embed a 2D transform into 3D
    :param transform: 2D transform
    :return: rr.Transform3D representing the 2D transform
    """

    translation = transform.translation()
    rotation = transform.rotation()

    # add zero-z coord to translation
    translation_3d = [translation[0], translation[1], 0]

    # rotation is 3D rotation around z-axis as quaternion
    rotation_3d_quat = Rotation.from_euler('z', rotation.item(), degrees=False).as_quat()

    return rr.Transform3D(translation=translation_3d, rotation=rotation_3d_quat, scale=1)

def embed_rays(origins, directions, ray_len=1):
    """
    Embed 2D rays into Arrows3D
    """

    return rr.Arrows3D(
        origins=embed_Points2D(origins),
        vectors=embed_Points2D(directions) * ray_len
    )

def pinhole_2D(focal, height):
    return rr.Pinhole(height=height, focal_length=focal, width=1, camera_xyz=CAM_2D)
