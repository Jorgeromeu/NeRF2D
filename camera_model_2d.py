import rerun as rr
import torch
from einops import repeat
from torch import Tensor

import visualizations.rerun_util as ru

def project_gt(p: Tensor, focal_length: float):
    return (p[:, 1] * focal_length / p[:, 0]).unsqueeze(-1)

def embed_homog(points: Tensor):
    """
    Embed points into homogeneous space
    :param points: N, D tensor
    :return: N, D+1 tensor
    """
    p_homog = torch.cat([points, torch.ones(points.shape[0], 1)], dim=1)
    return p_homog

def unembed_homog(points: Tensor):
    """
    Unembed points from homogeneous space
    :param points: N, D+1 tensor
    :return: N, D tensor
    """
    return points[:, :-1] / points[:, -1].unsqueeze(-1)

def get_rays(pixel_coords: Tensor, focal_length_px: float, c2w: Tensor):
    """
    Compute rays, mapping to a pixel coordinate
    """

    trans_c2w, rot_c2w = c2w[:2, 2], c2w[:2, :2]

    y_coords = pixel_coords
    x_coords = torch.ones_like(y_coords) * focal_length_px

    # compute directions to each coord
    directions = torch.stack([x_coords, y_coords], dim=1)

    # origin is c2w_translation for all rays
    origins = repeat(trans_c2w, 'd -> n d', n=pixel_coords.shape[0])

    # apply c2w rotation to directions
    directions = directions @ rot_c2w.T

    return origins, directions

def projection_matrix(focal_length):
    return torch.Tensor([[0, 1, 0],
                         [1 / focal_length, 0, 0]])

def generate_random_point_cloud(N, lo_x=1, hi_x=2, lo_y=-0.5, hi_y=0.5):
    p = torch.rand(N, 2)
    p[:, 0] = (hi_x - lo_x) * p[:, 0] + lo_x
    p[:, 1] = (hi_y - lo_y) * p[:, 1] + lo_y
    return p


