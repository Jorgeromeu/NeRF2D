import torch
import torch.nn.functional as F
from einops import repeat
from torch import Tensor

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

def camera_rays(pixel_coords: Tensor, focal_length_px: float, c2w: Tensor, normalized=True):
    """
    Get rays for a given set of pixel coordinates
    :param: pixel_coords N tensor of pixel coordinates
    """

    trans_c2w, rot_c2w = c2w[:2, 2], c2w[:2, :2]

    y_coords = pixel_coords
    x_coords = torch.ones_like(y_coords) * focal_length_px

    # compute directions to each coord
    directions = torch.stack([x_coords, y_coords], dim=1)

    if normalized:
        directions = F.normalize(directions, dim=1)

    # origin is c2w_translation for all rays
    origins = repeat(trans_c2w, 'd -> n d', n=pixel_coords.shape[0])

    # apply c2w rotation to directions
    directions = directions @ rot_c2w.T

    return origins, directions

def pixel_centers(lo, hi, n_pixels):
    """
    Compute the center of each pixel in a range
    :param lo: coordinate of lower-point of range
    :param hi: coordinate of upper-point of range
    :param n_pixels: numer of pixels
    :return: coordinates of pixel centers
    """
    pixel_w = (hi - lo) / n_pixels
    return torch.linspace(lo, hi - pixel_w, n_pixels) + pixel_w / 2

def pixel_center_rays(height: int, focal_length_px: float, c2w: torch.Tensor):
    """
    Compute the set of rays for a 2D camera
    :param height: amount of pixels
    :param focal_length_px: focal length in pixel space
    :param c2w: camera to world matrix
    :return:
    """

    trans_c2w, rot_c2w = c2w[:2, 2], c2w[:2, :2]

    # compute pixel center coordinates
    half_height = height / 2
    ys = pixel_centers(half_height, -half_height, height)

    # get camera rays for pixel centers
    return camera_rays(ys, focal_length_px, c2w)

def _project_gt(p: Tensor, focal_length: float):
    """
    Ground truth projection function for testing purposes should be equal
    to applying projection matrix in homogeneous coords
    :param p: N, 2 tensor of points
    :param focal_length: focal length
    :return: N, 1 tensor of projected points
    """
    return (p[:, 1] * focal_length / p[:, 0]).unsqueeze(-1)

def projection_matrix(focal_length):
    return torch.Tensor([[0, 1, 0],
                         [1 / focal_length, 0, 0]])

def project(points: Tensor, focal_length: float, w2c: Tensor):
    """
    Implement camera projection
    """

    # construct matrix
    proj_matrix = projection_matrix(focal_length)
    matrix = proj_matrix @ w2c

    # embed points to 2D homogeneous space
    p_homog = embed_homog(points)
    # apply matrix to points
    p_projected_homog = p_homog @ matrix.T
    # unembed to 1D coordinates
    p_projected = unembed_homog(p_projected_homog)

    return p_projected
