import torch
import torch.nn.functional as F
from einops import repeat

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

def get_rays2d(height, focal_length_px, c2w):
    """
    Compute the set of rays for a 2D camera
    :param height: amount of pixels
    :param focal_length_px: focal length in pixel space
    :param c2w: camera to world matrix
    :return:
    """

    trans_c2w, rot_c2w = c2w[:2, 2], c2w[:2, :2]

    # y-coords: center of each pixel in camera-space
    # x-coords: focal-length
    half_height = height / 2
    ys = pixel_centers(-half_height, half_height, height)
    xs = torch.ones_like(ys) * focal_length_px

    # normalize directions
    directions = F.normalize(torch.stack([xs, ys], dim=1), dim=1)
    directions = directions.__reversed__()

    # origin is c2w_translation for all rays
    origins = repeat(trans_c2w, 'd -> n d', n=height)

    # apply c2w rotation to directions
    directions = directions @ rot_c2w.T

    return origins, directions
