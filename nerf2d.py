import torch
import torch.nn.functional as F
from einops import repeat

def get_rays2d(height, focal_length_px, c2w):
    """
    Compute the set of rays for a 2D camera
    :param height: amount of pixels
    :param focal_length_px: focal length in pixel space
    :param c2w: camera to world matrix
    :return:
    """

    # extract translation and rotation
    translation, rotation = c2w[:2, 2], c2w[:2, :2]

    # compute ray-directions
    i = torch.arange(height)
    ys = (i - height * 0.5) / focal_length_px
    xs = torch.ones_like(i)
    directions = F.normalize(torch.stack([xs, ys], dim=1), dim=1)
    directions = directions.__reversed__()

    # apply translation to origins
    origins = repeat(translation, 'd -> n d', n=height)

    # apply rotation to directions
    directions = directions @ rotation.T

    return origins, directions
