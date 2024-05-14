import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from einops import repeat, rearrange
from torch import nn, Tensor

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

def get_rays2d(height: int, focal_length_px: float, c2w: torch.Tensor):
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

class NeRF2D_LightningModule(pl.LightningModule):

    def __init__(
            self,
            lr=1e-4,
            t_near=2,
            t_far=6
    ):
        super().__init__()

        self.save_hyperparameters()

        # dummy model
        self.model = nn.Linear(3, 4)

        # metrics
        self.train_mse = torchmetrics.MeanSquaredError()

    def render_view(self, height: int, focal_length_px: float, c2w: Tensor):
        """
        Render a view from a 2D camera
        :param height:
        :param focal_length_px:
        :param c2w:
        :return:
        """

        # render rays for view
        colors = self.forward(*get_rays2d(height, focal_length_px, c2w))

        # reshape as image
        colors_im = rearrange(colors, 'h c -> h 1 c').detach().numpy()
        return colors_im

    def forward(
            self,
            origins: Tensor,
            directions: Tensor,
    ) -> Tensor:
        """
        Perform volume rendering on model given a set of rays
        :param origins: origins of rays N, 3
        :param directions: directions of rays N, 3
        :return: rendered-colors at rays N, 3
        """

        # TODO replace with volume rendering
        return F.sigmoid(torch.randn(origins.shape[0], 3, requires_grad=True).to(origins))

    def training_step(self, batch, batch_idx):
        origins, directions, colors = batch

        # forward pass
        colors_pred = self.nerf_forward(origins, directions)

        # compute loss
        loss = self.train_mse(colors_pred, colors)

        self.log('train_loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
