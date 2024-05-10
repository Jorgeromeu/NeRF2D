import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from einops import repeat, rearrange, einsum
from torch import Tensor

from nerf_model import SimpleNeRF

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
    directions = directions.__reversed__().to(c2w)

    # origin is c2w_translation for all rays
    origins = repeat(trans_c2w, 'd -> n d', n=height)

    # apply c2w rotation to directions
    directions = directions @ rot_c2w.T

    return origins, directions

def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    dim = -1
    cumprod = torch.cumprod(tensor, dim)
    cumprod = torch.roll(cumprod, 1, dim)
    cumprod[..., 0] = 1.

    return cumprod

def volume_rendering_weights(ts, densities):
    """
    Compute volume rendering weights for a batch of rays
    :param ts: T
    :param densities: N, T
    :return: weights: N, T
    """

    delta = torch.cat([ts[1:] - ts[:-1], torch.Tensor([1]).to(ts)])
    delta_density = einsum(densities, delta, 'n t, t -> n t')
    alpha = 1 - torch.exp(-delta_density)
    transmittances = cumprod_exclusive(1 - alpha + 1e-10)
    weights = alpha * transmittances

    return weights

# noinspection PyAttributeOutsideInit
class NeRF2D_LightningModule(pl.LightningModule):

    def __init__(
            self,
            lr=1e-4,
            t_near=2.,
            t_far=6.,
            n_steps=64,
            n_gt_poses=100
    ):
        super().__init__()

        self.save_hyperparameters()

        # dummy model
        self.model = SimpleNeRF(d_input=2, d_output=4, n_layers=4, d_hidden=128)

        # metrics
        self.criterion = torch.nn.MSELoss()

    def compute_query_points(self, origins: Tensor, directions: Tensor):
        """
        Compute query points and depth values for a batch of rays
        :param origins: N, 3
        :param directions: N, 3
        :return: query points N, 3 and depth values N
        """

        # evenly space t-values from t_near to t_far
        ts = torch.linspace(self.hparams.t_near, self.hparams.t_far, self.hparams.n_steps).to(origins)

        # compute query points for each ray
        query_points = origins[:, None, :] + directions[:, None, :] * ts[None, :, None]

        return query_points, ts

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

        # query points
        query_points, ts = self.compute_query_points(origins, directions)

        # TODO query network in chunks
        points_flat = rearrange(query_points, 'n t d -> (n t) d')
        outputs_flat = self.model(points_flat)
        outputs = rearrange(outputs_flat, '(n t) c -> n t c', n=origins.shape[0])
        colors = outputs[:, :, 0:3]
        densities = outputs[:, :, 3]

        # compute volume rendering weights
        weights = volume_rendering_weights(ts, densities)

        # render colors by weighting samples
        rendered_colors = einsum(weights, colors, 'n t, n t c -> n c')

        return rendered_colors

    def render_view(self, height: int, focal_length_px: float, c2w: Tensor):
        """
        Render a view from a 2D camera
        :param height:
        :param focal_length_px:
        :param c2w:
        :return:
        """

        # render rays for view
        origins, directions = get_rays2d(height, focal_length_px, c2w)
        origins = origins.to(self.device)
        directions = directions.to(self.device)

        colors = self.forward(origins, directions)

        # reshape as image
        colors_im = rearrange(colors, 'h c -> c h 1')
        return colors_im

    def render_views(self, height: int, focal_length_px: float, c2ws: Tensor):
        """
        Render a list of views and concatenate them into 2D image
        :param height:
        :param focal_length_px:
        :param c2ws: N, 3, 3
        :return:
        """

        renders = torch.stack([self.render_view(height, focal_length_px, pose)
                               for pose in c2ws])

        # concatenate renders to image
        renders_im = rearrange(renders, 'w c h 1 -> h w c')

        return renders_im

    def training_step(self, batch, batch_idx):
        origins, directions, colors = batch
        # forward pass
        colors_pred = self(origins, directions)
        # compute loss
        loss = self.criterion(colors_pred, colors)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        origins, directions, colors = batch
        # forward pass
        colors_pred = self(origins, directions)
        # compute loss
        loss = self.criterion(colors_pred, colors)
        self.log('val_loss', loss)
        return loss

    def on_train_start(self) -> None:
        self.val_height = self.trainer.datamodule.h
        self.val_focal = self.trainer.datamodule.focal
        poses_all = self.trainer.datamodule.poses
        self.val_poses = poses_all[::len(poses_all) // self.hparams.n_gt_poses]

        gt_renders = self.trainer.datamodule.ims[::len(poses_all) // self.hparams.n_gt_poses]
        self.gt_renders = rearrange(gt_renders, 'w c h 1 -> h w c').detach().cpu().numpy()

        self.trainer.logger.experiment.log({
            'gt_renders': wandb.Image(self.gt_renders)
        })

    def on_train_epoch_end(self) -> None:
        if self.current_epoch % 10 == 0:
            renders = self.render_views(self.val_height, self.val_focal, self.val_poses).detach().cpu().numpy()

            self.trainer.logger.experiment.log({
                'renders': wandb.Image(renders)
            })

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
