import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import repeat, rearrange, einsum
from torch import Tensor

import wandb
from nerf_model import NeRF

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

def sample_stratified(near, far, n_samples):
    bin_borders = torch.linspace(near, far, n_samples + 1)
    lowers = bin_borders[:-1]
    bin_width = (far - near) / n_samples

    ts = torch.rand(n_samples) * bin_width + lowers
    return ts

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

    delta = ts[1:] - ts[:-1]
    delta = torch.cat([delta, delta[-1:]], dim=0)

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
            n_freqs_pos=8,
            n_freqs_dir=4,
            n_layers=4,
            d_hidden=128,
            t_near=2.,
            t_far=6.,
            n_steps=64,
            chunk_size=30000,
            n_gt_poses=100,
            depth_loss_weight=0.5,
    ):

        super().__init__()

        self.save_hyperparameters()

        # dummy model
        self.model = NeRF(
            d_pos_input=2,
            d_dir_input=1,
            n_freqs_position=n_freqs_pos,
            n_freqs_direction=n_freqs_dir,
            n_layers=n_layers,
            d_hidden=d_hidden,
            skip_indices=[n_layers // 2]
        )

        # metrics
        self.color_loss = torch.nn.MSELoss()
        self.depth_loss_weight = depth_loss_weight


    def compute_query_points(self, origins: Tensor, directions: Tensor):
        """
        Compute query points and depth values for a batch of rays
        :param origins: N, 3
        :param directions: N, 3
        :return: query points N, 3 and depth values N
        """

        # stratified sample t-points
        ts = sample_stratified(self.hparams.t_near,
                               self.hparams.t_far,
                               self.hparams.n_steps).to(origins)

        # compute query points for each ray
        query_points = origins[:, None, :] + directions[:, None, :] * ts[None, :, None]
        query_dirs = repeat(directions, 'b d -> b t d', t=len(ts))

        angles = torch.atan2(query_dirs[:, :, 1], query_dirs[:, :, 0]).unsqueeze(-1)

        return angles, query_points, ts

    def query_network_chunked(self, query_points: Tensor):

        if query_points.shape[0] <= self.hparams.chunk_size:
            return self.model(query_points, None)

        n_chunks = len(query_points) // self.hparams.chunk_size + 1
        chunks = torch.chunk(query_points, n_chunks, dim=0)

        outputs = []
        for chunk in chunks:
            out = self.model(chunk)
            outputs.append(out)

        return torch.cat(outputs, dim=0)

    def forward(
            self,
            origins: Tensor,
            directions: Tensor,
    ) -> (Tensor, Tensor, Tensor):
        """
        Perform volume rendering on model given a set of rays
        :param origins: origins of rays N, 3
        :param directions: directions of rays N, 3
        :return: rendered-colors at rays N, 3
        """

        # query points
        query_angles, query_points, ts = self.compute_query_points(origins, directions)

        # TODO query network in chunks
        points_flat = rearrange(query_points, 'n t d -> (n t) d')
        angles_flat = rearrange(query_angles, 'n t d -> (n t) d')
        # outputs_flat = self.query_network_chunked(points_flat)
        outputs_flat = self.model(points_flat, angles_flat)
        outputs = rearrange(outputs_flat, '(n t) c -> n t c', n=origins.shape[0])
        colors = outputs[:, :, 0:3]
        densities = outputs[:, :, 3]

        # compute volume rendering weights
        weights = volume_rendering_weights(ts, densities)

        # render colors by weighting samples
        rendered_colors = einsum(weights, colors, 'n t, n t c -> n c')

        # return the rendered color for each ray
        return rendered_colors, weights, ts

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

        colors, _, _ = self.forward(origins, directions)

        # reshape as image
        colors_im = rearrange(colors, 'h c -> c h 1')
        return colors_im
    
    def depth_loss(self, gt_depth, ts, weights, sigma=1.0):
        """
        Compute the depth loss for a single pixel.

        Args:
            ts (torch.Tensor): Depth values along the pixel ray (shape: [100]).
            weights (torch.Tensor): Weights along the pixel ray (shape: [1, 100]).
            depth (torch.Tensor): Ground truth depth of the pixel ray (shape: [1, 1]).
            sigma (float): Standard deviation for the Gaussian weighting, default is 1.0.

        Returns:
            torch.Tensor: The computed depth loss (shape: [1]).
        """
        device = gt_depth.device

        # Compute the intervals between sample points
        dists = ts[1:] - ts[:-1]
        dists = torch.cat([dists, torch.tensor([dists.max()]).to(device=device)])  # Last interval is a large number to avoid boundary issues

        # Compute the Gaussian weighting term
        gauss_weight = torch.exp(-0.5 * ((ts - gt_depth) ** 2) / (sigma ** 2))  # Shape: [100]

        # Compute the log probability term
        log_prob = torch.log(weights + 1e-5)  # Shape: [1, 100]

        # Compute the loss by summing over all sampled points, incorporating the intervals
        loss = -torch.sum(log_prob * gauss_weight * dists)  # Shape: []

        return loss

    def training_step(self, batch, batch_idx):
        origins, directions, colors, gt_depth = batch
        # forward pass
        colors_pred, weights, ts = self(origins, directions)

        # compute loss
        color_loss = self.color_loss(colors_pred, colors)
        depth_loss = self.depth_loss(gt_depth, ts, weights)
        depth_loss = depth_loss / 5000
        total_loss = (1 - self.depth_loss_weight) * color_loss + self.depth_loss_weight * depth_loss
        
        # log the losses
        self.log('train_color_loss', color_loss)
        self.log('train_depth_loss', depth_loss)
        self.log('train_loss', total_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        origins, directions, colors, gt_depth = batch
        # forward pass
        colors_pred, weights, ts = self(origins, directions)

        self.rendered_views.append(colors_pred)

        # compute loss
        color_loss = self.color_loss(colors_pred, colors)
        depth_loss = self.depth_loss(gt_depth, ts, weights)
        depth_loss = depth_loss / 5000
        loss = (1 - self.depth_loss_weight) * color_loss + self.depth_loss_weight * depth_loss

        # log the losses
        self.log('val_color_loss', color_loss)
        self.log('val_depth_loss', depth_loss)
        self.log('val_loss', loss)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # Logging Methods

    def on_validation_epoch_end(self) -> None:

        # stack validation renders into a single image
        all_renders = torch.stack(self.rendered_views, dim=1).detach().cpu()
        all_renders = rearrange(all_renders, 'h w c -> c h w')
        all_renders = TF.resize(all_renders, (256, 256), interpolation=TF.InterpolationMode.NEAREST)
        all_renders = TF.to_pil_image(all_renders)

        self.trainer.logger.experiment.log({
            'renders': wandb.Image(all_renders)
        })

    def on_validation_epoch_start(self) -> None:
        # clear validation renders for epoch
        self.rendered_views = []

    def on_train_start(self) -> None:
        # log ground-truth renders
        val_dataloader = self.trainer.val_dataloaders

        gt_views = []
        for batch in val_dataloader:
            _, _, colors, _ = batch
            gt_views.append(colors)

        gt_all = torch.stack(gt_views, dim=1).detach().cpu()

        gt_all = rearrange(gt_all, 'h w c -> c h w')
        gt_all = TF.resize(gt_all, (256, 256), interpolation=TF.InterpolationMode.NEAREST)
        gt_all = TF.to_pil_image(gt_all)

        self.trainer.logger.experiment.log({
            'renders_gt': wandb.Image(gt_all)
        })
