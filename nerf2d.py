from dataclasses import dataclass

import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from einops import repeat, rearrange, einsum
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio

import wandb
from camera_model_2d import pixel_center_rays
from nerf_model import NeRF

def sample_stratified(near, far, n_samples):
    """
    Sample stratified points between near and far
    """

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
    delta = torch.cat([delta, Tensor([1e10]).to(ts)], dim=0)

    delta_density = einsum(densities, delta, 'n t, t -> n t')
    alpha = 1 - torch.exp(-delta_density)
    transmittances = cumprod_exclusive(1 - alpha + 1e-10)
    weights = alpha * transmittances

    return weights

@dataclass
class NeRFOutput:
    """
    Data class for NeRF output tensors
    """

    colors: Tensor
    depths: Tensor
    weights: Tensor
    ts: Tensor

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
            depth_sigma=0.1,
            use_depth_supervision=True,
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

        # loss
        self.color_loss = torch.nn.MSELoss()

        # metrics
        self.val_psnr = PeakSignalNoiseRatio()

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

    def forward(
            self,
            origins: Tensor,
            directions: Tensor,
    ) -> NeRFOutput:

        """
        Perform volume rendering on model given a set of rays
        :param origins: origins of rays N, 2
        :param directions: directions of rays N, 2

        :return: output tensors:
            - rendered_colors: N, 3
            - rendered_depths: N
            - weights: N, T
            - ts: T
        """

        # query points
        query_angles, query_points, ts = self.compute_query_points(origins, directions)

        # flatten query points and angles
        points_flat = rearrange(query_points, 'n t d -> (n t) d')
        angles_flat = rearrange(query_angles, 'n t d -> (n t) d')

        # query network
        outputs_flat = self.model(points_flat, angles_flat)

        # unflatten outputs
        outputs = rearrange(outputs_flat, '(n t) c -> n t c', n=origins.shape[0])

        # split outputs into colors and densities
        colors = outputs[:, :, 0:3]
        densities = outputs[:, :, 3]

        # compute volume rendering weights
        weights = volume_rendering_weights(ts, densities)

        # render colors/depths by weighting samples
        rendered_colors = einsum(weights, colors, 'n t, n t c -> n c')
        rendered_depths = einsum(weights, ts, 'n t, t -> n')

        return NeRFOutput(rendered_colors, rendered_depths, weights, ts)

    def render_view(self, height: int, focal_length_px: float, c2w: Tensor):
        """
        Render a view from a 2D camera
        :param height:
        :param focal_length_px:
        :param c2w:
        :return:
        """

        # render rays for view
        origins, directions = pixel_center_rays(height, focal_length_px, c2w)
        origins = origins.to(self.device)
        directions = directions.to(self.device)

        outs = self.forward(origins, directions)

        # reshape as image
        colors_im = rearrange(outs.colors, 'h c -> c h 1')
        depth_im = rearrange(outs.depths, 'h -> h 1')

        return {
            'rgb': colors_im,
            'depth': depth_im
        }

    def render_density_field(self, res=100, lo=-1.5, hi=1.5):

        # densely sample field
        xs = torch.linspace(lo, hi, res)
        ys = torch.linspace(hi, lo, res)

        x, y = torch.meshgrid(xs, ys, indexing='xy')
        coords = torch.stack([x, y], dim=-1)
        coords_flat = rearrange(coords, 'h w c -> (h w) c')

        # random viewdirs, only care about density
        dirs = torch.randn(coords_flat.shape[0], 1).to()

        # query MLP
        with torch.no_grad():
            densities = self.model(coords_flat.to(self.device), dirs.to(self.device))[:, -1]

        densities = rearrange(densities, '(h w) -> h w', h=res)
        densities = (densities - densities.min()) / (densities.max() - densities.min())

        return densities

    def depth_loss(self, gt_depth: Tensor, ts, weights, sigma=1.0):

        """
        Compute the depth loss for a single pixel.

        :param ts (torch.Tensor): Depth values along the pixel ray (shape: [100]).
        :param weights (torch.Tensor): Weights along the pixel ray (shape: [1, 100]).
        :param depth (torch.Tensor): Ground truth depth of the pixel ray (shape: [1, 1]).
        :param sigma (float): Standard deviation for the Gaussian weighting, default is 1.0.
        :return torch.Tensor: The computed depth loss (shape: [1]).
        """

        # Compute the intervals between sample points
        dists = ts[1:] - ts[:-1]
        dist_pad = Tensor([dists.max()]).to(gt_depth)

        dists = torch.cat([dists, dist_pad])

        # Compute the Gaussian weighting term
        gauss_weight = torch.exp(-0.5 * ((ts - gt_depth) ** 2) / (sigma ** 2))  # Shape: [100]

        # Compute the log probability term
        log_prob = torch.log(weights + 1e-5)  # Shape: [1, 100]

        # Compute the loss by summing over all sampled points, incorporating the intervals
        loss = -torch.sum(log_prob * gauss_weight * dists)  # Shape: []

        return loss

    def compute_losses(self, outs: NeRFOutput, colors_gt: Tensor, gt_depth: Tensor):

        """
        Compute losses after a forward pass
        """

        color_loss = self.color_loss(outs.colors, colors_gt)

        if not self.hparams.use_depth_supervision:
            return {'loss': color_loss, 'color_loss': color_loss}
        else:

            depth_loss = self.depth_loss(gt_depth, outs.ts, outs.weights, self.hparams.depth_sigma) / 2000
            total_loss = (1 - self.hparams.depth_loss_weight) * color_loss + self.hparams.depth_loss_weight * depth_loss

            return {
                'loss': total_loss,
                'color_loss': color_loss,
                'depth_loss': depth_loss,
            }

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):

        origins, directions, colors_gt, depth_gt = batch

        # forward pass
        outs = self(origins, directions)

        # compute losses
        losses = self.compute_losses(outs, colors_gt, depth_gt)

        for k, v in losses.items():
            self.log(f'train_{k}', v)

        return losses['loss']

    def validation_step(self, batch, batch_idx):

        origins, directions, colors_gt, depth_gt = batch

        # forward pass
        outs = self(origins, directions)

        # save rendered views for logging
        self.val_renders.append(outs.colors)
        self.val_depths.append(self.normalize_depth(outs.depths.unsqueeze(-1)))

        # compute losses
        losses = self.compute_losses(outs, colors_gt, depth_gt)

        for k, v in losses.items():
            self.log(f'val_{k}', v)

        # log psnr
        self.log('val_psnr', self.val_psnr(outs.colors, colors_gt))

        return losses['loss']

    def on_validation_epoch_end(self) -> None:
        self.log_views('val_renders', self.val_renders)
        self.log_views('val_depths', self.val_depths)
        self.log_density_field()

    def on_validation_epoch_start(self) -> None:
        # clear validation renders for epoch
        self.val_renders = []
        self.val_depths = []

    def test_step(self, batch, batch_idx):

        origins, directions, colors_gt, depth_gt = batch

        # forward pass
        outs = self(origins, directions)

        # save rendered views for logging
        self.test_renders.append(outs.colors)
        self.test_depths.append(self.normalize_depth(outs.depths.unsqueeze(-1)))

        # log psnr
        self.log('test_psnr', self.val_psnr(outs.colors, colors_gt))

    def on_test_epoch_end(self) -> None:
        self.log_views('test_renders', self.test_renders)
        self.log_views('test_depths', self.test_depths)
        self.log_density_field()

    def on_test_epoch_start(self) -> None:
        # clear test renders
        self.test_renders = []
        self.test_depths = []

    def on_train_start(self) -> None:
        gt_views, gt_depth = self.get_gt_views(self.trainer.val_dataloaders)
        self.log_views('val_renders_gt', gt_views)
        self.log_views('val_depth_gt', gt_depth)

    def on_test_start(self) -> None:
        gt_views, gt_depth = self.get_gt_views(self.trainer.test_dataloaders)
        self.log_views('test_renders_gt', gt_views)
        self.log_views('test_depth_gt', gt_depth)

    def get_gt_views(self, dataloader):

        """
        Get the ground truth views of a dataloader
        """

        # make a copy of dataloader
        new_loader = DataLoader(dataloader.dataset, batch_size=dataloader.batch_size)
        gt_views = [colors for _, _, colors, _ in new_loader]
        gt_depth = [self.normalize_depth(depth) for _, _, _, depth in new_loader]
        return gt_views, gt_depth

    def normalize_depth(self, depth: Tensor):

        t_n = self.hparams.t_near
        t_f = self.hparams.t_far

        clipped = torch.clamp(depth, t_n, t_f)
        normalized = (clipped - t_n) / (t_f - t_n)
        return 1 - normalized

    def remap_depth(self, depth: Tensor):
        # map from 0 to 1 to t_n t_f
        t_n = self.hparams.t_near
        t_f = self.hparams.t_far

        return (1 - depth) * (t_f - t_n) + t_n

    def log_views(self, label: str, views: list[Tensor], size=100):

        """
        Log a list of views to wandb
        :label wandb label
        :param views: list of views of Tensors H, C
        :param size output resolution
        """

        if len(views) == 0:
            return

        views_all = torch.stack(views, dim=1).detach().cpu()
        views_all = rearrange(views_all, 'h w c -> c h w')
        views_all = TF.resize(views_all, (size, size), interpolation=TF.InterpolationMode.NEAREST)
        views_all = TF.to_pil_image(views_all)

        self.wandb_log({label: wandb.Image(views_all)})

    def log_density_field(self):
        density_field = self.render_density_field()
        self.wandb_log({'density': wandb.Image(density_field)})

    def wandb_log(self, d: dict):
        self.trainer.logger.experiment.log(d)
