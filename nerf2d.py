import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from einops import repeat, rearrange, einsum
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio

import wandb
from camera_model_2d import pixel_center_rays
from nerf_model import NeRF

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
    delta = torch.cat([delta, Tensor([delta.max()]).to(ts)], dim=0)

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
        origins, directions = pixel_center_rays(height, focal_length_px, c2w)
        origins = origins.to(self.device)
        directions = directions.to(self.device)

        colors, _, _ = self.forward(origins, directions)

        # reshape as image
        colors_im = rearrange(colors, 'h c -> c h 1')
        return colors_im

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

    def compute_losses(self, colors_pred, colors_gt, ts, weights, gt_depth):

        """
        Compute losses after a forward pass
        """

        color_loss = self.color_loss(colors_pred, colors_gt)

        if not self.hparams.use_depth_supervision:
            return {'loss': color_loss, 'color_loss': color_loss}
        else:

            depth_loss = self.depth_loss(gt_depth, ts, weights, self.hparams.depth_sigma) / 2000
            total_loss = (1 - self.hparams.depth_loss_weight) * color_loss + self.hparams.depth_loss_weight * depth_loss

            return {
                'loss': total_loss,
                'color_loss': color_loss,
                'depth_loss': depth_loss,
            }

    def training_step(self, batch, batch_idx):

        origins, directions, colors_gt, depth_gt = batch

        # forward pass
        colors_pred, weights, ts = self(origins, directions)

        # compute losses
        losses = self.compute_losses(colors_pred, colors_gt, ts, weights, depth_gt)

        for k, v in losses.items():
            self.log(f'train_{k}', v)

        return losses['loss']

    def validation_step(self, batch, batch_idx):

        origins, directions, colors_gt, depth_gt = batch

        # forward pass
        colors_pred, weights, ts = self(origins, directions)

        # save rendered views for logging
        self.rendered_views.append(colors_pred)

        # compute losses
        losses = self.compute_losses(colors_pred, colors_gt, ts, weights, depth_gt)

        for k, v in losses.items():
            self.log(f'val_{k}', v)

        # log psnr
        self.log('val_psnr', self.val_psnr(colors_pred, colors_gt))

        return losses['loss']

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
