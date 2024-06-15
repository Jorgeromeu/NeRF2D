import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from einops import repeat, rearrange, einsum
from torch import Tensor
from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass

import json
from torchvision.io import read_image
from imageencoder import ImageEncoder
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio



from camera_model_2d import pixel_center_rays, project, transform_p, transform_d

import wandb
from Pixelnerf_model import NeRF
from nerf2d_dataset import get_n_evenly_spaced_views, read_image_folder







def get_exact_pixels(points):
    points = torch.round(points)
    points[points > 49] = 50
    points[points < -50] = 50
    points = points + 50

    points[points < 100] -= 99
    points = abs(points)
    return points



class ProjectCoordinate:
    def __init__(
            self,
            image_resolution=300,
            focal_length=300,
            poses=None
    ):
        self.image_resolution = image_resolution
        self.focal_length = focal_length
        self.c2ws = [c2w for c2w in poses]

    # TODO:
    def project_coordinates_1d(self, xy_coords, directions,  camera_index):
        """
        Projects 2D coordinates onto 1D image axis.

        Parameters:
        image_resolution (int): The length of the 1D image.
        focal_length (int): The focal length of the camera.
        c2w (torch.Tensor): The camera-to-world transformation matrix (3x3).
        xy_coords (torch.Tensor): The 2D coordinates (Nx2).
 b
        Returns:
        torch.Tensor: The projected 1D coordinates (N).
        """
        c2w = self.c2ws[camera_index].to(xy_coords)
        # print('sackascjnkscjnkajnkasc', c2w.device)


        projected_points = project(xy_coords, self.focal_length, c2w.inverse().to(xy_coords))

        points = transform_p(xy_coords, c2w.to(xy_coords))

        directions = transform_d(directions, c2w.to(xy_coords))

        pixels = get_exact_pixels(projected_points)
        pixels = pixels.int()

        return pixels, points, directions


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
            n_gt_poses=100,
            depth_loss_weight=0.5,
            depth_sigma=0.1,
            use_depth_supervision=True,
    ):

        super().__init__()

        self.save_hyperparameters()

        # dummy model


        # metrics
        self.criterion = torch.nn.MSELoss()
        folder = Path('/teamspace/studios/this_studio/NeRF2D/artifacts/cube:v0')
        train_ims, train_poses, train_focal, _ = read_image_folder(folder / 'train')


        #TODO: hardcoded
        train_ims = get_n_evenly_spaced_views(train_ims, 6)

        self.coordinateProjector = ProjectCoordinate(image_resolution=train_ims.shape[2], poses=train_poses, focal_length=train_focal)
        self.feature_maps = []
        image_model = ImageEncoder().to(self.device)
        for train_im in train_ims:
            train_im_d = train_im.squeeze(-1)
            features = image_model.forward(train_im_d)
            features_padded = np.zeros((features.shape[0], features.shape[1] + 1))

            # Copy the values from the original tensor into the new tensor starting from the second row
            features_padded[:, : - 1] = features
            features_padded = torch.tensor(features_padded).float().to(self.device)
            self.feature_maps.append(features_padded.to(self.device))

        self.model = NeRF(
            d_pos_input=2,
            d_dir_input=1,
            n_freqs_position=n_freqs_pos,
            n_freqs_direction=n_freqs_dir,
            n_layers=n_layers,
            d_hidden=d_hidden,
            skip_indices=[n_layers // 2],
            if_hidden = 120,
            nr_images = len(self.feature_maps)
        ).to(self.device)
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
        directions_repeat  = directions.unsqueeze(1).expand(-1, 100, -1)


        image_features, proj_points_list, directions_list = self.sample_features(query_points, directions_repeat)

        return proj_points_list, ts, image_features, query_points, directions_list


    def sample_features(self, query_points, directions):
        points_flat = rearrange(query_points, 'n t d -> (n t) d')
        directions_flat = rearrange(directions, 'n t d -> (n t) d')


        image_features = []

        proj_points_list = []

        directions_list = []

        for i in range(len(self.feature_maps)):

            pixels, proj_points, proj_directions = self.coordinateProjector.project_coordinates_1d(points_flat, directions_flat,  i)

            int_pixels = pixels.squeeze().int().to(self.device)

            proj_points_list.append(proj_points)

            directions_list.append(proj_directions)

            # print( 'nkascjnkscajnk', type(int_pixels))
            # print(int_pixels.device)


            image_features.append(self.feature_maps[i].to(self.device)[:, int_pixels])

        return image_features, proj_points_list, directions_list


    def forward(
            self,
            origins: Tensor,
            directions: Tensor,
    ) -> NeRFOutput:
        """
        Perform volume rendering on model given a set of rays
        :param origins: origins of rays N, 3
        :param directions: directions of rays N, 3
        :return: rendered-colors at rays N, 3
        """
        # query points
        proj_points_list, ts, image_features, query_points, directions_list = self.compute_query_points(origins, directions)

        # TODO query network in 
        
        # print('device0', proj_points_list[0].device)
        # print('device1', directions_list[0].device)

        # print('device2', image_features[0].device)
        # print('model', next(self.model.parameters()).device)
        outputs_flat = self.model(proj_points_list, directions_list, image_features)

        outputs = rearrange(outputs_flat, '(n t) c -> n t c', n=origins.shape[0])

        colors = outputs[:, :, 0:3]
        densities = outputs[:, :, 3]

        # compute volume rendering weights
        weights = volume_rendering_weights(ts, densities)

        # render colors by weighting samples
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

    # def render_density_field(self, res=100, lo=-1.5, hi=1.5):

    #     # densely sample field
    #     xs = torch.linspace(lo, hi, res)
    #     ys = torch.linspace(hi, lo, res)

    #     x, y = torch.meshgrid(xs, ys, indexing='xy')
    #     coords = torch.stack([x, y], dim=-1)
    #     coords_flat = rearrange(coords, 'h w c -> (h w) c')

    #     # random viewdirs, only care about density
    #     dirs = torch.randn(coords_flat.shape[0], 1).to()

    #     # query MLP
    #     with torch.no_grad():
    #         densities = self.model(coords_flat.to(self.device), dirs.to(self.device))[:, -1]

    #     densities = rearrange(densities, '(h w) -> h w', h=res)
    #     densities = (densities - densities.min()) / (densities.max() - densities.min())

    #     return densities


    def training_step(self, batch, batch_idx):
        origins, directions, colors, depth_gt = batch
        # forward pass
        outs = self(origins, directions)
        # colors_pred, weights, ts = self(origins, directions)

        # compute loss
        loss = self.criterion(outs.colors, colors)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def validation_step(self, batch, batch_idx):

        origins, directions, colors_gt, depth_gt = batch

        # forward pass
        outs = self(origins, directions)

        # save rendered views for logging
        # self.rendered_views.append(outs.colors)

        self.val_renders.append(outs.colors)
        self.val_depths.append(self.normalize_depth(outs.depths.unsqueeze(-1)))

        # compute losses
        loss = self.criterion(outs.colors, colors_gt)

        self.log('val_loss', loss)


        # log psnr
        self.log('val_psnr', self.val_psnr(outs.colors, colors_gt))

        return loss

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # Logging Methods

    def on_validation_epoch_end(self) -> None:
        self.log_views('val_renders', self.val_renders)
        self.log_views('val_depths', self.val_depths)
        # self.log_density_field()

    def on_validation_epoch_start(self) -> None:
        # clear validation renders for epoch
        self.val_renders = []
        self.val_depths = []

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

    # def log_density_field(self):
    #     density_field = self.render_density_field()
    #     self.wandb_log({'density': wandb.Image(density_field)})

    def wandb_log(self, d: dict):
        self.trainer.logger.experiment.log(d)



