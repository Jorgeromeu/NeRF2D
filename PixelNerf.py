import pytorch_lightning as pl
import torch
import torchvision.transforms.functional as TF
from einops import repeat, rearrange, einsum
from torch import Tensor
from pathlib import Path
import json
from torchvision.io import read_image
from imageencoder import ImageEncoder
import numpy as np


from camera_model_2d import pixel_center_rays, project, transform_p, transform_d

import wandb
from Pixelnerf_model import NeRF

def read_image_folder(path: Path):
    with open(path / 'transforms.json') as f:
        transforms_json = json.load(f)

    # read poses
    poses = torch.stack([torch.Tensor(frame_data['transform_matrix']) for frame_data in transforms_json['frames']])

    #TODO: remove
    indices = np.arange(0, 50, 10)
    poses = poses[indices]
    # read images
    ims = torch.stack([read_image(str(path / f'cam-{10 * i}.png')) for i in range(len(poses))])
    # drop alpha channel and nromalize to floats
    ims = ims[:, :3, :, :] / 255

    focal = transforms_json['focal']

    return ims, poses, focal


def get_exact_pixels(points):
    points = torch.round(points)
    points[points > 249] = 250
    points[points < -250] = 250
    points = points + 250

    points[points < 500] -= 499
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
        c2w = self.c2ws[camera_index]


        projected_points = project(xy_coords, self.focal_length, c2w.inverse())

        points = transform_p(xy_coords, c2w.inverse())

        directions = transform_d(directions, c2w.inverse())

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
            n_gt_poses=100
    ):

        super().__init__()

        self.save_hyperparameters()

        # dummy model


        # metrics
        self.criterion = torch.nn.MSELoss()
        folder = Path('./data/cube/')
        train_ims, train_poses, train_focal = read_image_folder(folder / 'train')
        self.coordinateProjector = ProjectCoordinate(image_resolution=train_ims.shape[2], poses=train_poses, focal_length=train_focal)
        self.feature_maps = []
        image_model = ImageEncoder()
        for train_im in train_ims:
            train_im_d = train_im.squeeze(-1)
            features = image_model.forward(train_im_d)
            features_padded = np.zeros((features.shape[0], features.shape[1] + 1))

            # Copy the values from the original tensor into the new tensor starting from the second row
            features_padded[:, : - 1] = features
            features_padded = torch.tensor(features_padded).float().cuda()
            self.feature_maps.append(features_padded)

        self.model = NeRF(
            d_pos_input=2,
            d_dir_input=1,
            n_freqs_position=n_freqs_pos,
            n_freqs_direction=n_freqs_dir,
            n_layers=n_layers,
            d_hidden=d_hidden,
            skip_indices=[n_layers // 2],
            if_hidden = 120,
            nr_images = 5
        )




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

            int_pixels = pixels.squeeze().int().cuda()

            proj_points_list.append(proj_points)

            directions_list.append(proj_directions)


            image_features.append(self.feature_maps[i][:, int_pixels.cuda()].cuda())

        return image_features, proj_points_list, directions_list



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
    ) -> Tensor:
        """
        Perform volume rendering on model given a set of rays
        :param origins: origins of rays N, 3
        :param directions: directions of rays N, 3
        :return: rendered-colors at rays N, 3
        """
        # query points
        proj_points_list, ts, image_features, query_points, directions_list = self.compute_query_points(origins, directions)

        # TODO query network in chunks
        outputs_flat = self.model(proj_points_list, directions_list, image_features)

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
        origins, directions = pixel_center_rays(height, focal_length_px, c2w)
        origins = origins.to(self.device)
        directions = directions.to(self.device)

        colors = self.forward(origins, directions)

        # reshape as image
        colors_im = rearrange(colors, 'h c -> c h 1')
        return colors_im

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

        self.rendered_views.append(colors_pred)

        # compute loss
        loss = self.criterion(colors_pred, colors)
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
            _, _, colors = batch
            gt_views.append(colors)

        gt_all = torch.stack(gt_views, dim=1).detach().cpu()

        gt_all = rearrange(gt_all, 'h w c -> c h w')
        gt_all = TF.resize(gt_all, (256, 256), interpolation=TF.InterpolationMode.NEAREST)
        gt_all = TF.to_pil_image(gt_all)

        self.trainer.logger.experiment.log({
            'renders_gt': wandb.Image(gt_all)
        })




