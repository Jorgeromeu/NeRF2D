from pathlib import Path

import imageio
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.io import read_image, ImageReadMode

from nerf2d import NeRF2D_LightningModule
from nerf2d_dataset import NeRF2D_Datamodule
from transform2d import Transform2D
from wandb_utils import RunWrapper, RunDataManager, first_used_artifact_of_type, parse_run_file_name

def make_gif_from_run(manager, run: RunWrapper, label, out_path: Path, duration=0.1):
    files = manager.download_run_files_by_label(run, label)
    file_names = sorted(files, key=lambda x: parse_run_file_name(x)['step'])

    pil_ims = []
    for path in file_names:
        t = read_image(path, ImageReadMode.GRAY)
        pil_im = TF.to_pil_image(t)
        pil_ims.append(pil_im)

    imageio.mimsave(out_path, pil_ims, duration=0.1)

def poses_to_quiver(poses: Tensor):
    c2ws = [Transform2D.from_matrix(c2w) for c2w in poses]
    positions = torch.stack([c2w.translation() for c2w in c2ws])
    angles = torch.stack([c2w.rotation() for c2w in c2ws])
    dx = torch.cos(angles)
    dy = torch.sin(angles)

    xs = positions[:, 0]
    ys = positions[:, 1]

    return xs, ys, dx, dy

    # pass
    plt.quiver(positions[:, 0], positions[:, 1], d[:, 0], d[:, 1], color='tab:blue', label='test')

def vis_datamodule(dm: NeRF2D_Datamodule, H=100, W=100):
    nerf = NeRF2D_LightningModule(t_near=1.5, t_far=7)

    ims = dm.test_ims
    depths = dm.test_depths

    ims_im = rearrange(ims, 'n c h 1 -> c h n')
    ims_im = TF.resize(ims_im, (H, W), interpolation=TF.InterpolationMode.NEAREST_EXACT)

    depths_im = rearrange(depths, 'n c h 1 -> c h n')
    depths_im = nerf.normalize_depth(depths_im)
    depths_im = TF.resize(depths_im, (H, W), interpolation=TF.InterpolationMode.NEAREST_EXACT)

    fig, axs = plt.subplots(1, 3, figsize=(5, 10))
    axs[1].imshow(ims_im.permute(1, 2, 0))
    axs[2].imshow(depths_im.permute(1, 2, 0), cmap='gray')
    for ax in axs:
        ax.axis('off')

    x, y, dx, dy = poses_to_quiver(dm.test_poses)
    axs[0].quiver(x, y, dx, dy, color='tab:blue', label='test')
    axs[0].axis('equal')



    fig.tight_layout()

def compare_runs(
        runs: list[RunWrapper],
        manager: RunDataManager,
        with_density=True,
        with_depth=True,
        with_gt=True,
        with_renders=True,
        depth_cmap='gray',
        show_psnr=True,
        fig_scale=4,
        title_fun=lambda run: None
):
    n_rows = (1 if with_renders else 0) + (1 if with_depth else 0) + (1 if with_density else 0)
    n_cols = len(runs) + (1 if with_gt else 0)

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_scale * n_cols, fig_scale * n_rows))
    axs = axs.reshape(n_rows, n_cols)

    for ax in axs.flatten():
        ax.axis('off')

    if with_gt:
        # read ground_truth render
        axs[0, 0].set_title('GT')

        row = -1
        if with_renders:
            row += 1
            gt_render = read_image(manager.download_last_run_file_by_label(runs[0], 'val_renders_gt')[0])
            axs[row, 0].imshow(gt_render.permute(1, 2, 0))

        if with_depth:
            row += 1
            gt_depth = read_image(manager.download_last_run_file_by_label(runs[0], 'val_depth_gt')[0],
                                  ImageReadMode.GRAY)
            axs[row, 0].imshow(gt_depth[0], cmap=depth_cmap)

    for i, run in enumerate(runs):

        data_art = first_used_artifact_of_type(run.run, 'dataset')

        gt_offset = 1 if with_gt else 0
        row = -1

        # set title
        title = title_fun(run)
        axs[0, i + gt_offset].set_title(title)

        if with_renders:
            row += 1
            render = read_image(manager.download_last_run_file_by_label(run, 'val_renders')[-1])
            ax = axs[row, i + gt_offset]
            ax.imshow(render.permute(1, 2, 0))

            if show_psnr:
                psnr = run.run.summary["val_psnr"]
                H = render[0].shape[0]
                W = render[0].shape[1]
                ax.text(W * 0.05, H * 0.95, f'PSNR: {psnr:.2f}', color='red', fontsize=10)

        if with_depth:
            row += 1
            depth = read_image(manager.download_last_run_file_by_label(run, 'val_depths')[0])
            ax = axs[row, i + gt_offset]
            ax.imshow(depth[0], cmap=depth_cmap)

        if with_density:
            row += 1
            density = read_image(manager.download_last_run_file_by_label(run, 'density')[0])
            ax = axs[row, i + gt_offset]
            ax.imshow(density[0], cmap=depth_cmap)

    plt.tight_layout()
