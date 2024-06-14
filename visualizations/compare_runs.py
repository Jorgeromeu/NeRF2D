from matplotlib import pyplot as plt
from torchvision.io import read_image, ImageReadMode

from wandb_utils import RunWrapper, RunDataManager, first_used_artifact_of_type

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
            gt_depth = read_image(manager.download_last_run_file_by_label(runs[0], 'val_depth_gt')[0], ImageReadMode.GRAY)
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
