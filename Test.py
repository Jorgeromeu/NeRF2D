from pathlib import Path

import matplotlib.pyplot as plt

from PixelNerf import ProjectCoordinate, read_image_folder, get_rays2d, pixel_centers
import numpy as np
import torch

from transform2d import Transform2D

if __name__ == '__main__':
    # folder = Path('./data/cube/')
    # train_ims, train_poses, train_focal = read_image_folder(folder / 'train')
    # coordinateProjector = ProjectCoordinate(image_resolution=train_ims.shape[2], poses=train_poses,
    #                                             focal_length=train_focal)
    #
    # index = 0
    # c2w = coordinateProjector.c2ws[index]
    # c2w_transform = Transform2D.from_matrix(c2w)

    f = 3
    h = 10
    t = np.array([0,0])
    r = 0

    my_c2w = Transform2D.from_translation_and_rotation(t, r).as_matrix()
    # print(my_c2w)

    o, d = get_rays2d(h, f, my_c2w)
    centers = pixel_centers(-h/2, h/2, 3)

    idx = 8

    p = o[idx] + 1 * d[idx]

    coordinateProjector = ProjectCoordinate(image_resolution=h, poses=[my_c2w],
                                                focal_length=f)

    p = torch.tensor([[p[0], p[1]]])
    ans = coordinateProjector.project_coordinates_1d(p, 0)

    true_ans = o[idx] + (f / d[idx, 0]) * d[idx]


    print('ans', true_ans)
    # print(ans)


    #
    # print(c2w_transform.transla,tion())
    # print(c2w_transform.rotation())
    #
    # p = torch.Tensor([[2, 2]])
    #
    # u = coordinateProjector.project_coordinates_1d(p, index)
    # print(u)
    # print(p)

    #
    # plt.plot(x=p[: 0], y=p[:, 1])
    # plt.show()