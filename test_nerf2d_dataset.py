from pathlib import Path
from unittest import TestCase

from torchvision.transforms.v2 import Compose, ToImage

from nerf2d_dataset import read_image_folder, NeRFDataset2D

class TestNeRFDataset2D(TestCase):

    def test_reading(self):
        ims, poses, focal = read_image_folder(Path('data/views'))
        print(ims.shape)

        dataset = NeRFDataset2D(ims, poses, focal)

        print(dataset.image_resolution)

        # print(dataset.ims[0].shape)
        # print(dataset.ims[0][0])
