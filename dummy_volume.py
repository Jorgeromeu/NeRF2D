import torch

class DummyVolume:

    def __init__(self, density=100):
        self._volume = torch.zeros(100, 100, 4)
        self._volume[25:75, 25:75, 0] = 1

        # density
        self._volume[25:75, 25:75, 3] = density

    def x_to_px(self, x: torch.Tensor):
        """
        Map an x-coord to pixel space with NN interpolation
        :param x: 2D coordinate
        :return: pixel coordinate
        """
        return (x * 50 + 50).round().long()

    def __call__(self, x: torch.Tensor):
        """
        :param x: 2D coordinate
        :return: color
        """
        px = self.x_to_px(x)
        px = torch.clamp(px, 0, 99)

        # index volume with coords in px
        return self._volume[px[:, 0], px[:, 1]]
