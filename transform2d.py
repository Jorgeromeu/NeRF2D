import numpy as np
import torch

class Transform2D:
    """
    Utility class for working with 2D rigid transformations
    """

    def __int__(self):
        self.mx = torch.eye(4)

    def rotation_matrix(self):
        return self.mx[:2, :2]

    def translation(self):
        return self.mx[:2, 2]

    def rotation(self):
        return np.arctan2(self.mx[1, 0], self.mx[0, 0])

    @classmethod
    def from_matrix(cls, matrix: torch.Tensor):
        t = Transform2D()
        t.mx = matrix
        return t

    @classmethod
    def from_translation_and_rotation(cls, translation: torch.Tensor, rotation: float):
        t = Transform2D()
        t.mx = torch.Tensor([
            [np.cos(rotation), -np.sin(rotation), translation[0]],
            [np.sin(rotation), np.cos(rotation), translation[1]],
            [0, 0, 1]
        ])
        return t

    def as_matrix(self):
        return self.mx

    def __str__(self):
        return str(self.mx)

    def __repr__(self):
        return repr(self.mx)
