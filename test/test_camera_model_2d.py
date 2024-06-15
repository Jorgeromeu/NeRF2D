# from unittest import TestCase
#
# import rerun as rr
# import torch
#
# import visualizations.rerun_util as ru
# from camera_model_2d import camera_rays, project, projection_matrix, unembed_homog, embed_homog, _project_gt
# from transform2d import Transform2D
#
# class TestCameraModel(TestCase):
#     show_vis = False
#
#     def test_projection_matrix_matches_gt(self):
#         # random points
#         p = torch.randn(10, 2)
#         f = 3
#
#         P = projection_matrix(f)
#
#         # project with matrix
#         projected = unembed_homog(embed_homog(p) @ P.T)
#
#         # project with function
#         projected_gt = _project_gt(p, f)
#
#         self.assertTrue(torch.allclose(projected, projected_gt, atol=1e-5))
#
#     def test_ray_project_consistent(self):
#         """
#         Ensure the get_rays and project functions are consistent
#         """
#
#         rr.init('project-ray-consistency', spawn=True, default_enabled=self.show_vis)
#
#         # random translation and rotation
#         t = torch.randn(2)
#         r = torch.randn(1).item()
#
#         focal_length = 3
#
#         # get c2w and w2c
#         c2w = Transform2D.from_translation_and_rotation(t, r)
#         # c2w = Transform2D.identity()
#         w2c = c2w.inverse()
#
#         # log camera
#         rr.log('cam', ru.pinhole_2D(focal_length, 3))
#         rr.log('cam', ru.embed_Transform2D(c2w))
#
#         # random pixel coordinate
#         pixel_coord = torch.rand(1)
#         pixel_coord = torch.Tensor([0.1])
#
#         # get ray through pixel coord
#         o, d = camera_rays(pixel_coord, focal_length, c2w.as_matrix())
#
#         rr.log('ray', ru.embed_rays(o, d))
#
#         # get point at random depth along ray
#         t = torch.rand(1).item()
#         p = o + d * t
#
#         rr.log('p', rr.Points3D(ru.embed_Points2D(p)))
#
#         # project point back to image plane
#         projected = project(p, focal_length, w2c.as_matrix())[0]
#
#         # assert re-projection is the same as original
#         self.assertTrue(torch.allclose(projected, pixel_coord, atol=1e-5))
