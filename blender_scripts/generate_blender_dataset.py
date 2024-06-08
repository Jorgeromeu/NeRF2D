import json
import os
from pathlib import Path

import bpy
import numpy as np

def uniform_spaced_circle(radius, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points + 1)[:num_points]
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    pos = np.stack([x, y], axis=1)
    dirs = -pos / np.linalg.norm(pos)

    return pos, angles + np.pi

def sample_circle(radius, num_points):
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)

    pos = np.stack([x, y], axis=1)
    dirs = -pos / np.linalg.norm(pos)

    angles_cameras = angles + np.pi

    return pos, angles_cameras

def set_camera_config(scene, camera, focal_px, res_x, res_y):
    # set resolution
    scene.render.resolution_x = res_x * 4
    scene.render.resolution_y = res_y * 4
    scene.render.resolution_percentage = 25

    # set aspect ratio
    scene.render.pixel_aspect_x = 1.0
    scene.render.pixel_aspect_y = res_x / res_y

    # set intrinsics
    sensor_w = max(res_x, res_y)
    camera.data.sensor_height = sensor_w
    camera.data.sensor_width = sensor_w
    camera.data.lens = focal_px

class GenerateDatasetOp(bpy.types.Operator):
    bl_idname = "wm.generate_dataset"
    bl_label = "Generate Dataset"
    bl_options = {'REGISTER', 'UNDO'}

    n_cams: bpy.props.IntProperty(name='Number of cameras', default=10, min=1, max=1000)
    radius: bpy.props.FloatProperty(name='Radius', default=4, min=0.1, max=100)
    focal: bpy.props.IntProperty(name='Focal length (px)', default=100, min=1, max=1000)
    res: bpy.props.IntProperty(name='Resolution', default=100, min=1, max=1000)

    noise_rotation: bpy.props.IntProperty(name='Rotation σ (deg)', min=0, max=100)
    noise_radius: bpy.props.FloatProperty(name='Radius σ', min=0, max=50)
    noise_position: bpy.props.FloatProperty(name='Position σ', min=0, max=0.5)

    cam_collection = None

    def uniform_spaced_circle(self):
        # generate N uniformly spaced angles
        angles_even = np.linspace(0, 2 * np.pi, self.n_cams + 1)[:self.n_cams]
        angles_normal = np.random.normal(loc=angles_even, scale=self.noise_position)

        # generate radii with normal distribution
        radii = np.random.normal(loc=self.radius, scale=self.noise_radius, size=self.n_cams)

        # compute x-coords from angles and radii
        x = radii * np.cos(angles_normal)
        y = radii * np.sin(angles_normal)
        pos = np.stack([x, y], axis=1)

        # camera angles
        camera_angles = angles_normal + np.pi

        # add noise to camera angles
        camera_angles = np.random.normal(loc=camera_angles, scale=np.radians(self.noise_rotation), size=self.n_cams)

        return pos, camera_angles

    def place_cameras(self, positions, angles):
        cams = []

        for i in range(len(positions)):
            angle = angles[i]

            # create the camera
            bpy.ops.object.camera_add(
                location=(positions[i, 0], positions[i, 1], 0),
                rotation=(0, np.radians(-90), angle),
            )

            # rename
            cam = bpy.context.object
            cam.name = f'cam-{i}'

            og_collection = cam.users_collection[0]

            # set focal_length
            set_camera_config(bpy.context.scene, cam, self.focal, 1, self.res)

            cams.append(cam)

        return cams

    def execute(self, ctx):

        # if no collection selected, create a new one
        if ctx.collection.name == 'Scene Collection':
            self.report({'ERROR'}, 'Select a collection for cameras')
            return {'CANCELLED'}

        # set cam collection to current one
        self.cam_collection = ctx.collection

        # delete everything in it
        for obj in self.cam_collection.objects:
            bpy.data.objects.remove(obj)

        # positions, angles = uniform_spaced_circle(self.radius, self.n_cams)
        positions, angles = self.uniform_spaced_circle()

        cams = self.place_cameras(positions, angles)

        self.cam_collection['positions'] = positions
        self.cam_collection['angles'] = angles
        self.cam_collection['focal'] = self.focal

        return {'FINISHED'}

class RenderDatasetOp(bpy.types.Operator):
    bl_idname = "wm.render_dataset"
    bl_label = "Render Dataset"
    bl_options = {'REGISTER', 'UNDO'}

    def read_selected_collection(self, ctx):

        """
        Read camera collection
        :param ctx:
        :return: return cameras, positions, angles and focal length
        """

        cam_collection = ctx.collection

        cams = []
        for obj in cam_collection.objects:
            if obj.type == 'CAMERA':
                cams.append(obj)

        try:
            positions = np.array(cam_collection['positions'])
            positions = positions.reshape(-1, 2)
            angles = np.array(cam_collection['angles'])
            focal = cam_collection['focal']

            return cams, positions, angles, focal

        except KeyError as e:

            return None

    def render_from_cameras(self, cams: list, views_folder: Path):

        # get all cameras
        scene = bpy.context.scene

        for i, camera in enumerate(cams):
            scene.camera = camera  # Set the active camera
            bpy.context.scene.render.filepath = str(views_folder / f'cam-{i}.png')  # Set the output path
            bpy.ops.render.render(write_still=True)  # Render the image

            try:
                os.rename(views_folder / '0000.npz', views_folder / f'cam-{i}.npz')
            except FileNotFoundError:
                pass

    def save_transforms(self, positions, angles, focal, views_folder: Path):

        transforms = []
        for i in range(len(positions)):
            pos = positions[i]
            angle = angles[i]

            # get 2D c2w matrix
            c2w = np.array([
                [np.cos(angle), -np.sin(angle), pos[0]],
                [np.sin(angle), np.cos(angle), pos[1]],
                [0, 0, 1]
            ])

            transforms.append(c2w)

        transforms_dict = {
            'focal': focal,
            'frames': [{'transform_matrix': transforms[i].tolist()} for i in range(len(angles))]
        }

        with open(views_folder / 'transforms.json', 'w') as json_file:
            json.dump(transforms_dict, json_file, indent=2)

    def invoke(self, ctx, event):
        return self.execute(ctx)

    def execute(self, ctx):

        try:
            cams, positions, angles, focal = self.read_selected_collection(ctx)
        except TypeError:
            self.report({'ERROR'}, 'Select a Camera Collection')
            return {'CANCELLED'}

        # create folder
        views_folder = Path(ctx.collection.name)
        views_folder.mkdir(exist_ok=True)

        # remove all files in folder
        for file in os.listdir(views_folder):
            os.remove(views_folder / file)

        # save dataset
        self.render_from_cameras(cams, views_folder)
        self.save_transforms(positions, angles, focal, views_folder)

        return {'FINISHED'}

if __name__ == '__main__':

    for op in [GenerateDatasetOp, RenderDatasetOp]:
        def menu_func(self, context):
            self.layout.operator(op.bl_idname, text="Hello World Operator")

        bpy.utils.register_class(op)
        bpy.types.VIEW3D_MT_view.append(menu_func)
