import json
import os
from pathlib import Path

import bpy
import numpy as np
import torch

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

def place_cameras(positions, angles, focal_px=100):
    cams = []

    for i in range(N):
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

        # add to collection
        cam_collection.objects.link(cam)

        # remove from original collection
        og_collection.objects.unlink(cam)

        # set focal_length
        set_camera_config(bpy.context.scene, cam, focal_px, 1, res)

        cams.append(cam)

    return cams

def render_from_cameras(cams, folder: Path):
    # get all cameras
    scene = bpy.context.scene

    for camera in cams:
        scene.camera = camera  # Set the active camera
        bpy.context.scene.render.filepath = str(folder / camera.name)  # Set the output path
        bpy.ops.render.render(write_still=True)  # Render the image

if __name__ == '__main__':
    N = 100
    radius = 4
    focal_px = 100
    res = 100
    folder_path = '/home/jorge/repos/NeRF2D/data/views'
    render = True
    collection_name = 'cameras'

    folder = Path(folder_path)
    file_path = folder / 'transforms.json'

    # remove all files in folder
    for file in os.listdir(folder):
        os.remove(folder / file)

    # create collection
    cam_collection = bpy.data.collections.get(collection_name)

    # if collection does not exist create it
    if cam_collection is None:
        cam_collection = bpy.data.collections.new(collection_name)

    # if it does, remove all objects from it
    for obj in cam_collection.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # positions, angles = sample_sphere(radius, N)
    positions, angles = uniform_spaced_circle(radius, N)

    cams = place_cameras(positions, angles, focal_px)

    if render:
        render_from_cameras(cams, folder)

    transforms = []
    for i in range(N):
        pos = positions[i]
        angle = angles[i]
        cam = cams[i]

        # get 2D c2w matrix
        c2w = torch.Tensor([
            [np.cos(angle), -np.sin(angle), pos[0]],
            [np.sin(angle), np.cos(angle), pos[1]],
            [0, 0, 1]
        ])

        transforms.append(c2w)

    transforms_dict = {
        'focal': focal_px,
        'frames': [{'transform_matrix': transforms[i].tolist()} for i in range(N)]
    }

    with open(file_path, 'w') as json_file:
        json.dump(transforms_dict, json_file, indent=2)
