# Estimate camera pose demo
# From: https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/camera_position_optimization_with_differentiable_rendering.ipynb

import os
import torch
import numpy as np
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# io utils
from pytorch3d.io import load_obj

# datastructures
from pytorch3d.structures import Meshes, Textures

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    SoftSilhouetteShader, SoftPhongShader, HardPhongShader, PointLights
)


def prepareRenderer(sigma_, gamma_, blur_radius_, faces_per_pixel_, device):
    blend_params = BlendParams(sigma=sigma_, gamma=gamma_)

    raster_settings = RasterizationSettings(
        image_size=256, 
        blur_radius=np.log(1. / blur_radius_ - 1.) * blend_params.sigma, 
        faces_per_pixel=faces_per_pixel_, 
        bin_size=0
    )

    # renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(
    #         cameras=cameras, 
    #         raster_settings=raster_settings
    #     ),
    #     shader=SoftSilhouetteShader(blend_params=blend_params)
    # )
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(blend_params=blend_params, device=device, lights=lights)
    )
    return renderer
    

# Set the cuda device 
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Load the obj and ignore the textures and materials.
verts, faces_idx, _ = load_obj("../data/cad-models/teapot.obj")
#verts, faces_idx, _ = load_obj("../data/ikea-mug/cad/ikea_mug_scaled_reduced_centered.obj")
faces = faces_idx.verts_idx

# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = Textures(verts_rgb=verts_rgb.to(device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
teapot_mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)


# Initialize an OpenGL perspective camera.
cameras = OpenGLPerspectiveCameras(device=device)

# Select the viewpoint using spherical angles
viewpoint = [2.5, 140.0, 10.0] # distance, elevation, azimuth, ok...

# Get the position of the camera based on the spherical angles
R, T = look_at_view_transform(viewpoint[0], viewpoint[1], viewpoint[2], device=device)


sigma = [0.001] #[1e-3, 1e-4 ,1e-5]
gamma = [1.0] #[1e-0, 1e-1 ,1e-2, 1e-3, 1e-4 ,1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10] #[0.001] #[1e-3, 1e-4 ,1e-5]
faces = [100] #[1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
radius = [1e-0, 1e-1 ,1e-2, 1e-3] #, 1e-4 ,1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

for s in sigma:
    for g in gamma:
        for f in faces:
            for r in radius:
                print("S: {0}, G: {1}, F: {2}, R: {3}".format(s,g,f,r))
                renderer = prepareRenderer(s, g, r, f, device)
                image = renderer(meshes_world=teapot_mesh, R=R, T=T)
                image = image.detach().squeeze().cpu().numpy()[..., :3]
                #image = image[..., :3]
                print(image.shape)
                print(np.mean(image))

                fig = plt.figure(figsize=(6,6))
                plt.imshow(image)
                plt.title("S: {0}, G: {1}, F: {2}, R: {3}".format(s,g,f,r))
                plt.grid("off")
                plt.axis("off")
                fig.tight_layout()
                fig.savefig("s{0}g{1}f{2}r{3}.png".format(s,g,f,r), dpi=fig.dpi)
                plt.close()
