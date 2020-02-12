import os
import torch
import numpy as np
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

import pickle

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
    SilhouetteShader, PhongShader, PointLights
)


# Set the cuda device 
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Load the obj and ignore the textures and materials.
verts, faces_idx, _ = load_obj("./data/ikea_mug_original.obj")
faces = faces_idx.verts_idx

# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = Textures(verts_rgb=verts_rgb.to(device))

# Create a Meshes object. Here we have only one mesh in the batch.
object_mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)


# Initialize an OpenGL perspective camera.
cameras = OpenGLPerspectiveCameras(
    fov=40.0,
    device=device)

# We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=128, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
    bin_size=0
)
# We can add a point light in front of the object. 
lights = PointLights(device=device, location=((200.0, 200.0, -200.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=PhongShader(device=device, lights=lights)
)


# Load the pickle with the AE data
ae_data = pickle.load(open("./data/ae-codes.pickle", "rb"), encoding='latin1')
index = 11

R = torch.from_numpy(ae_data["Rs"][index].reshape(1,3,3)).to(device)
T = torch.from_numpy(ae_data["ts"][index]).to(device).unsqueeze(0)      

#R = torch.from_numpy(np.array([0.1668620,  0.8331380,  0.5272932,
#                               0.8331380,  0.1668620, -0.5272932,
#                               -0.5272932,  0.5272932, -0.6662760], dtype=np.float32).reshape(1,3,3)).to(device)
T = torch.from_numpy(np.array([0.0,  0.0, 300.0], dtype=np.float32)).to(device).unsqueeze(0)      

# Render the teapot providing the values of R and T. 
image_ref = phong_renderer(meshes_world=object_mesh, R=R, T=T)
image_ref = image_ref.cpu().numpy()

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(ae_data["images"][index])
#plt.grid("off")
plt.title("AE render")

plt.subplot(1, 2, 2)
plt.imshow(image_ref.squeeze())
#plt.grid("off")
plt.title("PyTorch3d render")
plt.show()
