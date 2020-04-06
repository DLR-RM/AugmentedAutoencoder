import os
import torch
import numpy as np
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

import time
import pickle
import random
from utils import *

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
    HardPhongShader, PointLights, DirectionalLights
)


# Set the cuda device 
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Load the obj and ignore the textures and materials.
#verts, faces_idx, _ = load_obj("../data/t-less-obj19/cad/obj_19_scaled.obj")
verts, faces_idx, _ = load_obj("../data/ikea-mug/cad/ikea_mug_scaled_reduced_centered.obj")
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
    fov=5.0,
    device=device)

# We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=640, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
    bin_size=0
)
# We can add a point light in front of the object. 
#lights = PointLights(device=device, location=((-1.0, -1.0, -2.0),))
#"ambient_color", "diffuse_color", "specular_color"
# 'ambient':0.4,'diffuse':0.8, 'specular':0.3
lights = DirectionalLights(device=device,
                           ambient_color=[[0.4, 0.4, 0.4]],
                           diffuse_color=[[0.8, 0.8, 0.8]],
                           specular_color=[[0.3, 0.3, 0.3]],
                           direction=[[-1.0, -1.0, 1.0]])
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, lights=lights)
)

Rs = []
ts = []
images = []
quats = []

start = time.time()
for i in np.arange(5000):
    quat = random_quaternion(None)       
    R = torch.from_numpy(quaternion_matrix(quat)[:3,:3]).to(device).unsqueeze(0)
    T = torch.from_numpy(np.array([0.0,  0.0, 20.0], dtype=np.float32)).to(device).unsqueeze(0)

    random_light = [random.uniform(-1.0,1.0) for i in np.arange(3)]
    phong_renderer.shader.lights.direction = [random_light]
    
    # Render the teapot providing the values of R and T. 
    image_ref = phong_renderer(meshes_world=object_mesh, R=R, T=T)
    image_ref = image_ref.cpu().numpy().squeeze()
    image_ref = image_ref[:,:,:3]

    Rs.append(R.cpu().numpy().squeeze())
    ts.append(T.cpu().numpy().squeeze())
    quats.append(quat)
    #print(quats[-1])

    ys, xs = np.nonzero(image_ref[:,:,0] > 0)
    obj_bb = calc_2d_bbox(xs,ys,[640,640])
    cropped = extract_square_patch(image_ref, obj_bb)   
    images.append(cropped)
    print("Sample: {0}".format(i))

    #plt.figure(figsize=(2, 2))
    #plt.imshow(cropped)
    #plt.show()
print("Elapsed: {0}".format(time.time()-start))
data={"images":images,"Rs":Rs,"ts":ts, "quats":quats}
pickle.dump(data, open("./training-images.p", "wb"), protocol=0)
