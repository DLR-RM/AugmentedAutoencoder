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
from pytorch3d.structures import Meshes, Textures, list_to_padded

# 3D transformations functions
from pytorch3d.transforms import Rotate, Translate

# rendering components
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    HardPhongShader, PointLights, DirectionalLights
)

# Parameters
batch_size = 1
dist = 20
loops = 10000
obj_path = "../data/ikea-mug/cad/ikea_mug_scaled_reduced_centered.obj"
visualize = False

# Set the cuda device 
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Load the obj and ignore the textures and materials.
verts, faces_idx, _ = load_obj(obj_path)
faces = faces_idx.verts_idx

verts_rgb = torch.ones_like(verts)
batch_verts_rgb = list_to_padded([verts_rgb for k in np.arange(batch_size)])  # B, Vmax, 3
        
batch_textures = Textures(verts_rgb=batch_verts_rgb.to(device))
batch_mesh = Meshes(
    verts=[verts.to(device) for k in np.arange(batch_size)],
    faces=[faces.to(device) for k in np.arange(batch_size)],
    textures=batch_textures
)

# Initialize an OpenGL perspective camera.
cameras = OpenGLPerspectiveCameras(
    fov=5.0,
    device=device)

# We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=320, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
    bin_size=0
)
# We can add a point light in front of the object. 
#lights = PointLights(device=device, location=((-1.0, -1.0, -2.0),))
#"ambient_color", "diffuse_color", "specular_color"
# 'ambient':0.4,'diffuse':0.8, 'specular':0.3
lights = DirectionalLights(device=device,
                     ambient_color=[[0.25, 0.25, 0.25]],
                     diffuse_color=[[0.6, 0.6, 0.6]],
                     specular_color=[[0.15, 0.15, 0.15]],
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
elevs = []
azims = []
images = []
lights = []

start = time.time()
for i in np.arange(loops):

    curr_Rs = []
    curr_ts = []
    cam_pose = None
    for k in np.arange(batch_size):
        pose = np.random.uniform(low=0.0, high=360.0, size=2)
        R, t = look_at_view_transform(dist, elev=pose[0], azim=pose[1])
        #R, t = look_at_view_transform(dist, elev=45, azim=0)
        elevs.append(pose[0])
        azims.append(pose[1])
        
        curr_Rs.append(R.squeeze())
        curr_ts.append(t.squeeze())
        np_t = t.cpu().numpy().squeeze()
        np_R = R.cpu().numpy().squeeze()
        cam_pose = np.dot(np_R,np_t)
        cam_pose = -cam_pose
        
    batch_R = torch.tensor(np.stack(curr_Rs), device=device, dtype=torch.float32)
    batch_T = torch.tensor(np.stack(curr_ts), device=device, dtype=torch.float32)

    random_light = [random.uniform(-1.0,1.0) for i in np.arange(3)]
    

    #phong_renderer.shader.lights.location = torch.tensor(cam_pose, device=device, dtype=torch.float32).unsqueeze(0)
    phong_renderer.shader.lights.direction = [random_light]
    lights.append(random_light)
    
    image_renders = phong_renderer(meshes_world=batch_mesh, R=batch_R, T=batch_T)

    for k in np.arange(batch_size):
        image_ref = image_renders[k].cpu().numpy().squeeze()
        image_ref = image_ref[:,:,:3]

        Rs.append(curr_Rs[k].cpu().numpy().squeeze())
        ts.append(curr_ts[k].cpu().numpy().squeeze())

        ys, xs = np.nonzero(image_ref[:,:,0] > 0)
        obj_bb = calc_2d_bbox(xs,ys,[640,640])
        cropped = extract_square_patch(image_ref, obj_bb)   
        images.append(cropped)
        print("Loop: {0} Batch: {1}".format(i,k))

        if(visualize):
            plt.figure(figsize=(2, 2))
            plt.imshow(image_ref)
            plt.figure(figsize=(2, 2))
            plt.imshow(cropped)    
            plt.show()
print("Elapsed: {0}".format(time.time()-start))

data={"images":images,
      "Rs":Rs,
      "ts":ts,
      "elevs":elevs,
      "azims":azims,
      "dist":dist,
      "light_dir":lights}
pickle.dump(data, open("./training-images-part3.p", "wb"), protocol=2)
