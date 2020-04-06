# Estimate camera pose demo
# From: https://github.com/facebookresearch/pytorch3d/blob/master/docs/tutorials/camera_position_optimization_with_differentiable_rendering.ipynb

import os
import torch
import numpy as np
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

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



class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        
        # Get the silhouette of the reference RGB image by finding all the non zero values. 
        #image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 0).astype(np.float32))
        image_ref = torch.from_numpy((image_ref).astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the x, y, z position of the camera. 
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([1.5, 140.0, 80.0], dtype=np.float32)).to(meshes.device))

    def forward(self):
        
        # Render the image using the updated camera position. Based on the new position of the 
        # camer we calculate the rotation and translation matrices
        # R = look_at_rotation(self.camera_position[None, :3],
        #                      up=self.camera_position[None, 3:], device=self.device)  # (1, 3, 3)
        # T = -torch.bmm(R.transpose(1, 2), self.camera_position[None, :3, None])[:, :, 0]   # (1, 3)
        R, T = look_at_view_transform(self.camera_position[None, 0],
                                      self.camera_position[None, 1],
                                      self.camera_position[None, 2],
                                      device=self.device)
        
        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        
        # Calculate the silhouette loss
        #loss = torch.sum((image[..., 3] - self.image_ref) ** 2)
        loss = torch.sum((image - self.image_ref) ** 2)
        return loss, image





# Set the cuda device 
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Load the obj and ignore the textures and materials.
#verts, faces_idx, _ = load_obj("../data/cad-models/teapot.obj")
verts, faces_idx, _ = load_obj("../data/ikea-mug/cad/ikea_mug_scaled_reduced_centered.obj")
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

# To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
# edges. Refer to blending.py for more details. 
blend_params = BlendParams(sigma=0.001, gamma=1.0)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 256x256. To form the blended image we use 100 faces for each pixel. Refer to rasterize_meshes.py
# for an explanation of this parameter. 
raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=np.log(1. / 0.001 - 1.) * blend_params.sigma, 
    faces_per_pixel=80, 
    bin_size=0
)

# Create a silhouette mesh renderer by composing a rasterizer and a shader.
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(blend_params=blend_params, device=device, lights=lights)
)


# We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
    bin_size=0
)
# We can add a point light in front of the object. 
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, lights=lights)
)

# Select the viewpoint using spherical angles
#viewpoint = [1.5, 240.0, 10.0] # distance, elevation, azimuth, stuck..
viewpoint = [1.5, 140.0, 10.0] # distance, elevation, azimuth, ok...
#viewpoint = [1.5, 160.0, 10.0] # distance, elevation, azimuth, ok...

# Get the position of the camera based on the spherical angles
R, T = look_at_view_transform(viewpoint[0], viewpoint[1], viewpoint[2], device=device)
print(T.data)
print(R.data)
print("#"*20)

# Render the teapot providing the values of R and T. 
silhouete = silhouette_renderer(meshes_world=teapot_mesh, R=R, T=T)
image = phong_renderer(meshes_world=teapot_mesh, R=R, T=T)
image_ref = silhouette_renderer(meshes_world=teapot_mesh, R=R, T=T)

image = image[0, ..., :3].detach().squeeze().cpu().numpy()
image = img_as_ubyte(image)

fig = plt.figure(figsize=(6,6))
plt.imshow(image[..., :3])
plt.title("ground truth")
plt.grid("off")
plt.axis("off")
fig.tight_layout()
fig.savefig("groundtruth.png", dpi=fig.dpi)
plt.close()


# We will save images periodically and compose them into a GIF.
#filename_output = "./teapot_optimization_demo.gif"
#writer = imageio.get_writer(filename_output, mode='I', duration=0.03)

silhouete = silhouete.cpu().numpy()
image_ref = image_ref.cpu().numpy()

# Initialize a model using the renderer, mesh and reference image
model = Model(meshes=teapot_mesh, renderer=silhouette_renderer, image_ref=image_ref).to(device)

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

plt.figure(figsize=(10, 10))

_, image_init = model()
# plt.subplot(1, 2, 1)
# plt.imshow(image_init.detach().squeeze().cpu().numpy()[..., 3])
# plt.grid("off")
# plt.title("Starting position")

# plt.subplot(1, 2, 2)
# plt.imshow(model.image_ref.cpu().numpy().squeeze())
# plt.grid("off")
# plt.title("Reference silhouette")
# plt.show()

for i in np.arange(1000):
    optimizer.zero_grad()
    loss, _ = model()
    loss.backward()
    optimizer.step()

    print("{0} - loss: {1}".format(i,loss.data))
    #loop.set_description('Optimizing (loss %.4f)' % loss.data)
    
    # Save outputs to create a GIF. 
    if True: #i % 10 == 0:
        # R = look_at_rotation(model.camera_position[None, :3],
        #                      up=model.camera_position[None, 3:], device=model.device)
        # T = -torch.bmm(R.transpose(1, 2), model.camera_position[None, :3, None])[:, :, 0]   # (1, 3)
        R, T = look_at_view_transform(model.camera_position[None, 0],
                                      model.camera_position[None, 1],
                                      model.camera_position[None, 2],
                                      device=device)
        #print(T.data)
        #print(R.data)
        #print("#"*20)
        image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
        image = image[0, ..., :3].detach().squeeze().cpu().numpy()
        image = img_as_ubyte(image)
        #writer.append_data(image)
        
        if(i % 10 == 0):
            fig = plt.figure(figsize=(6,6))
            plt.imshow(image[..., :3])
            plt.title("iter: %d, loss: %0.2f" % (i, loss.data))
            plt.grid("off")
            plt.axis("off")
            fig.tight_layout()
            fig.savefig("iteration{0}.png".format(i), dpi=fig.dpi)
            plt.close()

    if loss.item() < 350:
        break
    
#writer.close()
