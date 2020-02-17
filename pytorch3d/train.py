import os
import torch
import numpy as np
import pickle
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
    SilhouetteShader, PhongShader, PointLights, DirectionalLights
)

from utils import *
from model import Model

def Loss(gt_img, predicted_pose, renderer, object_mesh):
    #print(predicted_pose.shape)
    R = quat2mat(predicted_pose)
    T = torch.from_numpy(np.array([0.0,  0.0, 3.5], dtype=np.float32)).to(device).unsqueeze(0)

    #print(predicted_pose)
    
    # Render the teapot providing the values of R and T. 
    image_ref = renderer(meshes_world=object_mesh, R=R, T=T)

    #print(image_ref.shape)
    #print(gt_img.shape)
       
    #diff = torch.abs(gt_img - image_ref)
    #topk, indices = torch.topk(diff.flatten(), 10000)
    #loss = torch.mean(topk)
    #loss = torch.mean((gt_img - image_ref) ** 2)
    error = ((gt_img - image_ref) ** 2)
    topk, indices = torch.topk(error.flatten(), 16000)
    loss = torch.mean(topk)
    return loss, image_ref

# Set the cuda device 
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Load the obj and ignore the textures and materials.
verts, faces_idx, _ = load_obj("./data/ikea_mug_scaled_reduced.obj")
faces = faces_idx.verts_idx

# Initialize each vertex to be white in color.
verts_rgb = torch.ones_like(verts)[None]  # (1, V, 3)
textures = Textures(verts_rgb=verts_rgb.to(device))

# Create a Meshes object for the teapot. Here we have only one mesh in the batch.
object_mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)


# Initialize an OpenGL perspective camera.
cameras = OpenGLPerspectiveCameras(device=device)

# # We will also create a phong renderer. This is simpler and only needs to render one face per pixel.
# raster_settings = RasterizationSettings(
#     image_size=512, 
#     blur_radius=0, 
#     faces_per_pixel=1, 
#     bin_size=0
# )
# # We can add a point light in front of the object. 
# lights = DirectionalLights(device=device,
#                            ambient_color=[[0.4, 0.4, 0.4]],
#                            diffuse_color=[[0.8, 0.8, 0.8]],
#                            specular_color=[[0.3, 0.3, 0.3]],
#                            direction=[[-1.0, -1.0, 1.0]])
# phong_renderer = MeshRenderer(
#     rasterizer=MeshRasterizer(
#         cameras=cameras, 
#         raster_settings=raster_settings
#     ),
#     shader=PhongShader(device=device, lights=lights)
# )

# To blend the 100 faces we set a few parameters which control the opacity and the sharpness of 
# edges. Refer to blending.py for more details. 
blend_params = BlendParams(sigma=1e-4, gamma=1e-4)

# Define the settings for rasterization and shading. Here we set the output image to be of size
# 256x256. To form the blended image we use 100 faces for each pixel. Refer to rasterize_meshes.py
# for an explanation of this parameter. 
raster_settings = RasterizationSettings(
    image_size=256, 
    blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
    faces_per_pixel=100, 
    bin_size=0
)

# Create a silhouette mesh renderer by composing a rasterizer and a shader. 
phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SilhouetteShader(blend_params=blend_params)
)


# Initialize a model using the renderer, mesh and reference image
model = Model().to(device)

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

data = pickle.load(open("/shared-folder/AugmentedAutoencoder/pytorch3d/training-data/training-codes.p","rb"), encoding="latin1")

losses = []

for i in np.arange(100):
    optimizer.zero_grad()
    ae_code = data["codes"][i]
    ae_code = torch.from_numpy(ae_code).to(device).unsqueeze(0)
    predicted_pose = model(ae_code)

    R = torch.from_numpy(data["Rs"][i]).to(device).unsqueeze(0)
    T = torch.from_numpy(np.array([0.0,  0.0, 3.5], dtype=np.float32)).to(device).unsqueeze(0)
    gt_img = phong_renderer(meshes_world=object_mesh, R=R, T=T)
    #print(gt_img.shape)
    
    loss, predicted_image = Loss(gt_img, predicted_pose, phong_renderer, object_mesh)
    loss.backward()
    optimizer.step()

    print("Step: {0} - loss: {1}".format(i,loss.data))
    losses.append(loss)
    gt_img = gt_img.detach().cpu().numpy().squeeze()[..., 3]
    predicted_image = predicted_image.detach().squeeze().cpu().numpy()[..., 3]

    #continue
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Loss: {0}".format(loss.data))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_image)
    plt.title("Predicted")

    plt.subplot(1, 2, 2)
    plt.imshow(gt_img)
    plt.title("GT")

    fig.tight_layout()
    fig.savefig("output/training{0}.png".format(i), dpi=fig.dpi)
    plt.close()
    #plt.show()

#plt.plot(losses)
#plt.show()
