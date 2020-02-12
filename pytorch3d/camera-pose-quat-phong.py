import os
import torch
import numpy as np
import imageio
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage import img_as_ubyte

#from transforms3d.axangles import axangle2mat
import torchgeometry as tgm

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

# Convert quaternion to rotation matrix
# from: https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:,:1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat


class Model(nn.Module):
    def __init__(self, meshes, renderer, image_ref):
        super().__init__()
        self.meshes = meshes
        self.device = meshes.device
        self.renderer = renderer
        
        # Get the silhouette of the reference RGB image by finding all the non zero values. 
        #image_ref = torch.from_numpy((image_ref[..., :3].max(-1) != 0).astype(np.float32))
        image_ref = torch.from_numpy(image_ref.astype(np.float32))
        self.register_buffer('image_ref', image_ref)
        
        # Create an optimizable parameter for the camera as a quaternion. 
        self.camera_position = nn.Parameter(
            torch.from_numpy(np.array([-1.0, 1.2, -3.0], dtype=np.float32)).to(meshes.device))

    def forward(self):
        
        # Render the image using the updated camera orientation.
        # The translation to the camera T is fixed
        T = torch.from_numpy(np.array([0.0,  0.0, 2.0], dtype=np.float32)).to(device).unsqueeze(0)
        R = quat2mat(self.camera_position.unsqueeze(0))
        image = self.renderer(meshes_world=self.meshes.clone(), R=R, T=T)
        
        # Calculate the silhouette loss
        #loss = torch.sum((image - self.image_ref) ** 2)
        diff = torch.abs(image - self.image_ref)
        topk, indices = torch.topk(diff.flatten(), 10000)
        loss = torch.sum(topk)
        #loss = torch.mean(diff.flatten())
        return loss, image


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
teapot_mesh = Meshes(
    verts=[verts.to(device)],   
    faces=[faces.to(device)], 
    textures=textures
)


# Initialize an OpenGL perspective camera.
cameras = OpenGLPerspectiveCameras(device=device)

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
silhouette_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SilhouetteShader(blend_params=blend_params)
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
    shader=PhongShader(device=device, lights=lights)
)

# Render reference image
R = torch.from_numpy(np.array([0.4, 1.2, -3.0], dtype=np.float32)).to(device).unsqueeze(0)       
R = quat2mat(R)
T = torch.from_numpy(np.array([0.0,  0.0, 2.0], dtype=np.float32)).to(device).unsqueeze(0)      
image_ref = phong_renderer(meshes_world=teapot_mesh, R=R, T=T)
image_ref = image_ref.cpu().numpy()

# We will save images periodically and compose them into a GIF.
filename_output = "./camera-pose-optimization-demo.gif"
writer = imageio.get_writer(filename_output, mode='I', duration=0.03)

# Initialize a model using the renderer, mesh and reference image
model = Model(meshes=teapot_mesh, renderer=phong_renderer, image_ref=image_ref).to(device)

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

_, image_init = model()

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(image_init.detach().squeeze().cpu().numpy())
plt.grid("off")
plt.title("Starting position")

plt.subplot(1, 2, 2)
plt.imshow(model.image_ref.cpu().numpy().squeeze())
plt.grid("off")
plt.title("Reference silhouette")
plt.show()

last_loss = None

for i in np.arange(1000):
    optimizer.zero_grad()
    loss, out_image = model()
    loss.backward()
    optimizer.step()

    print("Step: {0} - loss: {1}".format(i,loss.data))
    
    #if last_loss is not None and (abs(last_loss-loss.data)<0.00001):
    #    break

    # Update last loss
    last_loss = loss.data
    
    # Save outputs to create a GIF. 
    if True: #i % 10 == 0:
        # Render the image using the updated camera orientation.
        # The translation to the camera T is fixed
        # This is only for producing the .gif file
        T = torch.from_numpy(np.array([0.0,  0.0, 2.0], dtype=np.float32)).to(device).unsqueeze(0)
        R = quat2mat(model.camera_position.unsqueeze(0))
        
        image = phong_renderer(meshes_world=model.meshes.clone(), R=R, T=T)
        image = image[0, ..., :3].detach().squeeze().cpu().numpy()
        image = img_as_ubyte(image)
        writer.append_data(image)

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.imshow(out_image.detach().squeeze().cpu().numpy())
plt.grid("off")
plt.title("End position")

plt.subplot(1, 2, 2)
plt.imshow(model.image_ref.cpu().numpy().squeeze())
plt.grid("off")
plt.title("Reference silhouette")
plt.show()

        
writer.close()
