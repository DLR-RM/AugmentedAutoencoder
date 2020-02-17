import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

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
    SilhouetteShader, PhongShader, PointLights, DirectionalLights
)

from utils import *
from model import Model

from BatchRender import BatchRender

def Loss(gt_img, predicted_poses, batch_renderer):

    # Render the object using the predicted_poses
    T = np.array([0.0,  0.0, 3.5], dtype=np.float32)

    Rs = quat2mat(predicted_poses)
    ts = []
    for k in br.batch_indeces:
        ts.append(T.copy())
    
    image_ref = batch_renderer.renderBatch(Rs, ts)

    print(image_ref.shape)
    print(gt_img.shape)
       
    #diff = torch.abs(gt_img - image_ref)
    #topk, indices = torch.topk(diff.flatten(), 10000)
    #loss = torch.mean(topk)
    loss = torch.mean((gt_img - image_ref) ** 2)
    #error = ((gt_img - image_ref) ** 2)
    #topk, indices = torch.topk(error.flatten(), 16000)
    #loss = torch.mean(topk)
    return loss, image_ref

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# Set the cuda device 
device = torch.device("cuda:0")
torch.cuda.set_device(device)

br = BatchRender("./data/ikea_mug_scaled_reduced.obj", device)
                   

# Initialize a model using the renderer, mesh and reference image
model = Model().to(device)

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

data = pickle.load(open("/shared-folder/AugmentedAutoencoder/pytorch3d/training-data/training-codes.p","rb"), encoding="latin1")

losses = []

data_indeces = np.arange(len(data["codes"]))

for curr_batch in batch(data_indeces, 12):
    optimizer.zero_grad()
    codes = []
    for b in curr_batch:
        codes.append(data["codes"][b])
    batch_codes = torch.tensor(np.stack(codes), device=device, dtype=torch.float32) # Bx128
    #ae_code = data["codes"][i]
    #ae_code = torch.from_numpy(ae_code).to(device).unsqueeze(0)
    predicted_poses = model(batch_codes)
    print("Predicted shape: {0}".format(predicted_poses.shape))

    T = np.array([0.0,  0.0, 3.5], dtype=np.float32)

    Rs = []
    ts = []
    print(curr_batch)
    for b in curr_batch:
        Rs.append(data["Rs"][b])
        ts.append(T.copy())
    
    gt_img = br.renderBatch(Rs, ts)
    print(gt_img.shape)
    
    loss, predicted_image = Loss(gt_img, predicted_poses, br)
    
    loss.backward()
    optimizer.step()

    print("Step: {0} - loss: {1}".format(curr_batch[-1],loss.data))
    losses.append(loss)
    gt_img = gt_img.detach().cpu().numpy().squeeze()[..., 3]
    predicted_image = predicted_image.detach().squeeze().cpu().numpy()[..., 3]


#plt.plot(losses)
#plt.show()
