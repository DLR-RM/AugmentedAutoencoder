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

def Loss(gt_img, predicted_poses, batch_renderer, method="diff-bootstrap"):

    # Render the object using the predicted_poses
    T = np.array([0.0,  0.0, 3.5], dtype=np.float32)

    Rs = quat2mat(predicted_poses)
    ts = []
    for k in br.batch_indeces:
        ts.append(T.copy())
    
    image_ref = batch_renderer.renderBatch(Rs, ts)

    if(method=="diff"):
        diff = torch.abs(gt_img - image_ref)
        loss = torch.mean(diff)
    elif(method=="diff-bootstrap"):
        bootstrap = 2
        batch_size = len(Rs)
        img_size = gt_img.shape[1]
        k = (img_size*img_size*batch_size)/bootstrap
        #print("Batch size: {0}\n Img size: {1}\n Bootstrap ratio: {2}\n K-value: {3}".format(batch_size, img_size, bootstrap, k))
        error = ((gt_img - image_ref) ** 2)
        topk, indices = torch.topk(error.flatten(), round(k))
        loss = torch.mean(topk)
    else:
        print("Unknown loss specified")
        return -1, None
    return loss, image_ref

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def trainEpoch(e):
    losses = []
    batch_size = br.batch_size
    num_samples = len(data["codes"])
    data_indeces = np.arange(num_samples)
    np.random.shuffle(data_indeces)
    for i,curr_batch in enumerate(batch(data_indeces, batch_size)):
        if(len(curr_batch) != batch_size):
            continue
        optimizer.zero_grad()
        codes = []
        for b in curr_batch:
            codes.append(data["codes"][b])
        batch_codes = torch.tensor(np.stack(codes), device=device, dtype=torch.float32) # Bx128

        predicted_poses = model(batch_codes)
            
        T = np.array([0.0,  0.0, 3.5], dtype=np.float32)

        Rs = []
        ts = []
        for b in curr_batch:
            Rs.append(data["Rs"][b])
            ts.append(T.copy())
    
        gt_img = br.renderBatch(Rs, ts)
    
        loss, predicted_image = Loss(gt_img, predicted_poses, br)
    
        loss.backward()
        optimizer.step()

        print("Step: {0}/{1} - loss: {2}".format(i,round(num_samples/batch_size),loss.data))
        losses.append(loss.data.detach().cpu().numpy())

        if(loss < 0.04):  
            fig = plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow((gt_img[0]).detach().cpu().numpy()[..., 3])
            plt.title("GT")
            
            plt.subplot(1, 2, 2)
            plt.imshow((predicted_image[0]).detach().cpu().numpy()[..., 3])
            plt.title("Predicted")

            fig.tight_layout()
            fig.savefig("output/test{0:.5f}.png".format(loss), dpi=fig.dpi)
            plt.close()
            #plt.show()

            
    torch.save(model.state_dict(), "./output/model-epoch{0}.pt".format(e))    
    return np.mean(losses)
        

# Set the cuda device 
device = torch.device("cuda:0")
torch.cuda.set_device(device)

br = BatchRender("./data/ikea_mug_scaled_reduced.obj", device, batch_size=38)
                   

# Initialize a model using the renderer, mesh and reference image
model = Model().to(device)
model.load_state_dict(torch.load("./output/model-epoch13.pt"))

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

data = pickle.load(open("/shared-folder/AugmentedAutoencoder/pytorch3d/training-data/training-codes-1k.p","rb"), encoding="latin1")


for e in np.arange(2000):
    np.random.seed(seed=42)
    loss = trainEpoch(e)
    print("-"*20)
    print("Epoch: {0} - loss: {1}".format(e,loss))
    print("-"*20)
    #torch.save(model.state_dict(), "./output/model-epoch{0}.pt".format(e))
