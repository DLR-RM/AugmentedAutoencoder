import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

# io utils
from pytorch3d.io import load_obj

from utils import *
from model import Model

from BatchRender import BatchRender

def Loss(gt_img, predicted_poses, batch_renderer, method="diff"):

    # Render the object using the predicted_poses
    T = np.array([0.0,  0.0, 3.5], dtype=np.float32)

    Rs = quat2mat(predicted_poses)
    ts = []
    for k in br.batch_indeces:
        ts.append(T.copy())
    
    image_ref = batch_renderer.renderBatch(Rs, ts)[...,3]
    image_ref = torch.clamp(image_ref, 3.0, 4.0)

    if(method=="diff"):
        diff = torch.abs(gt_img - image_ref)
        loss = torch.mean(diff)
    elif(method=="diff-bootstrap"):
        bootstrap = 2
        batch_size = len(Rs)
        img_size = gt_img.shape[1]
        k = (img_size*img_size*batch_size)/bootstrap
        error = ((gt_img - image_ref) ** 2)
        topk, indices = torch.topk(error.flatten(), round(k), sorted=True)
        loss = torch.mean(topk)        
        #loss = torch.mean(topk[batch_size*10:])
    else:
        print("Unknown loss specified")
        return -1, None
    return loss, image_ref

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def trainEpoch(e, visualize=False):
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
    
        gt_img = br.renderBatch(Rs, ts)[...,3]
        gt_img = torch.clamp(gt_img, 3.0, 4.0)
    
        loss, predicted_image = Loss(gt_img, predicted_poses, br)
    
        loss.backward()
        optimizer.step()

        print("Step: {0}/{1} - loss: {2}".format(i,round(num_samples/batch_size),loss.data))
        losses.append(loss.data.detach().cpu().numpy())

        #plt.hist((gt_img[0].flatten()).detach().cpu().numpy(), bins=20)
        #plt.show()

        if(visualize):
            fig = plt.figure(figsize=(10, 10))
            plt.subplot(1, 2, 1)
            plt.imshow((gt_img[0]).detach().cpu().numpy())
            plt.title("GT")
            
            plt.subplot(1, 2, 2)
            plt.imshow((predicted_image[0]).detach().cpu().numpy())
            plt.title("Predicted")

            fig.tight_layout()
            fig.savefig(output_path + "test{0:.5f}.png".format(loss), dpi=fig.dpi)
            plt.close()
            #plt.show()

            
    torch.save(model.state_dict(), output_path + "model-epoch{0}.pt".format(e))    
    return np.mean(losses)
        

# Set the cuda device 
device = torch.device("cuda:0")
torch.cuda.set_device(device)

# Set up batch renderer
br = BatchRender("./data/cad-models/ikea_mug_scaled_reduced.obj",
                 device,
                 batch_size=12,
                 render_method="silhouette",
                 image_size=256)
                   

# Initialize a model using the renderer, mesh and reference image
model = Model().to(device)
#model.load_state_dict(torch.load("./output/model-epoch720.pt"))

# Create an optimizer. Here we are using Adam and we pass in the parameters of the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

data = pickle.load(open("./data/dataset-1k/training-codes.p","rb"), encoding="latin1")
output_path = "./output/silhouette/"

train_loss = []

np.random.seed(seed=42)
for e in np.arange(2000):
    loss = trainEpoch(e, visualize=True)
    train_loss.append(loss)
    list2file(train_loss, "{0}train-loss.csv".format(output_path))
    print("-"*20)
    print("Epoch: {0} - loss: {1}".format(e,loss))
    print("-"*20)
