import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

from utils import *
from model import Model

from BatchRender import BatchRender
        
# Set the cuda device 
device = torch.device("cuda:0")
torch.cuda.set_device(device)

br = BatchRender("./data/ikea_mug_scaled_reduced.obj", device)           

# Initialize and load the model
model = Model().to(device)
model.load_state_dict(torch.load("./output/model-epoch240.pt"))
model.eval()

data = pickle.load(open("/shared-folder/AugmentedAutoencoder/pytorch3d/training-data/training-codes.p","rb"), encoding="latin1")

losses = []

for i in np.arange(1000):
    # Load ground truth
    R_gt = data["Rs"][i]
    
    # Predict pose
    codes = []
    codes.append(data["codes"][i])
    batch_codes = torch.tensor(np.stack(codes), device=device, dtype=torch.float32) # Bx128
    
    pose_quat = model(batch_codes)
    R_predicted = quat2mat(pose_quat)
    #print(R_predicted)
    R_predicted = R_predicted.detach().cpu().numpy().squeeze()
    #print(R_predicted.shape)
    #print(R_predicted)
    
    # Render images
    T = np.array([0.0,  0.0, 3.5], dtype=np.float32)   
    Ts = [T.copy(), T.copy()]
    Rs = [R_gt.copy(), R_predicted.copy()]

    print(Rs[0])
    print(Rs[1])

    images = br.renderBatch(Rs, Ts)

    diff = torch.abs(images[0] - images[1])
    loss = torch.mean(diff)
    losses.append(loss.detach().cpu().numpy())
    print("Step: {0} Loss: {1}".format(i,loss))
    
    #images = images.detach().cpu().numpy()
    #print(images.shape)

    if(loss < 0.01):
        continue
        fig = plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)
        plt.imshow((images[0])[..., 3])
        plt.title("GT")
    
        plt.subplot(1, 2, 2)
        plt.imshow((images[1])[..., 3])
        plt.title("Predicted")

        fig.tight_layout()
        #fig.savefig("output/training{0}.png".format(i), dpi=fig.dpi)
        #plt.close()
        plt.show()

print(np.mean(losses))
