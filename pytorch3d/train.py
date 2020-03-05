import os
import shutil
import torch
import numpy as np
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import configparser
import json
import argparse

from utils.utils import *

from Model import Model
from BatchRender import BatchRender 
from losses import Loss

learning_rate = -1
optimizer = None

def main():
    global learning_rate, optimizer
    # Read configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    arguments = parser.parse_args()
    
    cfg_file_path = os.path.join("./experiments", arguments.experiment_name)
    args = configparser.ConfigParser()
    args.read(cfg_file_path)

    # Set the cuda device 
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up batch renderer
    br = BatchRender(args.get('Dataset', 'CAD_PATH'),
                     device,
                     batch_size=args.getint('Training', 'BATCH_SIZE'),
                     render_method=args.get('Rendering', 'SHADER'),
                     image_size=args.getint('Rendering', 'IMAGE_SIZE'))
                   

    # Initialize a model using the renderer, mesh and reference image
    model = Model().to(device)
    #model.load_state_dict(torch.load("./output/model-epoch720.pt"))

    # Create an optimizer. Here we are using Adam and we pass in the parameters of the model
    learning_rate=args.getfloat('Training', 'LEARNING_RATE')

    data = pickle.load(open(args.get('Dataset', 'TRAIN_DATA_PATH'),"rb"), encoding="latin1")
    data["codes"] = data["codes"][:83*12]
    output_path = args.get('Training', 'OUTPUT_PATH')
    prepareDir(output_path)
    shutil.copy(cfg_file_path, os.path.join(output_path, cfg_file_path.split("/")[-1]))

    train_loss = []

    mean = 0
    std = 1
    mean, std = calcMeanVar(br, data, device, json.loads(args.get('Rendering', 'T')))

    np.random.seed(seed=args.getint('Training', 'RANDOM_SEED'))
    for e in np.arange(args.getint('Training', 'NUM_ITER')):
        loss = trainEpoch(mean, std, e, br, data, model, device, output_path,
                          loss_method=args.get('Training', 'LOSS'),
                          t=json.loads(args.get('Rendering', 'T')),
                          visualize=args.getboolean('Training', 'SAVE_IMAGES'))
        train_loss.append(loss)
        list2file(train_loss, os.path.join(output_path, "train-loss.csv"))
        plotLoss(train_loss, os.path.join(output_path, "train-loss.png"))
        print("-"*20)
        print("Epoch: {0} - loss: {1}".format(e,loss))
        print("-"*20)
    
def trainEpoch(mean, std, e, br, data, model,
               device, output_path, loss_method, t,
               visualize=False):
    global learning_rate, optimizer
    
    losses = []
    batch_size = br.batch_size
    num_samples = len(data["codes"])
    data_indeces = np.arange(num_samples)

    if(e % 2 == 1):
        learning_rate = learning_rate * 0.9
        print("Current learning rate: {0}".format(learning_rate))
    
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
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

        # Prepare ground truth poses for the loss function
        T = np.array(t, dtype=np.float32)
        Rs = []
        ts = []
        for b in curr_batch:
            Rs.append(data["Rs"][b])
            ts.append(T.copy())
        
        loss, batch_loss, gt_images, predicted_images = Loss(predicted_poses, Rs, br, ts,
                                                             mean, std, loss_method=loss_method)
    
        loss.backward()
        optimizer.step()

        print("Step: {0}/{1} - loss: {2}".format(i,round(num_samples/batch_size),loss.data))
        losses.append(loss.data.detach().cpu().numpy())

        if(visualize):
            batch_img_dir = os.path.join(output_path, "images/epoch{0}".format(e))
            prepareDir(batch_img_dir)
            gt_img = (gt_images[0]).detach().cpu().numpy()
            predicted_img = (predicted_images[0]).detach().cpu().numpy()
            
            vmin = min(np.min(gt_img), np.min(predicted_img))
            vmax = max(np.max(gt_img), np.max(predicted_img))
            
            fig = plt.figure(figsize=(12, 9))

            for viewNum in np.arange(4):
                plotView(viewNum, vmin, vmax, gt_images, predicted_images,
                         predicted_poses, batch_loss, batch_size)            
            fig.tight_layout()
            fig.savefig(os.path.join(batch_img_dir, "epoch{0}-batch{1}.png".format(e,i)), dpi=fig.dpi)
            plt.close()

    model_dir = os.path.join(output_path, "models/")
    prepareDir(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir,"model-epoch{0}.pt".format(e)))
    return np.mean(losses)

if __name__ == '__main__':
    main()
