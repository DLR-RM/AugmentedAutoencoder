import os
import shutil
import torch
import numpy as np
import pickle
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

    #gt = calcGroundtruth(br, data, device, json.loads(args.get('Rendering', 'T')))
    mean, std = calcMeanVar(br, data, device, json.loads(args.get('Rendering', 'T')))
    gt = None

    np.random.seed(seed=args.getint('Training', 'RANDOM_SEED'))
    for e in np.arange(args.getint('Training', 'NUM_ITER')):
        loss = trainEpoch(gt, mean, std, e, br, data, model, device, output_path,
                          loss_method=args.get('Training', 'LOSS'),
                          t=json.loads(args.get('Rendering', 'T')),
                          visualize=args.getboolean('Training', 'SAVE_IMAGES'))
        train_loss.append(loss)
        list2file(train_loss, os.path.join(output_path, "train-loss.csv"))
        plotLoss(train_loss, os.path.join(output_path, "train-loss.png"))
        print("-"*20)
        print("Epoch: {0} - loss: {1}".format(e,loss))
        print("-"*20)

def calcMeanVar(br, data, device, t):
    num_samples = len(data["codes"])
    data_indeces = np.arange(num_samples)
    np.random.shuffle(data_indeces)
    batch_size = br.batch_size

    all_data = []
    for i,curr_batch in enumerate(batch(data_indeces, batch_size)):
        # Render the ground truth images
        T = np.array(t, dtype=np.float32)
        Rs = []
        ts = []
        for b in curr_batch:
            Rs.append(data["Rs"][b])
            ts.append(T.copy())
        gt_images = br.renderBatch(Rs, ts)
        all_data.append(torch.mean(gt_images.flatten()))
        print("Step: {0}/{1}".format(i,round(num_samples/batch_size)))
    result = torch.FloatTensor(all_data) #torch.cat(all_data)
    print(torch.mean(result))
    print(torch.std(result))
    return torch.mean(result), torch.std(result)

    
def trainEpoch(gt, mean, std, e, br, data, model,
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

        # Render the ground truth images
        T = np.array(t, dtype=np.float32)
        Rs = []
        ts = []
        gt_images = []
        for b in curr_batch:
            Rs.append(data["Rs"][b])
            ts.append(T.copy())
            #gt_images.append(gt[b])
            #print(gt_images[-1].shape)
        gt_images = br.renderBatch(Rs, ts)
        #gt_images = torch.stack(gt_images)
        gt_images = (gt_images-mean)/std

        # Render the images using the predicted_poses
        #Rs_pred = quat2mat(predicted_poses)
        #predicted_images = br.renderBatch(Rs_pred, ts)
        #predicted_images = (predicted_images-mean)/std

        #(gt_images, renderer, predicted_poses, mean, std, method="diff"):
        loss, batch_loss, predicted_images = Loss(gt_images, br, predicted_poses, ts,
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

            #plt.hist(gt_img.flatten(), bins=20)
            #plt.hist(predicted_img.flatten(), bins=20)
            #plt.show()
            
            vmin = min(np.min(gt_img), np.min(predicted_img))
            vmax = max(np.max(gt_img), np.max(predicted_img))
            
            fig = plt.figure(figsize=(12, 5))
            fig.suptitle("loss: {0}".format(batch_loss[0].data))
            plt.subplot(1, 3, 1)
            plt.imshow(gt_img, vmin=vmin, vmax=vmax)
            plt.title("GT")
            
            plt.subplot(1, 3, 2)
            plt.imshow(predicted_img, vmin=vmin, vmax=vmax)
            predicted_pose = predicted_poses[0].detach().cpu().numpy()
            plt.title("Predicted: " + np.array2string(predicted_pose,precision=2))

            loss_contrib = np.abs(gt_img - predicted_img)
            plt.subplot(1, 3, 3)
            plt.imshow(loss_contrib) #, vmin=vmin, vmax=vmax)
            plt.title("L2-loss contribution")

            fig.tight_layout()
            fig.savefig(os.path.join(batch_img_dir, "epoch{0}-batch{1}.png".format(e,i)), dpi=fig.dpi)
            plt.close()

    model_dir = os.path.join(output_path, "models/")
    prepareDir(model_dir)
    torch.save(model.state_dict(), os.path.join(model_dir,"model-epoch{0}.pt".format(e)))
    return np.mean(losses)

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def prepareDir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)      


if __name__ == '__main__':
    main()
