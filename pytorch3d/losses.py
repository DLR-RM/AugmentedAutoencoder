import torch
import numpy as np
from utils.utils import *

def Loss(predicted_poses, gt_poses, renderer, ts, mean, std, loss_method="diff"):   
    if(loss_method=="diff"):
        Rs_pred = quat2mat(predicted_poses)
        predicted_images = renderer.renderBatch(Rs_pred, ts)
        predicted_images = (predicted_images-mean)/std
        gt_images = renderer.renderBatch(gt_poses, ts)
        gt_images = renderer.renderBatch(Rs_pred, ts)
        gt_images = (gt_images-mean)/std
        
        diff = torch.abs(gt_images - predicted_images).flatten(start_dim=1)
        loss = torch.mean(diff)
        return loss, torch.mean(diff, dim=1), gt_images, predicted_images
    elif(loss_method=="diff-bootstrap"):
        Rs_pred = quat2mat(predicted_poses)
        predicted_images = renderer.renderBatch(Rs_pred, ts)
        predicted_images = (predicted_images-mean)/std
        gt_images = renderer.renderBatch(gt_poses, ts)
        gt_images = renderer.renderBatch(Rs_pred, ts)
        gt_images = (gt_images-mean)/std
        
        bootstrap = 4
        error = torch.abs(gt_images - predicted_images).flatten(start_dim=1)
        k = error.shape[1]/bootstrap
        topk, indices = torch.topk(error, round(k), sorted=True)
        loss = torch.mean(topk)        
        return loss, topk, predicted_images, gt_images, predicted_images
    elif(loss_method=="multiview"):
        views = [torch.tensor([[1.0, 0.0, 0.0], # Original view
                               [0.0, 1.0, 0.0],
                               [0.0, 0.0, 1.0]],
                              device=renderer.device),
                 torch.tensor([[-0.5000042, -0.8660229, 0.0], # 120 degrees around z-axis 
                               [0.8660229, -0.5000042, 0.0],
                               [0.0, 0.0, 1.0]],
                              device=renderer.device),
                 torch.tensor([[-0.5000042,  0.4330151,  0.7499958], # 120 degrees around z-axis 
                               [0.8660229,  0.2500042,  0.4330151],  # then 120 degrees around x-axis
                               [0.0000000,  0.8660229, -0.5000042]],
                              device=renderer.device),
                 torch.tensor([[-0.5000042,  0.4330151, -0.7499958], # 120 degrees around z-axis 
                               [0.8660229,  0.2500042, -0.4330151], # then -120 degrees around x-axis
                               [0.0000000, -0.8660229, -0.5000042]],
                              device=renderer.device)]
        gt_imgs = []
        predicted_imgs = []
        Rs_gt = torch.tensor(np.stack(gt_poses), device=renderer.device,
                                dtype=torch.float32).permute(0,2,1) # Bx3x3
        Rs_predicted = quat2mat(predicted_poses)
        for v in views:
            # Render ground truth images
            Rs_new = torch.matmul(Rs_gt, v)
            gt_images = renderer.renderBatch(Rs_new, ts)
            gt_images = (gt_images-mean)/std
            gt_imgs.append(gt_images)
            
            # Render images based on predicted pose
            Rs_new = torch.matmul(Rs_predicted, v)
            predicted_images = renderer.renderBatch(Rs_new, ts)
            predicted_images = (predicted_images-mean)/std
            predicted_imgs.append(predicted_images)

        gt_imgs = torch.cat(gt_imgs)
        predicted_imgs = torch.cat(predicted_imgs)
        diff = torch.abs(gt_imgs - predicted_imgs).flatten(start_dim=1)
        loss = torch.mean(diff)
        #print(predicted_imgs.shape)
        return loss, torch.mean(diff, dim=1), gt_imgs, predicted_imgs                   
    print("Unknown loss specified")
    return -1, None, None, None
