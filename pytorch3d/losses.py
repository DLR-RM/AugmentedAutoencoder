import torch
from utils.utils import *

def Loss(gt_images, renderer, predicted_poses, ts, mean, std, loss_method="diff"):
    Rs_pred = quat2mat(predicted_poses)
    predicted_images = renderer.renderBatch(Rs_pred, ts)
    predicted_images = (predicted_images-mean)/std
    
    if(loss_method=="diff"):
        diff = torch.abs(gt_images - predicted_images).flatten(start_dim=1)
        loss = torch.mean(diff)
        return loss, torch.mean(diff, dim=1), predicted_images
    elif(loss_method=="diff-bootstrap"):
        bootstrap = 4
        error = torch.abs(gt_images - predicted_images).flatten(start_dim=1)
        k = error.shape[1]/bootstrap
        topk, indices = torch.topk(error, round(k), sorted=True)
        loss = torch.mean(topk)        
        return loss, topk, predicted_images
    
    print("Unknown loss specified")
    return -1, None, None
