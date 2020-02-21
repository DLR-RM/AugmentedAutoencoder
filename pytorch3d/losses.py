import torch

def Loss(gt_images, predicted_images, method="diff"):
    if(method=="diff"):
        diff = torch.abs(gt_images - predicted_images).flatten(start_dim=1)
        loss = torch.mean(diff)
        return loss, torch.mean(diff, dim=1)
    elif(method=="diff-bootstrap"):
        bootstrap = 4
        error = torch.abs(gt_images - predicted_images).flatten(start_dim=1)
        k = error.shape[1]/bootstrap
        topk, indices = torch.topk(error, round(k), sorted=True)
        loss = torch.mean(topk)        
        return loss, topk
    
    print("Unknown loss specified")
    return -1, None
