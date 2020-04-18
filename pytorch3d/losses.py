import torch
import numpy as np
from utils.utils import *
from utils.tools import *
import torch.nn as nn

def Loss(predicted_poses, gt_poses, renderer, ts, mean, std, loss_method="diff", views=None):
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

    elif(loss_method=="bce-loss"):
        gt_imgs = []
        predicted_imgs = []
        Rs_gt = torch.tensor(np.stack(gt_poses), device=renderer.device,
                                dtype=torch.float32)
        #Rs_predicted = quat2mat(predicted_poses)
        Rs_predicted = compute_rotation_matrix_from_ortho6d(predicted_poses)
        for v in views:
            # Render ground truth images
            Rs_new = torch.matmul(Rs_gt, v.to(renderer.device))
            gt_images = renderer.renderBatch(Rs_new, ts)
            gt_images = (gt_images-mean)/std
            gt_imgs.append(gt_images)

            # Render images based on predicted pose
            Rs_new = torch.matmul(Rs_predicted, v.to(renderer.device))
            predicted_images = renderer.renderBatch(Rs_new, ts)
            predicted_images = (predicted_images-mean)/std
            predicted_imgs.append(predicted_images)

        gt_imgs = torch.cat(gt_imgs)
        predicted_imgs = torch.cat(predicted_imgs)
        diff = torch.abs(gt_imgs - predicted_imgs).flatten(start_dim=1)

        loss = nn.BCEWithLogitsLoss(reduction="none")
        loss = loss(predicted_imgs.flatten(start_dim=1),gt_imgs.flatten(start_dim=1))
        loss = torch.mean(loss, dim=1)

        return torch.mean(loss), loss, gt_imgs, predicted_imgs

    elif(loss_method=="bce-loss-sum"):
        gt_imgs = []
        predicted_imgs = []
        Rs_gt = torch.tensor(np.stack(gt_poses), device=renderer.device,
                                dtype=torch.float32)
        #Rs_predicted = quat2mat(predicted_poses)
        Rs_predicted = compute_rotation_matrix_from_ortho6d(predicted_poses)
        for v in views:
            # Render ground truth images
            Rs_new = torch.matmul(Rs_gt, v.to(renderer.device))
            gt_images = renderer.renderBatch(Rs_new, ts)
            gt_images = (gt_images-mean)/std
            gt_imgs.append(gt_images)

            # Render images based on predicted pose
            Rs_new = torch.matmul(Rs_predicted, v.to(renderer.device))
            predicted_images = renderer.renderBatch(Rs_new, ts)
            predicted_images = (predicted_images-mean)/std
            predicted_imgs.append(predicted_images)

        gt_imgs = torch.cat(gt_imgs)
        predicted_imgs = torch.cat(predicted_imgs)
        diff = torch.abs(gt_imgs - predicted_imgs).flatten(start_dim=1)

        loss = nn.BCEWithLogitsLoss(reduction="none")
        loss = loss(predicted_imgs.flatten(start_dim=1),gt_imgs.flatten(start_dim=1))
        loss = torch.sum(loss, dim=1)

        return torch.sum(loss), loss, gt_imgs, predicted_imgs

    elif(loss_method=="l2-pose"):
        gt_imgs = []
        predicted_imgs = []
        Rs_gt = torch.tensor(np.stack(gt_poses), device=renderer.device,
                                dtype=torch.float32)
        #Rs_predicted = quat2mat(predicted_poses)
        Rs_predicted = compute_rotation_matrix_from_ortho6d(predicted_poses)
        for v in views:
            # Render ground truth images
            Rs_new = torch.matmul(Rs_gt, v.to(renderer.device))
            gt_images = renderer.renderBatch(Rs_new, ts)
            gt_images = (gt_images-mean)/std
            gt_imgs.append(gt_images)

            # Render images based on predicted pose
            Rs_new = torch.matmul(Rs_predicted, v.to(renderer.device))
            predicted_images = renderer.renderBatch(Rs_new, ts)
            predicted_images = (predicted_images-mean)/std
            predicted_imgs.append(predicted_images)

        gt_imgs = torch.cat(gt_imgs)
        predicted_imgs = torch.cat(predicted_imgs)
        diff = torch.abs(gt_imgs - predicted_imgs).flatten(start_dim=1)

        loss = nn.MSELoss()
        loss = loss(Rs_predicted, Rs_gt)
        return loss, torch.mean(diff, dim=1), gt_imgs, predicted_imgs
        
    elif(loss_method=="multiview"):
        gt_imgs = []
        predicted_imgs = []
        Rs_gt = torch.tensor(np.stack(gt_poses), device=renderer.device,
                                dtype=torch.float32)
        #Rs_predicted = quat2mat(predicted_poses)
        Rs_predicted = compute_rotation_matrix_from_ortho6d(predicted_poses)
        for v in views:
            # Render ground truth images
            Rs_new = torch.matmul(Rs_gt, v.to(renderer.device))
            gt_images = renderer.renderBatch(Rs_new, ts)
            gt_images = (gt_images-mean)/std
            gt_imgs.append(gt_images)

            # Render images based on predicted pose
            Rs_new = torch.matmul(Rs_predicted, v.to(renderer.device))
            predicted_images = renderer.renderBatch(Rs_new, ts)
            predicted_images = (predicted_images-mean)/std
            predicted_imgs.append(predicted_images)

        gt_imgs = torch.cat(gt_imgs)
        predicted_imgs = torch.cat(predicted_imgs)
        diff = torch.abs(gt_imgs - predicted_imgs).flatten(start_dim=1)
        loss = torch.mean(diff)
        #print(predicted_imgs.shape)
        return loss, torch.mean(diff, dim=1), gt_imgs, predicted_imgs

    elif(loss_method=="multiview-l2"):
        gt_imgs = []
        predicted_imgs = []
        Rs_gt = torch.tensor(np.stack(gt_poses), device=renderer.device,
                                dtype=torch.float32)
        Rs_predicted = compute_rotation_matrix_from_ortho6d(predicted_poses)
        for v in views:
            # Render ground truth images
            Rs_new = torch.matmul(Rs_gt, v.to(renderer.device))
            gt_images = renderer.renderBatch(Rs_new, ts)
            gt_images = (gt_images-mean)/std
            gt_imgs.append(gt_images)

            # Render images based on predicted pose
            Rs_new = torch.matmul(Rs_predicted, v.to(renderer.device))
            predicted_images = renderer.renderBatch(Rs_new, ts)
            predicted_images = (predicted_images-mean)/std
            predicted_imgs.append(predicted_images)

        gt_imgs = torch.cat(gt_imgs)
        predicted_imgs = torch.cat(predicted_imgs)

        for i in range(gt_imgs.shape[0]):
            mask = gt_imgs[i] == -1.0
            gt_imgs[i][mask] = 0.0
            predicted_imgs[i][mask] = 0.0

            mask = predicted_imgs[i] == -1.0
            gt_imgs[i][mask] = 0.0
            predicted_imgs[i][mask] = 0.0

        loss = nn.MSELoss(reduction="none")
        loss = loss(gt_imgs.flatten(start_dim=1), predicted_imgs.flatten(start_dim=1))
        loss = torch.mean(loss, dim=1)
        return torch.mean(loss), loss, gt_imgs, predicted_imgs

    elif(loss_method=='loss_landscape'):
        predicted_images = predicted_poses
        predicted_images = (predicted_images-mean)/std
        gt_images = gt_poses
        gt_images = (gt_images-mean)/std

        diff = torch.abs(gt_images - predicted_images).flatten(start_dim=1)
        loss = torch.mean(diff)
        return loss, torch.mean(diff, dim=1), gt_images, predicted_images

    print("Unknown loss specified")
    return -1, None, None, None
