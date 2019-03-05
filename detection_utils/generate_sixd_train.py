import numpy as np
import cv2
import os
import argparse
import progressbar as pb
import glob

from imgaug.augmenters import *
from auto_pose.ae.pysixd_stuff import misc

import yaml


visualize = True
gt_masks = False
dataset = 'tless'
num_train_imgs = 80000
max_objects_in_scene = 6
noofvoc_imgs = 15000
min_visib = 0.6
blackness_thres = 16
vocpath = '/home_local_nvme/sund_ma/data/VOCdevkit/VOC2012/JPEGImages/*.jpg'
voc_img_pathes = glob.glob(vocpath)
output_path = '/home_local_nvme/sund_ma/data/scene_renderings/linemod_real_imgs_voc_rotated/01/rgb'  


def load_info(path):
    with open(path, 'r') as f:
        info = yaml.load(f, Loader=yaml.CLoader)
        for eid in info.keys():
            if 'cam_K' in info[eid].keys():
                info[eid]['cam_K'] = np.array(info[eid]['cam_K']).reshape((3, 3))
            if 'cam_R_w2c' in info[eid].keys():
                info[eid]['cam_R_w2c'] = np.array(info[eid]['cam_R_w2c']).reshape((3, 3))
            if 'cam_t_w2c' in info[eid].keys():
                info[eid]['cam_t_w2c'] = np.array(info[eid]['cam_t_w2c']).reshape((3, 1))
    return info

def load_gt(path):
    with open(path, 'r') as f:
        gts = yaml.load(f, Loader=yaml.CLoader)
        for im_id, gts_im in gts.items():
            for gt in gts_im:
                if 'cam_R_m2c' in gt.keys():
                    gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
                if 'cam_t_m2c' in gt.keys():
                    gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
    return gts



def main():

    if dataset == 'linemod':
        sixd_train_path = '/home_local_nvme/sund_ma/data/train_linemod'
        cad_path = '/home_local/sund_ma/data/linemod_dataset/models'
        W, H = 640,480
        noofobjects = 15
    elif dataset == 'tless':
        sixd_train_path = '/home_local/sund_ma/data/t-less/t-less_v2/train_primesense'
        cad_path = '/home_local/sund_ma/data/t-less/t-less_v2/models_reconst'
        W, H = 720,540
        noofobjects = 30

    # with ground truth masks
    if gt_masks:
        from auto_pose.meshrenderer import meshrenderer_phong
        models_cad_files = sorted(glob.glob(os.path.join(cad_path,'*.ply')))
        renderer = meshrenderer_phong.Renderer(
            models_cad_files, 
            1
        )
        obj_gts = []
        for obj_id in xrange(1,noofobjects+1):
            obj_gts.append(load_gt(os.path.join(sixd_train_path,'{:02d}'.format(obj_id),'gt.yml')))

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    voc_imgs = []
    print 'loading bg'
    for i,path in enumerate(voc_img_pathes[:noofvoc_imgs]):
        voc_imgs.append(cv2.resize(cv2.imread(path),(W,H)))
        print i,voc_imgs[-1].shape

    obj_infos = []
    for obj_id in xrange(1,noofobjects+1):
        obj_infos.append(load_info(os.path.join(sixd_train_path,'{:02d}'.format(obj_id),'info.yml')))


    augmenters = Sequential([
        Sometimes(0.2, GaussianBlur(0.4)),
        Sometimes(0.1, AdditiveGaussianNoise(scale=10, per_channel=True)),
        Sometimes(0.4, Add((-15, 15), per_channel=0.5)),
        #Sometimes(0.3, Invert(0.2, per_channel=True)),
        Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
        # Sometimes(0.5, Multiply((0.6, 1.4))),
        Sometimes(0.5, ContrastNormalization((0.5, 2.2), per_channel=0.3)),
        #Sometimes(0.2, CoarseDropout( p=0.1, size_px = 10, size_percent=0.001) )
    ], random_order=True)


    new_scene_gt = {}

    bar = pb.ProgressBar(
        maxval=num_train_imgs, 
        widgets=[' [', pb.Timer(), ' | ', pb.Counter('%0{}d / {}'.format(len(str(num_train_imgs)), num_train_imgs)), ' ] ', 
        pb.Bar(), ' (', pb.ETA(), ') ']
    )

    for i in bar(xrange(num_train_imgs)):
        new_scene_gt[i] = []
        new_train_img = np.zeros((H,W,3),dtype=np.uint8)
        new_train_mask = np.zeros((H,W,1),dtype=np.uint8)
        random_imgs = []
        orig_bbs = []
        random_trans = []
        for k in xrange(max_objects_in_scene):
            rand_obj_id = np.random.randint(0,noofobjects)
            rand_view_id = np.random.randint(0,len(obj_infos[rand_obj_id]))
            img_path = os.path.join(sixd_train_path,'{:02d}'.format(rand_obj_id+1),'rgb','{:04d}.png'.format(rand_view_id))
            
            rand_img = cv2.imread(img_path)
            # rand_depth_img = load_depth2(os.path.join(sixd_train_path,'{:02d}'.format(rand_obj_id+1),'depth','{:04d}.png'.format(rand_view_id)))
            
            # random rotate in-plane
            rot_angle= np.random.rand()*360
            M = cv2.getRotationMatrix2D((int(rand_img.shape[1]/2),int(rand_img.shape[0]/2)),rot_angle,1)
            rand_img = cv2.warpAffine(rand_img, M, (rand_img.shape[1],rand_img.shape[0]))

            # with ground truth masks
            if gt_masks:
                gt = obj_gts[rand_obj_id][rand_view_id][0]
                K = obj_infos[rand_obj_id][rand_view_id]['cam_K']
                _, depth = renderer.render(rand_obj_id,rand_img.shape[1],rand_img.shape[0],K,gt['cam_R_m2c'],gt['cam_t_m2c'],10,5000)
                depth = cv2.warpAffine(depth, M, (depth.shape[1],depth.shape[0]))
                mask = depth > 0
                rand_img[mask == False] = 0
            else:
                rand_img[(rand_img[:,:,0] < blackness_thres) & (rand_img[:,:,1] < blackness_thres) & (rand_img[:,:,2] < blackness_thres)] = 0
                mask = np.all(rand_img > 0,axis=2)

            # print rand_img.shape,mask.shape
            # cv2.imshow('mask2',mask.astype(np.float32))
            # cv2.imshow('rand_img',rand_img)
            # cv2.waitKey(0)

            ys, xs = np.nonzero(mask)
            new_bb = misc.calc_2d_bbox(xs, ys, mask.shape[:2])
            
            if dataset == 'tless':
                #tless specific
                crop_x = np.array([20,380]) + np.random.randint(-15,15)
                crop_y = np.array([20,380]) + np.random.randint(-15,15)
            elif dataset == 'linemod':
                #linemod specific
                crop_x = np.array([80,560]) + np.random.randint(-20,20)
                crop_y = np.array([0,480])# + np.random.randint(-20,20)

            mask = mask[crop_y[0]:crop_y[1],crop_x[0]:crop_x[1]]
            rand_img = rand_img[crop_y[0]:crop_y[1],crop_x[0]:crop_x[1]] 




            orig_H, orig_W = rand_img.shape[:2]
            s = 0.5*np.random.rand()+0.5
            new_H, new_W = int(s*orig_H),int(s*orig_W)
            scaled_img = cv2.resize(rand_img,(new_W,new_H), interpolation=cv2.INTER_NEAREST)
            scaled_mask = cv2.resize(mask.astype(np.int32).reshape(orig_W,orig_H,1),(new_W,new_H), interpolation=cv2.INTER_NEAREST)
            y_offset = np.random.randint(0,H-scaled_img.shape[0])
            x_offset = np.random.randint(0,W-scaled_img.shape[1])

            y1, y2 = y_offset, y_offset + scaled_img.shape[0]
            x1, x2 = x_offset, x_offset + scaled_img.shape[1]

            alpha_s = np.dstack((scaled_mask,scaled_mask,scaled_mask)) > 0
            alpha_l = 1.0 - alpha_s
            old_train_mask = new_train_mask.copy()
            new_train_mask[y1:y2, x1:x2, 0] = alpha_s[:,:,0] * scaled_mask + alpha_l[:,:,0] * new_train_mask[y1:y2, x1:x2, 0]

            old_scene_pix = len(old_train_mask[y1:y2, x1:x2, 0]>0)
            new_scene_pix = len(new_train_mask>0)
            new_mask_pix = len(scaled_mask>0)
            if (new_scene_pix-old_scene_pix)/float(new_mask_pix) < min_visib:
                new_train_mask = old_train_mask.copy()
                continue


            new_train_img[y1:y2, x1:x2, :] = alpha_s * scaled_img + alpha_l * new_train_img[y1:y2, x1:x2, :]

            x,y,w,h = np.round((np.array(new_bb)+np.array([-crop_x[0],-crop_y[0],0,0]))*s+np.array([x_offset,y_offset,0,0])).astype(np.int32)
            # x,y,w,h = np.round(np.array(gt['obj_bb'])*s+np.array([x_offset,y_offset,0,0])).astype(np.int32)
            new_scene_gt[i].append({'obj_id':rand_obj_id+1,'obj_bb':[x,y,w,h]})

            
        
        bg = voc_imgs[np.random.randint(0,noofvoc_imgs)]
        stacked_new_train_mask = np.dstack((new_train_mask,new_train_mask,new_train_mask))
        new_train_img[stacked_new_train_mask==0] = bg[stacked_new_train_mask==0]
        new_train_img = augmenters.augment_image(new_train_img)

        if visualize:
            print new_scene_gt[i]
            for sc_gt in new_scene_gt[i]: 
                x,y,w,h = sc_gt['obj_bb']
                cv2.rectangle(new_train_img, (x, y), (x+w, y+h), color=(32, 192, 192))
            cv2.imshow('new_train_img', new_train_img)
            cv2.imshow('new_train_mask', new_train_mask.astype(np.float32))
            cv2.waitKey(0)

        cv2.imwrite(os.path.join(output_path,'%s.png' % i), new_train_img)



    with open(os.path.join(os.path.dirname(output_path),'gt.yml'), 'w') as f:
        yaml.dump(new_scene_gt, f, Dumper=yaml.CDumper, width=10000)



if __name__ == '__main__':
    main()


