import argparse
import progressbar as pb
import os, glob
import numpy as np

from imgaug.augmenters import *

from auto_pose.meshrenderer.scenerenderer import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render virtual views of cad model')
    parser.add_argument('-o', '--output_path', required=True)
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-n', '--num', required=True, type=int)
    parser.add_argument('--width', default=640, type=int)
    parser.add_argument('--height', default=480, type=int)
    parser.add_argument('--show_images', default=False, type=bool)
    parser.add_argument('-s', '--scale',required=True, type=float, help='Factor to scale the model to mm. In case of a model in meters, use 1000')
    parser.add_argument('--nearplane', type=float, default=10)
    parser.add_argument('--farplane', type=float, default=5000)
    parser.add_argument('-v', '--vocpath', required=True, help='Path to the Pascal Voc folder (VOCdevkit)')
    parser.add_argument('-type', '--model_type', required=False, default = 'reconst', help='Path to the Pascal Voc folder (VOCdevkit)')
    parser.add_argument('--radius', default=800, type=int)
    parser.add_argument('--min_rot_views', default=1000, type=int)
    args = parser.parse_args()
    
    output_path = args.output_path

    models_cad_files = glob.glob(os.path.join(args.model,'*.ply'))
    obj_ids = [int(file.split('_')[-1].split('.')[0]) for file in models_cad_files]
    # print obj_ids
    print ("***************", models_cad_files)
    # obj_ids = [1]
    # models_cad_files = [file for file in models_cad_files]
    # print models_cad_files
    # exit()
    
    vertex_tmp_store_folder = '.'
    model_type = args.model_type

    K = np.array([(args.width+args.height)/2., 0, args.width/2, 0, (args.width+args.height)/2., args.height/2, 0, 0, 1]).reshape(3,3)
    # K = np.array([1075.65, 0, args.width/2, 0, 1073.90, args.height/2, 0, 0, 1]).reshape(3,3)


    vocdevkit_path = args.vocpath
    min_num_objects_per_scene = 1
    max_num_objects_per_scene = 9
    near_plane = args.nearplane
    far_plane = args.farplane
    min_n_views = args.min_rot_views
    radius = args.radius

    augmenters = Sequential([
        Sometimes(0.1, GaussianBlur(0.5)),
        Sometimes(0.1, GaussianBlur(1.0)),
        Sometimes(0.4, Add((-25, 20), per_channel=0.5)),
        #Sometimes(0.3, Invert(0.2, per_channel=True)),
        Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
        Sometimes(0.5, Multiply((0.6, 1.4))),
        Sometimes(0.5, ContrastNormalization((0.5, 2.2), per_channel=0.3)),
        #Sometimes(0.2, CoarseDropout( p=0.1, size_px = 10, size_percent=0.001) )
    ], random_order=True)

    renderer = SceneRenderer(
        models_cad_files, 
        vertex_tmp_store_folder, 
        args.scale,
        args.width,
        args.height,
        K,
        augmenters,
        vocdevkit_path,
        min_num_objects_per_scene,
        max_num_objects_per_scene,
        near_plane,
        far_plane,
        min_n_views,
        radius,
        obj_ids=obj_ids,
        model_type=model_type
    )
    
    max_round = args.num
    
    #print max_round
    
    widgets = ['Processing: ', pb.Percentage(), ' ', pb.Bar(marker='#',left='[',right=']'),' ', pb.ETA()]
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    pbar = pb.ProgressBar(widgets=widgets, maxval=max_round).start()


    for round in range(max_round):
        filename = str('image_' + str(round))
        
        bgr, obj_info = renderer.render()
        
        cv2.imwrite(os.path.join(output_path, filename + '.png'),bgr)
        # cv2.imshow('bgr',bgr)
        write_xml(obj_info, args.width, args.height, obj_info, '', output_path, filename)
        
        if args.show_images:
            for o in obj_info:
                xmin, ymin, xmax, ymax = o['bb']
                bgr = cv2.rectangle(bgr,(xmin, ymin),(xmax, ymax),(0,255,0), 2)
            cv2.imshow('Scene Renderer', bgr)
            k = cv2.waitKey(0)
            if k == 27: # ESC
                break
        pbar.update(round)
