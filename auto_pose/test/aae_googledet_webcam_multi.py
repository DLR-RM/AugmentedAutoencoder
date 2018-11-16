import cv2
import tensorflow as tf
import numpy as np
import glob
import imageio
import os
import configparser
import re
import time

from auto_pose.meshrenderer import meshrenderer_phong
from auto_pose.ae.pysixd_stuff import misc

from auto_pose.ae import factory, utils

import argparse

import copy
import yaml
import tarfile
import six.moves.urllib as urllib
from tensorflow.core.framework import graph_pb2

# Protobuf Compilation (once necessary)
#os.system('protoc object_detection/protos/*.proto --python_out=.')

from auto_pose.test.googledet_utils import label_map_util
from auto_pose.test.googledet_utils.helper import FPS2, WebcamVideoStream, SessionWorker

import time




# helper function for split model
def _node_name(n):
    if n.startswith("^"):
        return n[1:]
    else:
        return n.split(":")[0]

# Load a (frozen) Tensorflow model into memory.
def load_frozenmodel():
    print('> Loading frozen model into memory')
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
    config.gpu_options.allow_growth=allow_memory_growth
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 ###Jetson only
    if not split_model:
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        return detection_graph, None, None

    else:
        # load a frozen Model and split it into GPU and CPU graphs
        # Hardcoded for ssd_mobilenet
        input_graph = tf.Graph()
        with tf.Session(graph=input_graph,config=config):
            if ssd_shape == 600:
                shape = 7326
                print 'ssd_shape = 600 :('
                exit()
            else:
                shape = 1917
            score = tf.placeholder(tf.float32, shape=(None, shape, num_classes), name="Postprocessor/convert_scores")
            expand = tf.placeholder(tf.float32, shape=(None, shape, 1, 4), name="Postprocessor/ExpandDims_1")
            for node in input_graph.as_graph_def().node:
                if node.name == "Postprocessor/convert_scores":
                    score_def = node
                if node.name == "Postprocessor/ExpandDims_1":
                    expand_def = node

        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                dest_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']
    
                edges = {}
                name_to_node_map = {}
                node_seq = {}
                seq = 0
                for node in od_graph_def.node:
                    n = _node_name(node.name)
                    name_to_node_map[n] = node
                    edges[n] = [_node_name(x) for x in node.input]
                    node_seq[n] = seq
                    seq += 1
                for d in dest_nodes:
                    assert d in name_to_node_map, "%s is not in graph" % d
    
                nodes_to_keep = set()
                next_to_visit = dest_nodes[:]
                
                while next_to_visit:
                    n = next_to_visit[0]
                    del next_to_visit[0]
                    if n in nodes_to_keep: continue
                    nodes_to_keep.add(n)
                    next_to_visit += edges[n]
    
                nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])
                nodes_to_remove = set()
                
                for n in node_seq:
                    if n in nodes_to_keep_list: continue
                    nodes_to_remove.add(n)
                nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])
    
                keep = graph_pb2.GraphDef()
                for n in nodes_to_keep_list:
                    keep.node.extend([copy.deepcopy(name_to_node_map[n])])
    
                remove = graph_pb2.GraphDef()
                remove.node.extend([score_def])
                remove.node.extend([expand_def])
                for n in nodes_to_remove_list:
                    remove.node.extend([copy.deepcopy(name_to_node_map[n])])
    
                with tf.device('/gpu:0'):
                    tf.import_graph_def(keep, name='')
                with tf.device('/cpu:0'):
                    tf.import_graph_def(remove, name='')

        return detection_graph, score, expand


def load_labelmap():
    print('> Loading label map')
    label_map = label_map_util.load_labelmap(label_path)
    print label_map
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    print categories
    category_index = label_map_util.create_category_index(categories)
    print category_index
    return category_index


def detection(detection_graph, category_index, score, expand):
    print("> Building Graph")
    print category_index
    # Session Config: allow seperate GPU/CPU adressing and limit memory allocation
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=log_device)
    config.gpu_options.allow_growth=allow_memory_growth
    config.gpu_options.per_process_gpu_memory_fraction = 0.4  ###Jetson only
    cur_frames = 0
    with detection_graph.as_default():
        #run_meta = tf.RunMetadata()
        with tf.Session(graph=detection_graph,config=config) as sess:
            # Define Input and Ouput tensors
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            if split_model:
                score_out = detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
                expand_out = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
                score_in = detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
                expand_in = detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
                # Threading
                gpu_worker = SessionWorker("GPU",detection_graph,config)
                cpu_worker = SessionWorker("CPU",detection_graph,config)
                gpu_opts = [score_out, expand_out]
                cpu_opts = [detection_boxes, detection_scores, detection_classes, num_detections]
                gpu_counter = 0
                cpu_counter = 0

            for i,experiment_name in enumerate(arguments.experiment_names):

                full_name = experiment_name.split('/')
                experiment_name = full_name.pop()
                experiment_group = full_name.pop() if len(full_name) > 0 else ''

                train_cfg_file_path = utils.get_config_file_path(workspace_path, experiment_name, experiment_group)
                train_args = configparser.ConfigParser()
                train_args.read(train_cfg_file_path)  
                h_train, w_train, c = train_args.getint('Dataset','H'),train_args.getint('Dataset','W'), train_args.getint('Dataset','C')
                model_paths.append(train_args.get('Paths','MODEL_PATH'))
                all_train_args.append(train_args)
                  
                log_dir = utils.get_log_dir(workspace_path,experiment_name,experiment_group)
                ckpt_dir = utils.get_checkpoint_dir(log_dir)

                all_codebooks.append(factory.build_codebook_from_name(experiment_name, experiment_group, return_dataset=False))
                factory.restore_checkpoint(sess, tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=experiment_name)), ckpt_dir)
            
            
            #opts = tf.profiler.ProfileOptionBuilder.float_operation()    
            #flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
            #exit()
            # i_class_mapping = {v: k for k, v in class_i_mapping.iteritems()}
            renderer = meshrenderer_phong.Renderer(
                model_paths, 
                1
            )

            # Start Video Stream and FPS calculation
            fps = FPS2(fps_interval).start()
            video_stream = WebcamVideoStream(video_input,width,height).start()
            cur_frames = 0
            print("> Press 'q' to Exit, 'a' to start auto_pose")
            print('> Starting Detection')
            while video_stream.isActive():
                # actual Detection
                if split_model:
                    # split model in seperate gpu and cpu session threads
                    if gpu_worker.is_sess_empty():
                        # read video frame, expand dimensions and convert to rgb
                        image = video_stream.read()
                        image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                        # put new queue
                        gpu_feeds = {image_tensor: image_expanded}
                        if visualize:
                            gpu_extras = image # for visualization frame
                        else:
                            gpu_extras = None
                        gpu_worker.put_sess_queue(gpu_opts,gpu_feeds,gpu_extras)

                    g = gpu_worker.get_result_queue()
                    if g is None:
                        # gpu thread has no output queue. ok skip, let's check cpu thread.
                        gpu_counter += 1
                    else:
                        # gpu thread has output queue.
                        gpu_counter = 0
                        score,expand,image = g["results"][0],g["results"][1],g["extras"]

                        if cpu_worker.is_sess_empty():
                            # When cpu thread has no next queue, put new queue.
                            # else, drop gpu queue.
                            cpu_feeds = {score_in: score, expand_in: expand}
                            cpu_extras = image
                            cpu_worker.put_sess_queue(cpu_opts,cpu_feeds,cpu_extras)

                    c = cpu_worker.get_result_queue()
                    if c is None:
                        # cpu thread has no output queue. ok, nothing to do. continue
                        cpu_counter += 1
                        time.sleep(0.005)
                        continue # If CPU RESULT has not been set yet, no fps update
                    else:
                        cpu_counter = 0
                        boxes, scores, classes, num, image = c["results"][0],c["results"][1],c["results"][2],c["results"][3],c["extras"]
                else:
                    # default session
                    image = video_stream.read()
                    image_expanded = np.expand_dims(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), axis=0)
                    boxes, scores, classes, num = sess.run(
                            [detection_boxes, detection_scores, detection_classes, num_detections],
                            feed_dict={image_tensor: image_expanded})

                # Visualization of the results of a detection.


                H, W = image.shape[:2]

                img_crops = []
                det_bbs = []
                det_classes = []
                det_scores = []                

                det_aae_bbs = []
                det_aae_objects_k = []
                #print vis_img.shape
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                classes = np.squeeze(classes).astype(np.int32)
                

                highest_class_score = {clas:0.0 for clas in classes}
                for box,score, clas in zip(boxes,scores, classes):
                    if score > det_th and score > highest_class_score[clas]:

                        highest_class_score[clas] = score
                        ymin, xmin, ymax, xmax = (np.array(box)*np.array([height,width,height,width])).astype(np.int32)

                        h, w = (ymax-ymin,xmax-xmin)
                        det_bbs.append([xmin,ymin,w,h])
                        det_classes.append(clas)
                        det_scores.append(score)
                        if clas in clas_k_map:
                            det_aae_bbs.append([xmin,ymin,w,h])

                            det_aae_objects_k.append(clas_k_map[clas])

                            size = int(np.maximum(h, w) * train_args.getfloat('Dataset','PAD_FACTOR'))
                            cx = xmin + (xmax - xmin)/2
                            cy = ymin + (ymax - ymin)/2

                            left = np.maximum(cx-size/2, 0)
                            top = np.maximum(cy-size/2, 0)

                            img_crop = image[top:cy+size/2,left:cx+size/2]
                            img_crop = cv2.resize(img_crop, (h_train,w_train))

                            img_crop = img_crop / 255.
                            img_crops.append(img_crop)



                if len(det_aae_bbs) > 0:

                    Rs = []
                    ts = []
                    for k,bb,img_crop in zip(det_aae_objects_k,det_aae_bbs,img_crops):
                        R, t = all_codebooks[k].auto_pose6d(sess, img_crop, bb, K_test, 1, all_train_args[k], upright=False)
                        Rs.append(R.squeeze())
                        ts.append(t.squeeze())

                    Rs = np.array(Rs)
                    ts = np.array(ts)                                    
                    
                    bgr_y,_,_ = renderer.render_many( 
                        obj_ids=np.array(det_aae_objects_k).astype(np.int32),
                        W=width/arguments.down,
                        H=height/arguments.down,
                        K=K_down, 
                        Rs=Rs, 
                        ts=ts,
                        near=1.,
                        far=10000.,
                        random_light=False,
                        # calc_bbs=False,
                        # depth=False
                    )
                    
                
                    bgr_y = cv2.resize(bgr_y,(width,height))

                    
                    g_y = np.zeros_like(bgr_y)
                    g_y[:,:,1]= bgr_y[:,:,1]    
                    im_bg = cv2.bitwise_and(image,image,mask=(g_y[:,:,1]==0).astype(np.uint8))                 
                    image = cv2.addWeighted(im_bg,1,g_y,1,0)
                    
                for bb,score,clas in zip(det_bbs,det_scores,det_classes):
                    xmin,ymin,xmax,ymax = bb[0],bb[1],bb[2]+bb[0],bb[1]+bb[3]
                    cv2.putText(image, '%s : %1.3f' % (category_index[clas]['name'],score), (xmin, ymax+20), cv2.FONT_ITALIC, .5, color_dict[clas-1], 2)
                    cv2.rectangle(image, (xmin,ymin),(xmax,ymax), color_dict[clas-1], 2)


                if vis_text:
                    cv2.putText(image,"fps: {}".format(fps.fps_local()), (10,30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)
                cv2.imshow('object_detection', image)
                # Exit Option
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break

                fps.update()

    # End everything
    if split_model:
        gpu_worker.stop()
        cpu_worker.stop()
    fps.stop()
    video_stream.stop()
    cv2.destroyAllWindows()
    print('> [INFO] elapsed time (total): {:.2f}'.format(fps.elapsed()))
    print('> [INFO] approx. FPS: {:.2f}'.format(fps.fps()))





## LOAD CONFIG PARAMS ##
try:
    with open("googledet_utils/googledet_config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
except:
    print 'no config file found'
    exit()


video_input         = cfg['video_input']
visualize           = cfg['visualize']
vis_text            = cfg['vis_text']
width               = cfg['width']
height              = cfg['height']
fps_interval        = cfg['fps_interval']
allow_memory_growth = cfg['allow_memory_growth']
det_th              = cfg['det_th']
model_path          = cfg['model_path']
label_path          = cfg['label_path']
num_classes         = cfg['num_classes']
split_model         = cfg['split_model']
log_device          = cfg['log_device']
ssd_shape           = cfg['ssd_shape']
K_test              = cfg['K_test']

parser = argparse.ArgumentParser()
parser.add_argument("experiment_names", nargs='+',type=str)
parser.add_argument('-down', default=1, type=int)
parser.add_argument("-s", action='store_true', default=False)

# parser.add_argument("-gt_bb", action='store_true', default=False)
arguments = parser.parse_args()

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
if workspace_path == None:
    print 'Please define a workspace path:\n'
    print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
    exit(-1)


all_codebooks = []
all_train_args = []
model_paths = []

K_test =  np.array(K_test).reshape(3,3)
K_down = K_test.copy()
if arguments.down > 1:
    K_down[:2,:] = K_down[:2,:] / arguments.down

aae_list = [full_name.pop() for full_name in [experiment_name.split('/') for experiment_name in arguments.experiment_names]]
color_dict = [(0,255,0),(0,0,255),(255,0,0),(255,255,0)] * 10

graph, score, expand = load_frozenmodel()
category = load_labelmap()

clas_k_map = {}
for _,val in category.iteritems():
    if val['name'] in aae_list:
        clas_k_map[int(val['id'])] = aae_list.index(val['name'])

detection(graph, category, score, expand)

