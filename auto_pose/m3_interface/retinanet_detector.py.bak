from m3vision.interfaces.detector_bb import BoundingBoxDetector
from m3vision.interfaces.detector_bb import BoundingBox
from m3vision.interfaces.detector_bb import preprocess_image_colors_type

import numpy as np
import tensorflow as tf
import keras

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr,\
        preprocess_image, resize_image

import keras.models
from keras_retinanet.models.retinanet import retinanet_bbox
from keras_retinanet.models.retinanet import AnchorParameters
from keras_retinanet.models.retinanet import __build_anchors as build_anchors
from keras_retinanet.models import backbone
from keras_retinanet import layers
from keras_retinanet.utils.image import preprocess_image\
        as krn_u_preprocess_image

class RetinaNetDetector(BoundingBoxDetector):
    def __init__(self, modelpath, config):
        super(RetinaNetDetector, self).__init__()
        self._internal_type = np.float32
        self._internal_format = 'bgr'

        if config == None:
            self._params = {
                    'gpu_memory_fraction' : 0.3,
                    'nms_threshold' : 0.5,
                    'det_threshold' : 0.45
                    }
            self._modelpath = modelpath
        else:
            self._params = self.get_params(config)
            if self._params.has_key('model_path'):
                self._modelpath = self._params['model_path']
            else:
                self._modelpath = modelpath

        print((self._params))
        gpu_options = tf.GPUOptions(
                allow_growth=True,
                per_process_gpu_memory_fraction = \
                        self._params['gpu_memory_fraction']
                )

        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

        keras.backend.tensorflow_backend.set_session(sess)

        self._det = models.load_model(self._modelpath, backbone_name='resnet50')
        #
        # self._det = self._load_model_with_nms()

        print('LOADED')

    def process_raw(self, image_in):
        image = preprocess_image(image_in)
        image, scale = resize_image(image_in)

        boxes, scores, labels = \
                self._det.predict_on_batch(np.expand_dims(image, axis=0))

        boxes /= scale
        # Filter scores < 0
        valid_dets = np.where(scores[0] >= 0)[0]
        boxes  = boxes[0][valid_dets]
        scores = scores[0][valid_dets]
        labels = labels[0][valid_dets]

        rel_boxes = []

        boxes /= np.array(
                [
                    image_in.shape[1],
                    image_in.shape[0],
                    image_in.shape[1],
                    image_in.shape[0]
                    ]
                )

        dets = []
        for det in zip(labels, scores, boxes):
            bbox = BoundingBox()
            bbox.xmin = det[2][0]
            bbox.ymin = det[2][1]
            bbox.xmax = det[2][2]
            bbox.ymax = det[2][3]
            bbox.classes = {str(det[0] + 1) : det[1]}
            dets.append(bbox)

        return dets

    def preprocess_image(self, image, color_format, dtype):
        # This only does rgb -> bgr if necessary.
        # type_out is intentionally == dtype since the retinanet preprocessing
        # function converts to float32 already.
        ret_img, _ = preprocess_image_colors_type(
                image, color_format, dtype, self._internal_format, dtype)
        ret_img = krn_u_preprocess_image(ret_img)

        return ret_img, True

    def _load_model_with_nms(self):
        """ This is mostly copied fomr retinanet.py """

        backbone_name = 'resnet50'
        model = keras.models.load_model(
                self._modelpath,
                custom_objects=backbone(backbone_name).custom_objects
                )

        # compute the anchors
        features = [model.get_layer(name).output
                for name in ['P3', 'P4', 'P5', 'P6', 'P7']]
        anchors  = build_anchors(AnchorParameters.default, features)

        # we expect the anchors, regression and classification values as first
        # output
        regression     = model.outputs[0]
        classification = model.outputs[1]

        # "other" can be any additional output from custom submodels,
        # by default this will be []
        other = model.outputs[2:]

        # apply predicted regression to anchors
        boxes = layers.RegressBoxes(name='boxes')([anchors, regression])
        boxes = layers.ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

        # filter detections (apply NMS / score threshold / select top-k)
        detections = layers.FilterDetections(
                nms=True,
                name='filtered_detections',
                nms_threshold   = self._params['nms_threshold'],
                score_threshold = self._params['det_threshold'],
                max_detections  = self._params['max_detections']
                )([boxes, classification] + other)

        outputs = detections

        # construct the model
        return keras.models.Model(
                inputs=model.inputs, outputs=outputs, name='retinanet-bbox')
