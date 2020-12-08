import cv2
from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
import sys
import yaml


class Roi3D:
    def __init__(
            self,
            shape='cube',
            pose=np.identity(4),
            scale=[1, 1, 1],
            is_world_coords=True
    ):
        self.__shape = shape
        self.__pose = pose
        self.__scale = scale
        self.__is_world_coords = is_world_coords

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, shape):
        assert shape in ['cube', 'sphere', 'cylinder']
        self.__shape = shape

    @property
    def pose(self):
        return self.__pose

    @pose.setter
    def pose(self, pose):
        assert pose.shape == (4, 4)
        self.__pose = pose

    @property
    def scale(self):
        return self.__scale

    @scale.setter
    def scale(self, scale):
        assert len(scale) == 3
        self.__scale = scale

    @property
    def is_world_coords(self):
        return self.__is_world_coords

    @is_world_coords.setter
    def is_world_coords(self, iwc):
        self.__is_world_coords = iwc


class PoseEstimate(object):
    def __init__(self, name='SLC', trafo=np.identity(4), quality=1.0):
        self.__name = name
        self.__trafo = trafo
        self.__quality = quality

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def trafo(self):
        return self.__trafo

    @trafo.setter
    def trafo(self, trafo):
        assert trafo.shape == (4, 4)
        self.__trafo = trafo

    @property
    def quality(self):
        return self.__quality

    @quality.setter
    def quality(self, quality):
        self.__quality = quality


class PoseEstInterface:
    __metaclass__ = ABCMeta

    def __init__(self, configpath, m3vision_cfg):
        pass

    @abstractmethod
    def set_parameter(self, string_name, string_val):
        pass

    def get_params(self, config):
        isstring = False
        if sys.version_info[0] == 3:
            if isinstance(config, str):
                isstring = True
        else:
            if isinstance(config, basestring):
                isstring = True

        if isstring:
            if '.yml' in config or '.yaml' in config:
                with open(config, 'r') as f:
                    params = yaml.load(f)
            else:
                import configparser
                params = configparser.ConfigParser(inline_comment_prefixes="#")
                params.read(config)
        else:
            params = config

        return params

    @abstractmethod
    def query_process_requirements(self):
        """ Returns description of resources needed to call self.process() """
        return ['color_img', 'depth_img', 'camK', 'camPose']

    @abstractmethod
    def query_image_format(self):
        """ Returns a dictionary.
        E.g: {'color_format':'rgb', 'data_type':np.float32} """
        return {
            'color_format': 'rgb',  # Or bgr
            'color_data_type': np.float32,  # Or np.uint8
            'depth_data_type': np.float32  # Or np.float64
        }

    @abstractmethod
    def process(
            self,
            bboxes=[],
            color_img=None,
            depth_img=None,
            camK=None,
            camPose=None,
            rois3ds=[]
    ):
        pass


# Example detector. Kind of an interface
# implemented in other detectors...


class BoundingBox:
    def __init__(
            self,
            xmin=0.0,
            ymin=0.0,
            xmax=1.0,
            ymax=1.0,
            classes={'SLC': 1.0}
    ):
        self.__xmin = xmin
        self.__ymin = ymin
        self.__xmax = xmax
        self.__ymax = ymax
        self.__classes = classes

    @property
    def xmin(self):
        return self.__xmin

    @xmin.setter
    def xmin(self, xmin):
        assert 0 <= xmin <= 1
        self.__xmin = xmin

    @property
    def ymin(self):
        return self.__ymin

    @ymin.setter
    def ymin(self, ymin):
        assert 0 <= ymin <= 1
        self.__ymin = ymin

    @property
    def xmax(self):
        return self.__xmax

    @xmax.setter
    def xmax(self, xmax):
        assert 0 <= xmax <= 1
        self.__xmax = xmax

    @property
    def ymax(self):
        return self.__ymax

    @ymax.setter
    def ymax(self, ymax):
        assert 0 <= ymax <= 1
        self.__ymax = ymax

    @property
    def classes(self):
        return self.__classes

    @classes.setter
    def classes(self, val):
        self.__classes = val


class BoundingBoxDetector(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._clip_bb = None

    @abstractmethod
    def process_raw(self, image):
        raise NotImplementedError

    @abstractmethod
    def preprocess_image(self, image, color_format_in, type_in):
        raise NotImplementedError

    def process(self, image):
        if self._clip_bb is not None:
            bb = self._clip_bb
            im_s = image.shape

            proc_image = image[
                int(im_s[0] * bb['ymin']): int(im_s[0] * bb['ymax']),
                int(im_s[1] * bb['xmin']): int(im_s[1] * bb['xmax']),
            ]
        else:
            # No clipping. Nothing to do here
            proc_image = image

        dets = self.process_raw(proc_image)

        if self._clip_bb is not None:
            bb = self._clip_bb
            im_s = image.shape

            xscale = bb['xmax'] - bb['xmin']
            yscale = bb['ymax'] - bb['ymin']

            for det in dets:
                det.ymin = bb['ymin'] + (yscale * det.ymin)
                det.xmin = bb['xmin'] + (xscale * det.xmin)
                det.ymax = bb['ymin'] + (yscale * det.ymax)
                det.xmax = bb['xmin'] + (xscale * det.xmax)

        return dets

    def get_params(self, config):
        isstring = False
        if sys.version_info[0] == 3:
            if isinstance(config, str):
                isstring = True
        else:
            if isinstance(config, basestring):
                isstring = True

        if isstring:
            if '.yml' in config or '.yaml' in config:
                with open(config, 'r') as f:
                    params = yaml.load(f)
            else:
                import configparser
                parser = configparser.ConfigParser(inline_comment_prefixes="#")
                parser.read(config)
                if parser.has_section('methods'):
                    det_type = parser.get('methods', 'object_detector')
                else:
                    det_type = parser.sections()[0]
                params = {}
                for k, v in parser[det_type].items():
                    try:
                        params[k] = eval(v)
                    except:
                        params[k] = v
                print 'heeeerrree!!'
                print params
        else:
            if isinstance(config, dict):
                params = config
            else:
                params = {}
                for k, v in config.items():
                    try:
                        params[k] = eval(v)
                    except:
                        params[k] = v
        return params

    def set_clip_bb(self, xmin, ymin, xmax, ymax):

        if (xmin >= xmax or
                ymin >= ymax or not
                0 <= xmin <= 1 or not
                0 <= ymin <= 1 or not
                0 <= xmax <= 1 or not
                0 <= ymax <= 1
            ):
            raise RuntimeError('Bounding box out of bounds!')
        self._clip_bb = {
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax
        }

    def visualize_clip_bb(self, image):
        if self._clip_bb is not None:
            int_clip = [
                int(self._clip_bb['xmin'] * image.shape[1]),
                int(self._clip_bb['ymin'] * image.shape[0]),
                int(self._clip_bb['xmax'] * image.shape[1]),
                int(self._clip_bb['ymax'] * image.shape[0])
            ]
            cv2.rectangle(
                image,
                (int_clip[0], int_clip[1]),
                (int_clip[2], int_clip[3]),
                [0, 0, 255],
                2
            )


def preprocess_image_colors_type(
        image,
        color_format_in,
        type_in,
        color_format_out,
        type_out
):
    changed = False

    if color_format_in == 'rgb' and color_format_out == 'bgr':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        changed = True
    elif color_format_in == 'bgr' and color_format_out == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        changed = True

    if type_in == np.float and type_out == np.uint8:
        image = (image * 255).astype(np.uint8)
        changed = True
    elif type_in == np.uint8 and type_out == np.float:
        image = image.astype(np.float) / 255
        changed = True

    return image, changed
