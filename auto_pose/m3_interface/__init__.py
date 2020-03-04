from importlib import import_module

name_to_class = {
        'auto_pose' : ('ae_pose_estimator','AePoseEstimator')
        }

def get_available_estimators():
    return list(name_to_class.keys())

def get_estimator(name, config=None):

    le_module = import_module('.' + name_to_class[name][0], package='auto_pose.m3_interface')
    le_class = getattr(le_module, name_to_class[name][1])

    return le_class(config)

