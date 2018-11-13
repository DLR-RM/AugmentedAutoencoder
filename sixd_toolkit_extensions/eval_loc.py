# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Calculates performance scores for the 6D localization task.

# For evaluation of the SIXD Challenge 2017 task (6D localization of a single
# instance of a single object), use these parameters:
# n_top = 1
# visib_gt_min = 0.1
# error_type = 'vsd'
# vsd_cost = 'step'
# vsd_delta = 15
# vsd_tau = 20
# error_thresh['vsd'] = 0.3

import os
from os.path import join as pjoin
import sys
from collections import defaultdict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sixd_toolkit.pysixd import inout, pose_matching
from sixd_toolkit.params.dataset_params import get_dataset_params


def match_poses(gts, gt_stats, errs, scene_id, visib_gt_min,
                error_threshs, n_top):

    # Organize the errors by image id and object id (for faster query)
    errs_org = {}
    for e in errs:
        errs_org.setdefault(e['im_id'], {}). \
            setdefault(e['obj_id'], []).append(e)

    # Matching
    matches = []
    for im_id, gts_im in gts.items():
        matches_im = []
        for gt_id, gt in enumerate(gts_im):
            valid = gt_stats[im_id][gt_id]['visib_fract'] >= visib_gt_min
            matches_im.append({
                'scene_id': scene_id,
                'im_id': im_id,
                'obj_id': gt['obj_id'],
                'gt_id': gt_id,
                'est_id': -1,
                'score': -1,
                'error': -1,
                'error_norm': -1,
                # 'stats': gt_stats[im_id][gt_id],
                'valid': int(valid)
            })

        # Mask of valid GT poses (i.e. GT poses with sufficient visibility)
        gt_valid_mask = [m['valid'] for m in matches_im]

        # Treat estimates of each object separately
        im_obj_ids = set([gt['obj_id'] for gt in gts_im])
        for obj_id in im_obj_ids:
            if im_id in errs_org.keys() and obj_id in errs_org[im_id].keys():

                # Greedily match the estimated poses to the ground truth
                # poses in the order of decreasing score
                errs_im_obj = errs_org[im_id][obj_id]
                ms = pose_matching.match_poses(errs_im_obj,
                                               error_threshs[obj_id],
                                               n_top, gt_valid_mask)
                for m in ms:
                    g = matches_im[m['gt_id']]
                    g['est_id'] = m['est_id']
                    g['score'] = m['score']
                    g['error'] = m['error']
                    g['error_norm'] = m['error_norm']

        matches += matches_im

    return matches


def calc_recall(tp_count, targets_count):
    if targets_count == 0:
        return 0.0
    else:
        return tp_count / float(targets_count)


def calc_scores(scene_ids, obj_ids, matches, n_top, do_print=True):

    # Count the number of visible object instances in each image
    insts = {i: {j: defaultdict(lambda: 0) for j in scene_ids} for i in obj_ids}
    for m in matches:
        if m['valid']:
            insts[m['obj_id']][m['scene_id']][m['im_id']] += 1

    # Count the number of targets = object instances to be found
    # (e.g. for 6D localization of a single instance of a single object, there
    # is either zero or one target in each image - there is just one even if
    # there are more instances of the object of interest)
    tars = 0 # Total number of targets
    obj_tars = {i: 0 for i in obj_ids} # Targets per object
    scene_tars = {i: 0 for i in scene_ids} # Targets per scene
    for obj_id, obj_insts in insts.items():
        for scene_id, scene_insts in obj_insts.items():

            # Count the number of targets in the current scene
            if n_top > 0:
                count = sum(np.minimum(n_top, scene_insts.values()))
            else:  # 0 = all estimates, -1 = given by the number of GT poses
                count = sum(scene_insts.values())

            tars += count
            obj_tars[obj_id] += count
            scene_tars[scene_id] += count

    # Count the number of true positives
    tps = 0 # Total number of true positives
    obj_tps = {i: 0 for i in obj_ids} # True positives per object
    scene_tps = {i: 0 for i in scene_ids} # True positives per scene
    for m in matches:
        if m['valid'] and m['est_id'] != -1:
            tps += 1
            obj_tps[m['obj_id']] += 1
            scene_tps[m['scene_id']] += 1

    # Total recall
    total_recall = calc_recall(tps, tars)

    # Recall per object
    obj_recalls = {}
    for i in obj_ids:
        obj_recalls[i] = calc_recall(obj_tps[i], obj_tars[i])
    mean_obj_recall = float(np.mean(obj_recalls.values()).squeeze())

    # Recall per scene
    scene_recalls = {}
    for i in scene_ids:
        scene_recalls[i] = calc_recall(scene_tps[i], scene_tars[i])
    mean_scene_recall = float(np.mean(scene_recalls.values()).squeeze())

    scores = {
        'total_recall': total_recall,
        'obj_recalls': obj_recalls,
        'mean_obj_recall': mean_obj_recall,
        'scene_recalls': scene_recalls,
        'mean_scene_recall': mean_scene_recall,
        'gt_count': len(matches),
        'targets_count': tars,
        'tp_count': tps,
    }

    if do_print:
        obj_recalls_str = ', '.join(
            ['{}: {:.3f}'.format(i, s)
             for i, s in scores['obj_recalls'].items()])

        scene_recalls_str = ', '.join(
            ['{}: {:.3f}'.format(i, s)
             for i, s in scores['scene_recalls'].items()])

        print('')
        print('GT count:           {:d}'.format(scores['gt_count']))
        print('Target count:       {:d}'.format(scores['targets_count']))
        print('TP count:           {:d}'.format(scores['tp_count']))
        print('Total recall:       {:.4f}'.format(scores['total_recall']))
        print('Mean object recall: {:.4f}'.format(scores['mean_obj_recall']))
        print('Mean scene recall:  {:.4f}'.format(scores['mean_scene_recall']))
        print('Object recalls:\n{}'.format(obj_recalls_str))
        print('Scene recalls:\n{}'.format(scene_recalls_str))
        print('')

    return scores


# def main():
def match_and_eval_performance_scores(eval_args, eval_dir):
    # Paths to pose errors (calculated using eval_calc_errors.py)
    #---------------------------------------------------------------------------
    # error_bpath = '/path/to/eval/'
    # error_paths = [
    #     pjoin(error_bpath, 'hodan-iros15_hinterstoisser'),
    #     # pjoin(error_bpath, 'hodan-iros15_tless_primesense'),
    # ]

    #[METHOD]
    method = eval_args.get('METHOD','METHOD')

    #[DATA]
    dataset = eval_args.get('DATA','DATASET')
    test_type = eval_args.get('DATA','CAM_TYPE')
    

    #[METRIC]
    # Top N pose estimates (with the highest score) to be evaluated for each
    # object in each image
    n_top = eval_args.getint('EVALUATION','TOP_N_EVAL') # 0 = all estimates, -1 = given by the number of GT poses
    n_top_str = eval_args.getint('METRIC','TOP_N')
    # Pose error function
    error_types = eval(eval_args.get('METRIC','ERROR_TYPE')) # 'vsd', 'adi', 'add', 'cou', 're', 'te'
    error_thresh = eval(eval_args.get('METRIC','ERROR_THRES'))
    error_thresh_fact = eval(eval_args.get('METRIC','ERROR_THRESH_FACT'))
    # VSD parameters
    vsd_delta = eval_args.getint('METRIC','VSD_DELTA')
    vsd_tau = eval_args.getint('METRIC','VSD_TAU')
    vsd_cost = eval_args.get('METRIC','VSD_COST') # 'step', 'tlinear'

    idx_th = 0
    idx_thf = 0

    for error_type in error_types:

        # Error signature
        error_sign = 'error=' + error_type + '_ntop=' + str(n_top_str)
        if error_type == 'vsd':
            error_sign += '_delta={}_tau={}_cost={}'.format(
                vsd_delta, vsd_tau, vsd_cost)


        error_path = os.path.join(eval_dir, error_sign)
        # error_dir = 'error=vsd_ntop=1_delta=15_tau=20_cost=step'
        # Other paths
        #---------------------------------------------------------------------------
        # Mask of path to the input file with calculated errors
        errors_mpath = pjoin('{error_path}', 'errors_{scene_id:02d}.yml')

        # Mask of path to the output file with established matches and calculated scores
        matches_mpath = pjoin('{error_path}', 'matches_{eval_sign}.yml')
        scores_mpath = pjoin('{error_path}', 'scores_{eval_sign}.yml')

        # Parameters
        #---------------------------------------------------------------------------
        require_all_errors = False # Whether to break if some errors are missing
        visib_gt_min = 0.1 # Minimum visible surface fraction of valid GT pose
        visib_delta = 15 # [mm]

        # # Threshold of correctness
        # error_thresh = {
        #     'vsd': 0.3,
        #     'cou': 0.5,
        #     'te': 5.0, # [cm]
        #     're': 5.0 # [deg]
        # }

        # # Factor k; threshold of correctness = k * d, where d is the object diameter
        # error_thresh_fact = {
        #     'add': 0.1,
        #     'adi': 0.1
        # }

        # Evaluation
        #---------------------------------------------------------------------------


        # Evaluation signature
        if error_type in ['add', 'adi']:
            if type(error_thresh_fact[error_type]) is list:
                cur_thres_f = error_thresh_fact[error_type][idx_thf]
                idx_thf += 1
            else:
                cur_thres_f = error_thresh_fact[error_type]
            eval_sign = 'thf=' + str(cur_thres_f)
        else:
            if type(error_thresh[error_type]) is list:
                cur_thres = error_thresh[error_type][idx_th]
                idx_th += 1
            else:
                cur_thres = error_thresh[error_type]
            eval_sign = 'th=' + str(cur_thres)
        eval_sign += '_min-visib=' + str(visib_gt_min)

        print('--- Processing: {}, {}, {}'.format(method, dataset, error_type))

        # Load dataset parameters
        dp = get_dataset_params(dataset, test_type=test_type)
        obj_ids = range(1, dp['obj_count'] + 1)
        scene_ids = range(1, dp['scene_count'] + 1)

        # Set threshold of correctness (might be different for each object)
        error_threshs = {}
        if error_type in ['add', 'adi']:
            # Relative to object diameter
            models_info = inout.load_yaml(dp['models_info_path'])
            for obj_id in obj_ids:
                obj_diameter = models_info[obj_id]['diameter']
                error_threshs[obj_id] = cur_thres_f * obj_diameter
        else:
            # The same threshold for all objects
            for obj_id in obj_ids:
                error_threshs[obj_id] = cur_thres

        # Go through the test scenes and match estimated poses to GT poses
        #-----------------------------------------------------------------------
        matches = []  # Stores info about the matching estimate for each GT
        for scene_id in scene_ids:

            # Load GT poses
            gts = inout.load_gt(dp['scene_gt_mpath'].format(scene_id))

            # Load statistics (e.g. visibility fraction) of the GT poses
            gt_stats_path = dp['scene_gt_stats_mpath'].format(scene_id,
                                                              visib_delta)
            gt_stats = inout.load_yaml(gt_stats_path)

            # Load pre-calculated errors of the pose estimates
            scene_errs_path = errors_mpath.format(
                error_path=error_path, scene_id=scene_id)

            if os.path.isfile(scene_errs_path):
                errs = inout.load_errors(scene_errs_path)

                matches += match_poses(gts, gt_stats, errs, scene_id,
                                       visib_gt_min, error_threshs, n_top)

            elif require_all_errors:
                raise IOError(
                    '{} is missing, but errors for all scenes are required'
                    ' (require_all_results = True).'.format(scene_errs_path)
                )

        # Calculate the performance scores
        #-----------------------------------------------------------------------
        # Split the dataset of Hinterstoisser to the original LINEMOD dataset
        # and the Occlusion dataset by TUD (i.e. the extended GT for scene #2)
        if dataset == 'hinterstoisser':

            print('-- LINEMOD dataset')
            eval_sign_lm = 'linemod_' + eval_sign
            matches_lm = [m for m in matches if m['scene_id'] == m['obj_id']]
            scores_lm = calc_scores(scene_ids, obj_ids, matches_lm, n_top)

            # Save scores
            scores_lm_path = scores_mpath.format(
                error_path=error_path, eval_sign=eval_sign_lm)
            inout.save_yaml(scores_lm_path, scores_lm)

            # Save matches
            matches_path = matches_mpath.format(
                error_path=error_path, eval_sign=eval_sign_lm)
            inout.save_yaml(matches_path, matches_lm)

            print('-- Occlusion dataset')
            eval_sign_occ = 'occlusion_' + eval_sign
            matches_occ = [m for m in matches if m['scene_id'] == 2]
            scene_ids_occ = [2]
            obj_ids_occ = [1, 2, 5, 6, 8, 9, 10, 11, 12]
            scores_occ = calc_scores(scene_ids_occ, obj_ids_occ,
                                     matches_occ, n_top)
            # Save scores
            scores_occ_path = scores_mpath.format(
                error_path=error_path, eval_sign=eval_sign_occ)
            inout.save_yaml(scores_occ_path, scores_occ)

            # Save matches
            matches_path = matches_mpath.format(
                error_path=error_path, eval_sign=eval_sign_occ)
            inout.save_yaml(matches_path, matches_occ)
        else:
            scores = calc_scores(scene_ids, obj_ids, matches, n_top)

            # Save scores
            scores_path = scores_mpath.format(
                error_path=error_path, eval_sign=eval_sign)
            inout.save_yaml(scores_path, scores)

            # Save matches
            matches_path = matches_mpath.format(
                error_path=error_path, eval_sign=eval_sign)
            inout.save_yaml(matches_path, matches)

    print('Done.')


if __name__ == '__main__':
    main()
