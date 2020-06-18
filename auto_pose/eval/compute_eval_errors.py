from sixd_toolkit.tools import eval_calc_errors, eval_loc
import argparse
import os
import configparser


def get_log_dir(workspace_path, experiment_name, experiment_group=''):
    return os.path.join(
        workspace_path, 
        'experiments',
        experiment_group,
        experiment_name
    )
def get_config_file_path(workspace_path, experiment_name, experiment_group=''):
    return os.path.join(
        workspace_path, 
        'cfg',
        experiment_group,
        '{}.cfg'.format(experiment_name)
    )

def get_eval_config_file_path(workspace_path, eval_cfg='eval.cfg'):
    return os.path.join(
        workspace_path, 
        'cfg_eval',
        eval_cfg
    )

def get_eval_dir(log_dir, evaluation_name, data):
    return os.path.join(
        log_dir,
        'eval',
        evaluation_name,
        data
    )

os.environ['AE_WORKSPACE_PATH'] = '/net/rmc-lx0314/home_local/sund_ma/autoencoder_ws'

parser = argparse.ArgumentParser()

parser.add_argument('experiment_name')
parser.add_argument('evaluation_name')
parser.add_argument('--eval_cfg', default='eval.cfg', required=False)
arguments = parser.parse_args()
full_name = arguments.experiment_name.split('/')
experiment_name = full_name.pop()
experiment_group = full_name.pop() if len(full_name) > 0 else ''
evaluation_name = arguments.evaluation_name
eval_cfg = arguments.eval_cfg

workspace_path = os.environ.get('AE_WORKSPACE_PATH')
train_cfg_file_path = get_config_file_path(workspace_path, experiment_name, experiment_group)
eval_cfg_file_path = get_eval_config_file_path(workspace_path, eval_cfg=eval_cfg)

train_args = configparser.ConfigParser(inline_comment_prefixes="#")
eval_args = configparser.ConfigParser(inline_comment_prefixes="#")
train_args.read(train_cfg_file_path)
eval_args.read(eval_cfg_file_path)


dataset_name = eval_args.get('DATA','DATASET')
cam_type = eval_args.get('DATA','cam_type')
estimate_bbs = eval_args.getboolean('BBOXES', 'ESTIMATE_BBS')
icp = eval_args.getboolean('EVALUATION','ICP')
gt_trans = eval_args.getboolean('EVALUATION','gt_trans')
gt_masks = eval_args.getboolean('BBOXES','gt_masks')
estimate_masks = eval_args.getboolean('BBOXES','estimate_masks')
iterative_code_refinement = eval_args.getboolean('EVALUATION','iterative_code_refinement')


evaluation_name = evaluation_name + '_icp' if icp else evaluation_name
evaluation_name = evaluation_name + '_bbest' if estimate_bbs else evaluation_name
evaluation_name = evaluation_name + '_maskest' if estimate_masks else evaluation_name
evaluation_name = evaluation_name + '_gttrans' if gt_trans else evaluation_name
evaluation_name = evaluation_name + '_gtmasks' if gt_masks else evaluation_name
evaluation_name = evaluation_name + '_refined' if iterative_code_refinement else evaluation_name

data = dataset_name + '_' + cam_type if len(cam_type) > 0 else dataset_name

log_dir = get_log_dir(workspace_path, experiment_name, experiment_group)
eval_dir = get_eval_dir(log_dir, evaluation_name, data)


eval_calc_errors.eval_calc_errors(eval_args, eval_dir)
eval_loc.match_and_eval_performance_scores(eval_args, eval_dir)

# report = latex_report.Report(eval_dir,log_dir)
# report.write_configuration(train_cfg_file_path,eval_cfg_file_path)
# report.merge_all_tex_files()
# report.include_all_figures()
# report.save(open_pdf=True)
