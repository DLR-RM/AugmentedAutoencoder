# -*- coding: utf-8 -*-
import os
import glob
import shutil

from . import utils as u

def main():

	workspace_path = os.environ.get('AE_WORKSPACE_PATH')

	if workspace_path == None:
	    print('Please define a workspace path:\n')
	    print('export AE_WORKSPACE_PATH=/path/to/workspace\n')
	    exit(-1)


	if len(os.listdir(workspace_path)) > 0:
	    print('\n[Error] Workspace folder needs to be empty.\n')
	    exit(-1)

	cfg_path = os.path.join(workspace_path, 'cfg' )
	eval_cfg_path = os.path.join(workspace_path, 'cfg_eval' )
	experiments_path = os.path.join(workspace_path, 'experiments' )
	dataset_path = os.path.join(workspace_path, 'tmp_datasets' )


	this_dir = os.path.dirname(os.path.abspath(__file__))
	
	if not os.path.exists(cfg_path):
		cfg_template_path = os.path.join(this_dir, 'cfg')
		shutil.copytree(cfg_template_path, cfg_path)
	if not os.path.exists(eval_cfg_path):
		eval_cfg_template_path = os.path.join(this_dir, 'cfg_eval')
		shutil.copytree(eval_cfg_template_path, eval_cfg_path)
		
	if not os.path.exists(experiments_path):
		os.makedirs(experiments_path)

	if not os.path.exists(dataset_path):
		os.makedirs(dataset_path)
