# -*- coding: utf-8 -*-
import os
import glob
import shutil

import utils as u

def main():

	workspace_path = os.environ.get('AE_WORKSPACE_PATH')

	if workspace_path == None:
	    print 'Please define a workspace path:\n'
	    print 'export AE_WORKSPACE_PATH=/path/to/workspace\n'
	    exit(-1)


	if len(os.listdir(workspace_path)) > 0:
	    print '\n[Error] Workspace folder needs to be empty.\n'
	    exit(-1)

	cfg_path = os.path.join(workspace_path, 'cfg' )
	eval_cfg_path = os.path.join(workspace_path, 'cfg_eval' )
	experiments_path = os.path.join(workspace_path, 'experiments' )

	if not os.path.exists(cfg_path):
		this_dir = os.path.dirname(os.path.abspath(__file__))
		cfg_template_path = os.path.join(this_dir, 'cfg')
		eval_cfg_path = os.path.join(this_dir, 'cfg_eval')
		shutil.copytree(cfg_template_path, cfg_path)
		shutil.copytree(eval_cfg_path, eval_cfg_path)
		
	if not os.path.exists(experiments_path):
		os.makedirs(experiments_path)

