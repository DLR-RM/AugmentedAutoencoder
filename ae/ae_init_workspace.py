# -*- coding: utf-8 -*-
import os
import glob
import shutil

import utils as u

def main():
	current_dir = os.getcwd()

	if len(os.listdir(current_dir)) > 0:
	    print '\n[Error] Workspace folder needs to be empty.\n'
	    exit(-1)

	cfg_path = os.path.join(current_dir, 'cfg' )
	eval_cfg_path = os.path.join(current_dir, 'cfg_eval' )
	experiments_path = os.path.join(current_dir, 'experiments' )
	setup_bash = os.path.join(current_dir, 'setup.bash' )

	if not os.path.exists(cfg_path):
		this_dir = os.path.dirname(os.path.abspath(__file__))
		cfg_template_path = os.path.join(this_dir, 'cfg')
		eval_cfg_path = os.path.join(this_dir, 'cfg_eval')
		shutil.copytree(cfg_template_path, cfg_path)
		shutil.copytree(eval_cfg_path, eval_cfg_path)
		

	if not os.path.exists(experiments_path):
		os.makedirs(experiments_path)

	with open(setup_bash, 'w') as f:
		f.write('export AE_WORKSPACE_PATH={}\n'.format(current_dir))
