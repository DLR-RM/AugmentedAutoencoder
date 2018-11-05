import numpy as np
import pandas as pd
import os
import glob
import time
import argparse
import re

prolog =\
r'''
\documentclass[a4paper,table]{article}
\usepackage{graphicx}
\usepackage{float}
\usepackage{tikz}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{fancyhdr}
\usepackage[a4paper]{geometry}
\usepackage{hyperref}
\usepackage{pdflscape}
\usepackage{booktabs}
\usepackage{nameref}
\usepackage{xcolor}
\usepackage{adjustbox}
\usepackage{gensymb}
\usepackage[parfill]{parskip}
\usepackage[utf8]{inputenc}
\usepackage{pgfplots}

\usepackage{pifont}
\hypersetup{
    colorlinks=true, 
    linktoc=all,     
    linkcolor=blue,
}
\pagestyle{fancy}

\newcommand*\rot{\rotatebox{90}}
\newcommand*\OK{\ding{51}}

\begin{document}
\begin{center}
{\Huge Experimental Protocol}\\
\textbf{%s}\\
\textbf{%s}\\
\end{center}
'''


epilog=\
r'''
\end{document}
'''

from auto_pose.ae import utils as u#
from sixd_toolkit.pysixd import inout
import configparser
from auto_pose.eval import eval_utils


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('experiment_group')
	parser.add_argument('--eval_name', default='*', required=False)
	args = parser.parse_args()

	experiment_group = args.experiment_group
	eval_name = args.eval_name
	print eval_name

	workspace_path = os.environ.get('AE_WORKSPACE_PATH')

	exp_group_path = os.path.join(workspace_path, 'experiments', experiment_group)
	print exp_group_path
	error_score_files = glob.glob(os.path.join(exp_group_path, '*/eval',eval_name,'*/error*/scores*'))
	print error_score_files
	data_re = []
	data_auc_re = []
	data_auc_rerect = []
	data_te = []
	data_vsd = []
	data_cou = []
	data_add = []
	data_adi = []
	data_proj = []
	data_paper_vsd = {}
	data_paper_auc = {}
	latex_content = []

	for error_score_file in error_score_files:
		split_path = error_score_file.split('/')
		exp_name = split_path[-6]
		eval_name = split_path[-4]
		occl = 'occlusion' if 'occlusion' in error_score_file else ''
		test_data = split_path[-3]
		error_type = split_path[-2].split('_')[0].split('=')[1]
		print error_type
		topn = split_path[-2].split('=')[2].split('_')[0]
		error_thres = split_path[-1].split('=')[1].split('_')[0]
		
		eval_cfg_file_path = os.path.join(workspace_path, 'experiments', experiment_group, 
			exp_name, 'eval', eval_name, test_data, '*.cfg')
		eval_cfg_file_pathes = glob.glob(eval_cfg_file_path)
		if len(eval_cfg_file_pathes) == 0:
			continue
		else:
			eval_cfg_file_path = eval_cfg_file_pathes[0]

		eval_args = configparser.ConfigParser()
		eval_args.read(eval_cfg_file_path)
		print eval_cfg_file_path
		estimate_bbs = eval_args.getboolean('BBOXES', 'ESTIMATE_BBS')
		try:
			obj_id = eval_args.getint('DATA', 'OBJ_ID')
		except:
			obj_id = eval(eval_args.get('DATA', 'OBJECTS'))[0]

		scenes = eval_utils.get_all_scenes_for_obj(eval_args)


		data = [item[1] for item in eval_args.items('DATA')]
		data[2] = eval(eval_args.get('DATA','SCENES')) if len(eval(eval_args.get('DATA','SCENES'))) > 0 else eval_utils.get_all_scenes_for_obj(eval_args)

		# print str(data)

		error_score_dict = inout.load_yaml(error_score_file)
		try:
			sixd_recall = error_score_dict['obj_recalls'][obj_id]
		except:
			continue
		


		if error_type=='re':
			data_re.append({'exp_name':exp_name, 'eval_name':eval_name, 'error_type':error_type, 'thres':error_thres,
				'top': topn, 'sixd_recall': sixd_recall, 'EST_BBS': estimate_bbs, 'eval_data': str(data[:2]+ [occl]),
				'eval_scenes': str(data[2]),'eval_obj': str(data[3])})
			err_file = os.path.join(os.path.dirname(os.path.dirname(error_score_file)),'latex/R_err_hist.tex')
			try:
				with open(err_file,'r') as f:
					for line in f:
						if re.match('(.*)legend entries(.*)',line):
							auc_re = float(line.split('=')[2].split('}')[0])
							auc_rerect = float(line.split('=')[3].split('}')[0])

				data_auc_re.append({'exp_name':exp_name, 'eval_name':eval_name, 'error_type':'auc_re', 'thres':'None',
					'top': topn, 'sixd_recall': auc_re, 'EST_BBS': estimate_bbs, 'eval_data': str(data[:2]+ [occl]),
					'eval_scenes': str(data[2]),'eval_obj': str(data[3])})
				data_auc_rerect.append({'exp_name':exp_name, 'eval_name':eval_name, 'error_type':'auc_rerect', 'thres':'None',
					'top': topn, 'sixd_recall': auc_rerect, 'EST_BBS': estimate_bbs, 'eval_data': str(data[:2]+ [occl]),
					'eval_scenes': str(data[2]),'eval_obj': str(data[3])})

				if not data_paper_auc.has_key(int(data[3])):
					data_paper_auc[int(data[3])] = {}
					data_paper_auc[int(data[3])]['eval_obj'] = int(data[3])
				data_paper_auc[int(data[3])][eval_name+'_'+'auc_re'+'_'+str(data[1])] = float(auc_re)*100
				data_paper_auc[int(data[3])][eval_name+'_'+'auc_rerect'+'_'+str(data[1])] = float(auc_rerect)*100
			except:
				print err_file, 'not found'
			
		elif error_type=='te':
			data_te.append({'exp_name':exp_name, 'eval_name':eval_name, 'error_type':error_type, 'thres':error_thres,
				'top': topn, 'sixd_recall': sixd_recall, 'EST_BBS': estimate_bbs, 'eval_data': str(data[:2] + [occl]),
				'eval_scenes': str(data[2]), 'eval_obj': str(data[3])})
		elif error_type=='vsd':
			data_vsd.append({'exp_name':exp_name, 'eval_name':eval_name, 'error_type':error_type, 'thres':error_thres,
				'top': topn, 'sixd_recall': sixd_recall, 'EST_BBS': estimate_bbs, 'eval_data': str(data[:2]+ [occl]),
				'eval_scenes': str(data[2]),'eval_obj': int(data[3]) if '[' not in data[3] else eval(data[3])[0]})
			if not data_paper_vsd.has_key(int(data[3])):
				data_paper_vsd[int(data[3])] = {}
				data_paper_vsd[int(data[3])]['eval_obj'] = int(data[3])
			data_paper_vsd[int(data[3])][eval_name+'_'+error_type+'_'+str(data[1])] = float(sixd_recall)*100

		elif error_type=='cou':
			data_cou.append({'exp_name':exp_name, 'eval_name':eval_name, 'error_type':error_type, 'thres':error_thres,
				'top': topn, 'sixd_recall': sixd_recall, 'EST_BBS': estimate_bbs, 'eval_data': str(data[:2]+ [occl]),
				'eval_scenes': str(data[2]),'eval_obj': str(data[3])})
		elif error_type=='add':
			data_add.append({'exp_name':exp_name, 'eval_name':eval_name, 'error_type':error_type, 'thres':error_thres,
				'top': topn, 'sixd_recall': sixd_recall, 'EST_BBS': estimate_bbs, 'eval_data': str(data[:2]+ [occl]),
				'eval_scenes': str(data[2]),'eval_obj': str(data[3])})
		elif error_type=='proj':
			data_proj.append({'exp_name':exp_name, 'eval_name':eval_name, 'error_type':error_type, 'thres':error_thres,
				'top': topn, 'sixd_recall': sixd_recall, 'EST_BBS': estimate_bbs, 'eval_data': str(data[:2]+ [occl]),
				'eval_scenes': str(data[2]),'eval_obj': str(data[3])})
		elif error_type=='adi':
			data_adi.append({'exp_name':exp_name, 'eval_name':eval_name, 'error_type':error_type, 'thres':error_thres,
				'top': topn, 'sixd_recall': sixd_recall, 'EST_BBS': estimate_bbs, 'eval_data': str(data[:2]+ [occl]),
				'eval_scenes': str(data[2]),'eval_obj': str(data[3])})
		else:
			print 'error not known: ', error_type


	
	if len(data_re) > 0:
		df_re = pd.DataFrame(data_re).sort_values(by=['eval_obj','eval_name','eval_data','sixd_recall'])
		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_content.append(df_re.to_latex(index=False, multirow=True))
		latex_content.append('\\end{adjustbox}')
		latex_content.append('\n')
		latex_content.append('\n')
	if len(data_auc_re) > 0:
		df_re = pd.DataFrame(data_auc_re).sort_values(by=['eval_obj','eval_name','eval_data','sixd_recall'])
		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_content.append(df_re.to_latex(index=False, multirow=True))
		latex_content.append('\\end{adjustbox}')
		latex_content.append('\n')
		latex_content.append('\n')
	if len(data_auc_rerect) > 0:
		df_re = pd.DataFrame(data_auc_rerect).sort_values(by=['eval_obj','eval_name','eval_data','sixd_recall'])
		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_content.append(df_re.to_latex(index=False, multirow=True))
		latex_content.append('\\end{adjustbox}')
		latex_content.append('\n')
		latex_content.append('\n')
	if len(data_te) > 0:
		df_te = pd.DataFrame(data_te).sort_values(by=['eval_obj','eval_name','eval_data','sixd_recall'])
		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_content.append(df_te.to_latex(index=False, multirow=True))
		latex_content.append('\\end{adjustbox}')
		latex_content.append('\n')
		latex_content.append('\n')
	if len(data_cou) > 0:
		df_cou = pd.DataFrame(data_cou).sort_values(by=['eval_obj','eval_name','eval_data','sixd_recall'])
		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_content.append(df_cou.to_latex(index=False, multirow=True))
		latex_content.append('\\end{adjustbox}')
		latex_content.append('\n')
		latex_content.append('\n')
	if len(data_add) > 0:
		df_add = pd.DataFrame(data_add).sort_values(by=['eval_obj','eval_name','eval_data','sixd_recall'])
		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_content.append(df_add.to_latex(index=False, multirow=True))
		latex_content.append('\\end{adjustbox}')
		latex_content.append('\n')
		latex_content.append('\n')
	if len(data_proj) > 0:
		df_proj = pd.DataFrame(data_proj).sort_values(by=['eval_obj','eval_name','eval_data','sixd_recall'])
		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_content.append(df_proj.to_latex(index=False, multirow=True))
		latex_content.append('\\end{adjustbox}')
		latex_content.append('\n')
		latex_content.append('\n')
	if len(data_adi) > 0:
		df_adi = pd.DataFrame(data_adi).sort_values(by=['eval_obj','eval_name','eval_data','sixd_recall'])
		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_content.append(df_adi.to_latex(index=False, multirow=True))
		latex_content.append('\\end{adjustbox}')
		latex_content.append('\n')
		latex_content.append('\n')
	if len(data_paper_vsd) > 0:
		df_paper = pd.DataFrame.from_dict(data_paper_vsd, orient='index')
		cols = ['eval_obj']  + [col for col in df_paper if col != 'eval_obj']
		df_paper = df_paper[cols]
		df_paper = df_paper.sort_index(axis=1)
		df_paper.loc['mean'] = df_paper.mean(axis=0)
		# df_paper.loc['mean'][0] = 0

		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_list = df_paper.to_latex(index=False, multirow=True, float_format='%.2f').splitlines()
		latex_list.insert(len(latex_list)-3, '\midrule')
		latex_new = '\n'.join(latex_list)
		latex_content.append(latex_new)
		latex_content.append('\\end{adjustbox}')
		latex_content.append('\n')
		latex_content.append('\n')
	if len(data_paper_auc) > 0:
		df_paper = pd.DataFrame.from_dict(data_paper_auc, orient='index')
		cols = ['eval_obj']  + [col for col in df_paper if col != 'eval_obj']
		df_paper = df_paper[cols]
		df_paper = df_paper.sort_index(axis=1)
		df_paper.loc['mean'] = df_paper.mean(axis=0)
		# df_paper.loc['mean'][0] = 0

		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_list = df_paper.to_latex(index=False, multirow=True, float_format='%.2f').splitlines()
		latex_list.insert(len(latex_list)-3, '\midrule')
		latex_new = '\n'.join(latex_list)
		latex_content.append(latex_new)
		latex_content.append('\\end{adjustbox}')
		latex_content.append('\n')
		latex_content.append('\n')
	if len(data_vsd) > 0:
		df_vsd = pd.DataFrame(data_vsd).sort_values(by=['eval_obj','eval_name','eval_data','sixd_recall'])
		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_content.append(df_vsd.to_latex(index=False, multirow=True))
		latex_content.append('\\end{adjustbox}')

	latex_content = ''.join(latex_content)


	full_filename = os.path.join(exp_group_path,'latex','report.tex')
	if not os.path.exists(os.path.join(exp_group_path,'latex')):
		os.makedirs(os.path.join(exp_group_path,'latex'))


	with open(full_filename, 'w') as f:
	    f.write(prolog % (time.ctime(),experiment_group.replace('_','\_')))
	    f.write(latex_content)
	    f.write(epilog)

	from subprocess import check_output, Popen
	check_output(['pdflatex', 'report.tex'], cwd=os.path.dirname(full_filename))
	Popen(['okular', 'report.pdf'], cwd=os.path.dirname(full_filename))

	print 'finished'



if __name__ == '__main__':
    main()