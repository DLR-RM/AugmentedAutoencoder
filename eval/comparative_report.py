import numpy as np
import pandas as pd
import os
import glob
import time
import argparse

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

from ae import utils as u#
from sixd_toolkit.pysixd import inout
import ConfigParser


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('experiment_group')
	args = parser.parse_args()

	experiment_group = args.experiment_group

	workspace_path = os.environ.get('AE_WORKSPACE_PATH')

	exp_group_path = os.path.join(workspace_path, 'experiments', experiment_group)
	error_score_files = glob.glob(os.path.join(exp_group_path, '*/eval/*/*/error*/scores*'))

	data_re = []
	data_te = []
	data_vsd = []
	latex_content = []

	for error_score_file in error_score_files:
		split_path = error_score_file.split('/')
		exp_name = split_path[-6]
		eval_name = split_path[-4]
		test_data = split_path[-3]
		error_type = split_path[-2].split('_')[0].split('=')[1]
		topn = split_path[-2].split('=')[2].split('_')[0]
		

		eval_cfg_file_path = os.path.join(workspace_path, 'experiments', experiment_group, 
			exp_name, 'eval', eval_name, test_data, 'eval.cfg')
		eval_args = ConfigParser.ConfigParser()
		eval_args.read(eval_cfg_file_path)
		estimate_bbs = eval_args.getboolean('BBOXES', 'ESTIMATE_BBS')
		obj_id = eval(eval_args.get('DATA', 'OBJECTS'))[0]
		data = [item[1] for item in eval_args.items('DATA')]
		print str(data)


		error_score_dict = inout.load_yaml(error_score_file)
		sixd_recall = error_score_dict['obj_recalls'][obj_id]

		if error_type=='re':
			data_re.append({'exp_name':exp_name, 'eval_name':eval_name, 'error_type':error_type, 
				'top': topn, 'sixd_recall': sixd_recall, 'EST_BBS': estimate_bbs, 'eval_data': str(data)})
		elif error_type=='te':
			data_te.append({'exp_name':exp_name, 'eval_name':eval_name, 'error_type':error_type, 
				'top': topn, 'sixd_recall': sixd_recall, 'EST_BBS': estimate_bbs, 'eval_data': str(data)})
		elif error_type=='vsd':
			data_vsd.append({'exp_name':exp_name, 'eval_name':eval_name, 'error_type':error_type, 
				'top': topn, 'sixd_recall': sixd_recall, 'EST_BBS': estimate_bbs, 'eval_data': str(data)})
		else:
			print 'error not known: ', error_type



	df_re = pd.DataFrame(data_re).sort_values(by=['eval_name','sixd_recall'])
	df_te = pd.DataFrame(data_te).sort_values(by=['eval_name','sixd_recall'])
	df_vsd = pd.DataFrame(data_vsd).sort_values(by=['eval_name','sixd_recall'])


	if len(data_re) > 0:
		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_content.append(df_re.to_latex(index=False, multirow=True))
		latex_content.append('\\end{adjustbox}')
		latex_content.append('\n')
	if len(df_te) > 0:
		latex_content.append('\\begin{adjustbox}{max width=\\textwidth}')
		latex_content.append(df_te.to_latex(index=False, multirow=True))
		latex_content.append('\\end{adjustbox}')
		latex_content.append('\n')
	if len(df_vsd) > 0:
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



if __name__ == '__main__':
    main()