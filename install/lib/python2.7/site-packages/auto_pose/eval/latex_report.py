import os
import glob
import time

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
\end{center}
''' % time.ctime()


epilog=\
r'''
\end{document}
'''



class Report(object):

    def __init__(self, eval_dir, log_dir):
        self.latex = []
        self.eval_dir = eval_dir
        self.log_dir = log_dir

    def write_configuration(self, train_cfg_file_path, eval_cfg_file_path):
        names = self.eval_dir.replace('_','\_').split('/')
        self.latex.append(
r'''
\begin{table}[H]
    \centering
    \begin{adjustbox}{max width=\textwidth}
        \begin{tabular}{c|c}
            \textbf{experiment group:}& %s\\
            \textbf{experiment name} & %s  \\
            \textbf{evaluation name} & %s \\
            \textbf{test dataset} & %s \\
        \end{tabular}
    \end{adjustbox}
\end{table}
''' % (names[-5],names[-4],names[-2],names[-1])
        )

        workspace_path = os.environ.get('AE_WORKSPACE_PATH')

        with open(train_cfg_file_path,'r') as f:
            with open(eval_cfg_file_path,'r') as g:
                train = f.read().replace('_','\_').replace('#','%')
                evalu = g.read().replace('_','\_').replace('#','%')
                self.latex.append(
r'''
\section{\Large Train Config}

%s

\section{\Large Evaluation Config}

%s

''' % (train, evalu))

    def merge_all_tex_files(self):
        tex_files = glob.glob(os.path.join(self.eval_dir,'latex','*.tex'))
        for file in tex_files:
            if 'report' not in file:
                with open(file,'r') as f:
                    self.latex.append('\\begin{center}\n')
                    self.latex.append('\input{%s}' % file)
                    self.latex.append('\\end{center}\n')

    def include_all_figures(self):
        pdf_files = glob.glob(os.path.join(self.eval_dir,'figures','*.pdf'))
        png_files_eval = glob.glob(os.path.join(self.eval_dir,'figures','*.png'))
        png_files = glob.glob(os.path.join(self.log_dir,'train_figures','*29999.png'))
        for file in pdf_files+png_files+png_files_eval:
            self.latex.append(
r'''
\begin{figure}
\centering
\includegraphics[width=1.\textwidth,height=0.45\textheight,keepaspectratio]{%s}
\end{figure}

''' % file)

    def save(self, pdf=True, filename = 'report.tex',open_pdf=True):
        data = ''.join(self.latex)
        full_filename = os.path.join(self.eval_dir,'latex','report.tex')
        with open(full_filename, 'w+') as f:
            f.write(prolog)
            f.write(data)
            f.write(epilog)

        if pdf:
            from subprocess import check_output, Popen
            check_output(['pdflatex', filename], cwd=os.path.dirname(full_filename))
            if open_pdf:
                Popen(['okular', filename.split('.')[0] + '.pdf'], cwd=os.path.dirname(full_filename))
