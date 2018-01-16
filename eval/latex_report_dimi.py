# -*- coding: utf-8 -*-

import os

font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')

prolog =\
r'''
\documentclass[a4paper,table]{article}
\usepackage{graphicx}
\usepackage{float}
\usepackage{tikz}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{fancyhdr}
\usepackage{fontspec}
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

\title{Experimental Protocol}
\date{\today}

\begin{document}
\setmainfont[Path = %s/, BoldFont={myriad-pro-light_675.otf},ItalicFont={Myriad Pro Light Italic.otf}]{MyriadPro-Light.otf}
''' % (font_path)

epilog=\
r'''
\end{document}
'''

class Report(object):

    def __init__(self, experiment_name='None'):
        self.latex = []
        self.raw_latex = []
        self.experiment_name = experiment_name

    #def add_net(self, net_spec, network_file_path):
    #   to_add = network_to_tikz(net_spec, network_file_path)

    def add_figure(self, figure_name, caption, size=r'width=\textwidth'):
        to_add = r'''
\begin{{figure}}[H]
    \centering
    \includegraphics[{size}]{{{figure_name}}}
    \caption{{{caption}}}
\end{{figure}}''' .format(size=size, figure_name=figure_name, caption = caption)
        self.latex.append(to_add)
        to_add = r'''
\begin{{figure}}[H]
    \centering
    \includegraphics[{size}]{{{figure_name}}}
    \caption{{{caption}}}
\end{{figure}}''' .format(size=size, figure_name=self.experiment_name+'/'+figure_name, caption = caption)
        self.raw_latex.append(to_add)



    def add_subfigures(self, name, figure_name1, figure_name2, file1, file2):
        to_add = r'''
\begin{{figure}}[H]
        \centering
\begin{{subfigure}}[h]{{0.45\textwidth}}
                \includegraphics[width=\textwidth]{{{file1}}}
                \caption{{ {figure_name1} }}
\end{{subfigure}}
~~~~~~~~~~~~~~~
\begin{{subfigure}}[h]{{0.45\textwidth}}
                \includegraphics[width=\textwidth]{{{file2}}}
                \caption{{ {figure_name2} }}  
\end{{subfigure}}
\caption{{ {name} }}
\end{{figure}}'''.format(file1=file1, figure_name1=figure_name1, file2=file2, figure_name2=figure_name2, name=name)
        self.latex.append(to_add)
        to_add = r'''
\begin{{figure}}[H]
        \centering
\begin{{subfigure}}[h]{{0.45\textwidth}}
                \includegraphics[width=\textwidth]{{{file1}}}
                \caption{{ {figure_name1} }}
\end{{subfigure}}
~~~~~~~~~~~~~~~
\begin{{subfigure}}[h]{{0.45\textwidth}}
                \includegraphics[width=\textwidth]{{{file2}}}
                \caption{{ {figure_name2} }}  
\end{{subfigure}}
\caption{{ {name} }}
\end{{figure}}'''.format(file1=self.experiment_name + '/' + file1, figure_name1=figure_name1, file2=self.experiment_name + '/' +file2, figure_name2=figure_name2, name=name)
        self.raw_latex.append(to_add)

    def add_section(self, section, label):
        to_add = r'''
\section{{{}}}
\label{{{}}}'''.format(section, label)
        self.latex.append(to_add)
        self.raw_latex.append(to_add)

    def add_subsection(self, section):
        to_add = r'''
\subsection{{{}}}'''.format(section)
        self.latex.append(to_add)
        self.raw_latex.append(to_add)   


    def create_imgs(self, title, img_paths, captions):
        prolog =\
r'''\begin{figure}[H]
\captionsetup[subfigure]{labelformat=empty}
\centering'''
        
        subimage =\
r'''
\begin{subfigure}[h]{0.095\textwidth}
    \includegraphics[width=\textwidth]{%s}
    \caption{%s}
\end{subfigure}'''

        self.latex.append(prolog)
        self.raw_latex.append(prolog)

        for path, caption  in zip(img_paths, captions):
            subimage_formatted = subimage % (path, caption)
            self.latex.append(subimage_formatted)
            self.raw_latex.append(subimage_formatted)

        epilog =\
r'''\caption{ %s }
\end{figure}'''

        epilog_formatted = epilog % (title)
        self.latex.append(epilog_formatted)
        self.raw_latex.append(epilog_formatted)


    def add_table(self, kw):

        prolog = r'''
\begin{table}[H]
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{lc|lc|lc} 
'''
        epilog = r'''
\end{tabular}
\end{adjustbox}
\end{table}'''

        #        prolog = r'''
        #\begin{center}
        #\begin{tabular}{lc|lc|lc} 
        #'''
        #        epilog = r'''
        #\end{tabular}
        #\end{center}'''

        buf = []
        for i, val in enumerate(kw):
            if val[0]==None:
                val = ('', '')
            if i%3 == 0 or i%3 == 1:
                buf.append('{val[0]} & {val[1]} &'.format(val=val))
            else:
                buf.append('{val[0]} & {val[1]} \\\\ \n'.format(val=val))

        self.latex.append( prolog + ''.join(buf) + epilog )
        self.raw_latex.append(prolog + ''.join(buf) + epilog)

    def add(self, data):
        self.latex.append(data)
        self.raw_latex.append(data)

    def create_stat_table(self, exp_names, exp_stat_to_show, evaluation_metrics):
        prolog = r'''

\begin{table}[H]
\centering
\begin{adjustbox}{max width=\textwidth}
\begin{tabular}{|l|%s|}
\cline{2-%d}
'''
        epilog = r'''
\end{tabular}
\end{adjustbox}
\caption{Summary of all experiments} 
\end{table}
'''
# 

        cols = len(exp_stat_to_show.keys()) + len(evaluation_metrics.keys())
        rows = len(exp_stat_to_show[exp_stat_to_show.keys()[0]])
        
        tabs = '|'.join( ['c' for _ in xrange( len(exp_stat_to_show.keys()) )] ) 
        tabs += '||'
        tabs += '|'.join( ['c' for _ in xrange( len( evaluation_metrics.keys()) ) ] )

        prolog = prolog % ( tabs , cols+1)

        buf = []

        header =  '\multicolumn{1}{c|}{} &' + '&'.join([ '\\rot{%s}' % key.replace('_',' ').title() for key in sorted(exp_stat_to_show.keys())]) + '&'
        header += '&'.join([ '\\rot{%s}' % key[0].replace('_','\_') for key in sorted(evaluation_metrics.keys())]) + '\\\\ \n'

        buf.append( header )
        buf.append('\hline \n')

        for row in xrange(rows):
            t_buf = []
            for key in sorted(exp_stat_to_show.keys()):
                value = exp_stat_to_show[key][row]
                if isinstance(value, str):
                	if value.lower() == 'true':
                	    value = r'\OK'
                	elif value.lower() == 'false':
                	    value = ' '
                t_buf.append(value)
            for key in sorted(evaluation_metrics.keys()):
                value = evaluation_metrics[key][row]
                if key == ('Training Time [dd:hh:mm]', False):
                    training_time = value
                    days = int(training_time / 86400)
                    hours = int((training_time - days * 86400)/3600)
                    minutes = int((training_time - days * 86400 - hours * 3600)/60)

                    time_formated = '{DD:02}:{HH:02}:{MM:02}'.format(DD=days, HH=hours, MM=minutes)
                    if key[1] == True: # Highest IS BEST
                        if float(value) == max([float(metric) for metric in evaluation_metrics[key]]):
                            value = '\\bf{%s}' % time_formated
                        else:
                            value = time_formated
                    else: # LOWEST IS BEST
                        if float(value) == min([float(metric) for metric in evaluation_metrics[key]]):
                            value = '\\bf{%s}' % time_formated
                        else:
                            value = time_formated
                else:
                    if value != '':
                        if key[1] == True: # Highest IS BEST
                            if float(value) == max([float(metric) for metric in evaluation_metrics[key] if metric != '']):
                                value = '\\bf{%s}' % value
                        else: # LOWEST IS BEST
                            if float(value) == min([float(metric) for metric in evaluation_metrics[key] if metric != '']):
                                value = '\\bf{%s}' % value

                t_buf.append(value)

            if row % 2 == 0:
                buf.append('\\rowcolor{black!15}\n')
            else:
                buf.append('\\rowcolor{white!15}\n')
            buf.append( r'\hyperref[%s]{''%s''}' % (exp_names[row].replace('_', ''), exp_names[row].replace('_', '\_')) + '&' + '&'.join(t_buf) +'\\\\ \n')
            buf.append('\hline \n')

        self.latex.append( prolog + ''.join(buf) + epilog )
        self.raw_latex.append(prolog + ''.join(buf) + epilog)

    def save(self, filename):
        data = ''.join(self.latex)
        with open(filename, 'w') as f:
            f.write(prolog)
            f.write(data)
            f.write(epilog)

    def save_raw(self, report_raw_file):
        assert self.experiment_name != 'None'
        data = ''.join(self.raw_latex)
        with open(report_raw_file, 'w') as f:
            f.write(data)
