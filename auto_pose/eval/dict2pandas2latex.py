import pandas as pd                                                                                                 
import glob
import numpy as np
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('file_path')
args = argparser.parse_args()

path = args.file_path
results = []

# result_paths = glob.glob('pose_errors*.npy')                                                                                                                                                                
result_paths = glob.glob(path)                                                                                                                                                                
for r in result_paths:         
    print(r)                                                                                     
    s = np.load(r, allow_pickle=True)                                                                                                  
    s = s.item()
    # print len(s['preds'][0]['t_3'])                                                                                                 
    s['class'] = r.split('pose_errors_')[1].split('.npy')[0]                                                        
    results.append(s)                                                                                               
                                                                                                             
data = pd.DataFrame(results)                                                                                        
                                                                                                                
data.loc['mean'] = data.mean()                                                                                                                                                                                                      
data_paper = data[['class', '<5deg_<5cm_init', '<5deg_<5cm', 'mean_add_recall_init', 'mean_add_recall', 'mean_proj_recall_init', 'mean_proj_recall']]                                                                                   

data_paper.loc[:,'<5deg_<5cm_init':] =  data_paper.loc[:,'<5deg_<5cm_init':] *100

l = data_paper.to_latex(index=False, float_format='%.1f')
print(l)