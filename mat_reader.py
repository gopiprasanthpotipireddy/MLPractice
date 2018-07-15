import os
import h5py as h5
import numpy as np
import pandas as pd
import seaborn as sns
import mat_img_reader as mir
from sklearn import tree



f=[]
file_path="C:/Users/HOME/Desktop/PRASANTH/brainTumorDataPublic_1766/"

for i in range(10):
     f.append(h5.File(os.path.join(file_path,str(i+1)+".mat"),'a'))
     

 

"""
f=h5.File(os.path.join(file_path,"10.mat"),'a')
  for k,v in f.items():
    np.array(k)
    np.array(v)
   
"""

p=[]
for i in range(0,10):
    p.append(mir.Patient('','','','',''))
    p[i].image=np.mat(f[i]['/cjdata/image'])
    p[i].PID=np.array(f[i]['/cjdata/PID'])
    p[i].label=np.array(f[i]['/cjdata/label'])
    p[i].tumorBorder=np.array(f[i]['/cjdata/tumorBorder'])
    p[i].tumorMask=np.array(f[i]['/cjdata/tumorMask'])

columns =['PID', 'image', 'label', 'tumorBorder', 'tumorMask']    

d={'PID':[], 'image':[], 'label':[], 'tumorBorder':[], 'tumorMask':[]}

for i in range(10):
    d['PID'].append(p[i].PID)
    d['image'].append(p[i].image)
    d['label'].append(p[i].label)
    d['tumorBorder'].append(p[i].tumorBorder)
    d['tumorMask'].append(p[i].tumorMask)
    

Patient_data=pd.DataFrame(list(d.values()),columns)

Patient_data=Patient_data.transpose()

t=tree.DecisionTreeClassifier()
#Decision tree classifier
x_train=Patient_data[['tumorBorder']]
y_train=Patient_data[['label']]




#preprocessing work


