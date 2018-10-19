# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 13:53:14 2018

@author: hejiew
"""

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing


folder_name = './ck_processed'
file_list = os.listdir(folder_name)

new_folder = './resp_data'
if not os.path.exists(new_folder):
    os.mkdir(new_folder)

n = 0
for file_name in file_list:
    
    if file_name.find('.csv') == -1:
        continue
    
    if file_name.find('m1') == -1:
        continue
    
    full_name = os.path.join(folder_name, file_name)
    _df = pd.read_csv(full_name, sep=' ', 
                     engine='python')
    _df = _df.dropna(axis='columns')
    
    d = {'time': _df.values[:,0],
         'value': _df.values[:,1]}
    
    df = pd.DataFrame(d)
    
    n_samples = int(len(df) / 30) -30
    
    f = 26; data = []
    for i in np.arange(start=60, stop=n_samples):
        X = preprocessing.scale ( df.value[(i-60)*f:i*f] )
        data.append(X[-30*f:])
    data = np.array(data)
    
    npy_name = '{:0>3d}.npy'.format(n+1)
    save_file = os.path.join(new_folder, npy_name)
    np.save(save_file, data)

    assert(np.allclose([0,1], [np.mean(X), np.std(X)]))
    print('{} --> {}, data shape: {}'.format(file_name, npy_name, data.shape))
    
    n = n + 1
