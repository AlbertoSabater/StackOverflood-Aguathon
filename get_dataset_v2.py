# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:59:41 2019

@author: Alberto
"""

import pandas as pd
import numpy as np
import pickle
import datetime
from tqdm import tqdm
import os
import sys


dataset_filename = './ftp/ENTRADA/datos.csv'
validation_split = 0.15
return_labels = True
    
df = pd.read_csv(dataset_filename, index_col=0)
df.index = pd.to_datetime(df.index)

label_columns = ['pred_24h', 'pred_48h', 'pred_72h']
data_columns = ['ALAGON_NR', 'GRISEN_NR', 'NOVILLAS_NR', 'TAUSTE_NR',  'TUDELA_NR', 'ZGZ_NR']
	
data = df.loc[:, data_columns]
if return_labels: labels = df.loc[:, label_columns]
else: labels = None
del df
    

# %%

# Medias de cada día con los X días contiguos a esa hora y las horas contiguas
# Diferencia de las medias de cada puesto con el resto

# Entrenar con las distancias de cada día a sus media(d,m,h)/diferencias de puestos
# Distancias de cada día a X días anteriores
# Gestionar calores nan


# %%

# TODO: ponderar más fuertemente al día/hora central

side_days = 4
side_hours = 3
means_filename = './datasets/means_sd{}_sh{}.pckl'.format(side_days, side_hours)

if os.path.isfile(means_filename):
    print('Loading: ',means_filename)
    means = pickle.load(open(means_filename, 'rb'))

else:
    
    print('Calculating:', means_filename)

    hours = sorted(list(set(data.index.hour)))
    days = sorted(list(set(data.index.day)))
    months = sorted(list(set(data.index.month)))
    years = sorted(list(set(data.index.year)))
    
    
    means = {}
    
    for col in tqdm(data_columns, total=len(data_columns), file=sys.stdout):
        means[col] = {}
    
        for d in days:
            means[col]['day_{}'.format(d)] = {}
            
            for m in months:
                
                # Check if date is valid
                try : dt = datetime.datetime(2000, m, d)
                except ValueError : continue
                means[col]['day_{}'.format(d)]['month_{}'.format(m)] = {}
                
                for h in hours:
                
                    values = []
                    for y in years:
    #                    # Check if date is valid
                        try : dt = datetime.datetime(y, m, d, hour=h)
                        except ValueError: continue
    #                    dt = datetime.datetime(y, m, d, hour=h)
                        
                        sd_values = []
                        # Get side day delays
                        for sd in list(range(-side_days, side_days+1)):
                            dt_sd = dt + datetime.timedelta(days=sd)
                            
                            sh_values = []
                            # Get side hour delays
                            for sh in list(range(-side_hours, side_hours+1)):
                                dt_sh = dt_sd + datetime.timedelta(hours=sh)
                                
                                if dt_sh in data.index: sh_values.append(data.loc[dt_sh, col])
                                
                            sh_values = [ v for v in sh_values if not np.isnan(v) ]
                            sd_values.append(np.mean(sh_values))
                        
                        sd_values = [ v for v in sd_values if not np.isnan(v) ]
                        values.append(np.mean(sd_values))
                    
                    values = [ v for v in values if not np.isnan(v) ]
                    means[col]['day_{}'.format(d)]['month_{}'.format(m)]['hour_{}'.format(h)] = np.mean(values)
        #            break
        #        break
        #    break
    
    pickle.dump(means, open(means_filename, 'wb'))


# %%

# Distancia de un sample respecto a su media -> a X días/horas antes

# 1,2,3,5,7 días
# 8,12 horas
    
print('Adding daily feaures')
iter_days = list(range(-7, 7+1)) + [10,15,20,25,30, 35,40,45, 50,55]
for d in tqdm(iter_days, total=len(iter_days), file=sys.stdout):
    inds = data.index + pd.DateOffset(days=d)
    for col in data_columns:
        col_vals = []
        for old_ind, new_ind in zip(data.index, inds):
            col_vals.append(
                    data.loc[old_ind, col] - 
                    means[col]['day_{}'.format(new_ind.day)]\
                                ['month_{}'.format(new_ind.month)]\
                                ['hour_{}'.format(new_ind.hour)])

        data = data.assign(**{'{}-d{}'.format(col, d): col_vals})
    

# %%
 
#print('Adding hourly features')
#iter_hours = [4,8,12,16]
#for h in tqdm(iter_hours, total=len(iter_hours), file=sys.stdout):
#    inds = data.index + pd.DateOffset(hours=h)
#    for col in data_columns:
#        col_vals = []
#        for old_ind, new_ind in zip(data.index, inds):
#            col_vals.append(
#                    data.loc[old_ind, col] - 
#                    means[col]['day_{}'.format(new_ind.day)]\
#                                ['month_{}'.format(new_ind.month)]\
#                                ['hour_{}'.format(new_ind.hour)])
#
#        data = data.assign(**{'{}-h{}'.format(col, h): col_vals})


# %%

print('Adding date columns')
data = data.assign(day = data.index.day)
data = data.assign(month = data.index.month)


# %%
# %%
# %%


num_val = int(len(data) * validation_split)
num_train = len(data) - num_val

inds = np.random.permutation(data.index)
inds_train = inds[:num_train]
inds_val = inds[num_train:]


X_train = data.loc[inds_train]
X_val = data.loc[inds_val]
for c in labels.columns:
    dataset_filename = 'datasets/XY_{}_{}_{}.pckl'.format(validation_split, c, len(data.columns))
    pickle.dump([X_train, X_val, labels.loc[inds_train, c], labels.loc[inds_val, c]],
			open(dataset_filename, 'wb')
		)
    print(dataset_filename, 'stored')


