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
from scipy import signal
import sys




np.random.seed(123)

gaussian, sigma = True, 0.5
side_days = 4
side_hours = 1

gaussian_suffix = 'w{}'.format(sigma) if gaussian else 'nw'
means_filename = './datasets/means_sd{}_sh{}_{}.pckl'.format(side_days, side_hours, 
									 gaussian_suffix)


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

def get_mean(values, gaussian, sigma):
	if len(values) == 0:
		return np.nan
	elif gaussian:
		# Apply gaussian filter to values
		window = signal.gaussian(len(values), sigma)
		for i in range(len(values)):
			if np.isnan(values[i]): values[i], window[i] = 0, 0
		window /= sum(window)
		return sum(np.array(values) * window)		
				
	else:
		values = [ v for v in values if not np.isnan(v) ]
		return np.mean(values)


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
				
					sy_values = []
					for y in years:
	#					# Check if date is valid
						try : dt = datetime.datetime(y, m, d, hour=h)
						except ValueError: continue
	#					dt = datetime.datetime(y, m, d, hour=h)
						
						sd_values = []
						# Get side day delays
						for sd in list(range(-side_days, side_days+1)):
							dt_sd = dt + datetime.timedelta(days=sd)
							
							sh_values = []
							# Get side hour delays
							for sh in list(range(-side_hours, side_hours+1)):
								dt_sh = dt_sd + datetime.timedelta(hours=sh)
								
								if dt_sh in data.index: 
									sh_values.append(data.loc[dt_sh, col])
								
							# Add hourly mean  
							sd_values.append(get_mean(sh_values, gaussian, sigma))
						
						# Add daily mean
						sy_values.append(get_mean(sd_values, gaussian, sigma))
					
					# Add yearly mean
					means[col]['day_{}'.format(d)]['month_{}'.format(m)]['hour_{}'.format(h)] = \
												get_mean(sy_values, False, sigma)
	
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
#	inds = data.index + pd.DateOffset(hours=h)
#	for col in data_columns:
#		col_vals = []
#		for old_ind, new_ind in zip(data.index, inds):
#			col_vals.append(
#					data.loc[old_ind, col] - 
#					means[col]['day_{}'.format(new_ind.day)]\
#								['month_{}'.format(new_ind.month)]\
#								['hour_{}'.format(new_ind.hour)])
#
#		data = data.assign(**{'{}-h{}'.format(col, h): col_vals})


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
	dataset_filename = 'datasets/XY_{}_{}_{}_{}_sd{}_sh{}.pckl'.format(validation_split, 
								 c, len(data.columns), gaussian_suffix, side_days, side_hours)
	pickle.dump([X_train, X_val, labels.loc[inds_train, c], labels.loc[inds_val, c]],
			open(dataset_filename, 'wb')
		)
	print(dataset_filename, 'stored')


# %%

# TODO: actualizar dataset a todos los json stats
	



