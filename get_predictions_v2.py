# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:33:04 2019

@author: Alberto
"""

import os
import json
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm


# TODO: Explorar feature importances de los mejores modelos


models_dir = './models/'


def get_models():
	return { m:json.load(open(models_dir+m+'/stats.json', 'r')) for m in os.listdir(models_dir) }


label = '72h'
models = get_models()

best_score = np.inf
best_model_folder = None
best_stats = None
for model_folder, stats in models.items():
	if label in stats['dataset_filename']:
#		print(stats['score'])
		if stats['score'] < best_score:
			best_score = stats['score']
			best_model_folder = model_folder
			best_stats = stats


# %%

dataset_filename = './ftp/ENTRADA/datos.csv'
data = pd.read_csv(dataset_filename, index_col=0)
data.index = pd.to_datetime(data.index)
data_columns = ['ALAGON_NR', 'GRISEN_NR', 'NOVILLAS_NR', 'TAUSTE_NR',  'TUDELA_NR', 'ZGZ_NR']
data = data.loc[:, data_columns]


dataset_filename = best_stats['dataset_filename']
print(dataset_filename)

gaussian_mode = dataset_filename[:-5].split('_')[-1]

means = [ f for f in os.listdir('datasets') if gaussian_mode in f ][0]
means = pickle.load(open('datasets/' + means, 'rb'))


def get_column_data(col, data, means):
	
	if col == 'day':
		col_data = data.index.day
	elif col == 'month':
		col_data = data.index.month
	else:
		col_splt = col.split('-')
		
		if len(col_splt) == 1:
			col_data = data[col]
		else:
			place = col_splt[0]
			days_diff = int('-'.join(col_splt[1:])[1:])
			
			inds = data.index + pd.DateOffset(days=days_diff)
			col_data = []
			for old_ind, new_ind in zip(data.index, inds):
				col_data.append(
								data.loc[old_ind, place] - \
								means[place]['day_{}'.format(new_ind.day)]\
											['month_{}'.format(new_ind.month)]\
											['hour_{}'.format(new_ind.hour)]
							)
				
	return col, col_data


from joblib import Parallel, delayed

columns = best_stats['columns']
new_columns = Parallel(n_jobs=-1, max_nbytes=None)(delayed(get_column_data)(col, data, means) 
		for col in tqdm(columns, total=len(columns)))
				

data = pd.DataFrame(dict(new_columns))[columns]


# %%

from sklearn.metrics import mean_squared_error

dataset_filename = './ftp/ENTRADA/datos.csv'
y_true = pd.read_csv(dataset_filename, index_col=0)['pred_'+label]

model = pickle.load(open('models/' + best_model_folder + '/model.pckl', 'rb'))
preds = model.predict(data)

print('Pred score:', mean_squared_error(y_true, preds))





