#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:51:54 2019

@author: asabater
"""

import pickle
import json
import datetime
import pandas as pd
from tqdm import tqdm
import time


model_folder = 'models/24h_0413_0449_model_92/'

model = pickle.load(open(model_folder + 'model.pckl', 'rb'))
stats = json.load(open(model_folder + 'stats.json', 'r'))

d_file = stats['dataset_filename'][:-5].split('_')
w = [ i[1:] for i in d_file if i.startswith('w')][0]
sd = [ i[2:] for i in d_file if i.startswith('sd')][0]
sh = [ i[2:] for i in d_file if i.startswith('sh')][0]
means = pickle.load(open('datasets/means_sd{}_sh{}_w{}.pckl'.format(sd, sh, w), 'rb'))


#%%


# loc d m h

hours = list(range(0, 23+1))
days = list(range(1, 31+1))
months = list(range(1, 12+1))
#years = list(range(2008, 2018+1))
year = 2016

locations = list(means.keys())

data = { loc:[] for loc in locations }

for loc in locations:
	for d in days:
		for m in months:
			for h in hours:
#				for y in years:
					
				try:
					dt = datetime.datetime(year, m, d, hour=h)
				except:
					continue
				
				data[loc].append((dt, 
					 means[loc]['day_{}'.format(d)]['month_{}'.format(m)]['hour_{}'.format(h)]))
					

for loc in locations: data[loc] = sorted(data[loc], key=lambda tup: tup[0])
index = [ i for i,d in data[loc] ]
for loc in locations: data[loc] = [ d for i,d in data[loc] ]
means_df = pd.DataFrame(data, index = index)


data = pd.read_csv('./ftp/ENTRADA/datos.csv', index_col=0)
data.index = pd.to_datetime(data.index)
data_index = data.index


data_pred_v1 = {}
t_v1 = time.time()
for col in tqdm(stats['columns'], total=len(stats['columns'])):
	
	if col == 'day':
		data_pred_v1['day'] = data.index.day
	elif col == 'month':
		data_pred_v1['month'] = data.index.month
	elif '-d' in col:
		diff = int(col.split('-d')[1])
		loc = col.split('-d')[0]

#		col_index = data_index + pd.DateOffset(days=diff)
		col_index = data_index + datetime.timedelta(days=diff)
		col_index = [ ci.replace(year=year) for ci in col_index ]
		
		data_pred_v1[col] = data[loc].values - means_df.loc[col_index, loc].values
	else:
		data_pred_v1[col] = data[loc].values

data_pred_v1 = pd.DataFrame(data_pred_v1)
t_v1 = time.time() - t_v1


# %%

from joblib import Parallel, delayed


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


dataset_filename = './ftp/ENTRADA/datos.csv'
data_pred_v2 = pd.read_csv(dataset_filename, index_col=0)
data_pred_v2.index = pd.to_datetime(data_pred_v2.index)
data_columns = ['ALAGON_NR', 'GRISEN_NR', 'NOVILLAS_NR', 'TAUSTE_NR',  'TUDELA_NR', 'ZGZ_NR']
data_pred_v2 = data_pred_v2.loc[:, data_columns]


t_v2 = time.time()
print('Building prediction dataset')
new_columns = Parallel(n_jobs=-1, max_nbytes=None)(delayed(get_column_data)(col, data_pred_v2, means) 
		for col in tqdm(stats['columns'], total=len(stats['columns'])))
data_pred_v2 = pd.DataFrame(dict(new_columns))[stats['columns']]

t_v2 = time.time() - t_v2


#%%

print('t_v1:', t_v1)
print('t_v2:', t_v2)


# %%

X_train, X_val, y_train, y_val = pickle.load(open(stats['dataset_filename'], 'rb'))
data_pred_v3 = pd.concat([X_train, X_val])
y = pd.concat([y_train, y_val])


#%%

from sklearn.metrics import mean_squared_error

data_pred_v1 = data_pred_v1.sort_index()
data_pred_v2 = data_pred_v2.sort_index()
data_pred_v3 = data_pred_v3.sort_index()
y = y.sort_index()

preds_v1 = model.predict(data_pred_v1[stats['columns']])
preds_v2 = model.predict(data_pred_v2[stats['columns']])
preds_v3 = model.predict(data_pred_v3[stats['columns']])
preds_train = model.predict(X_train[stats['columns']])
preds_val = model.predict(X_val[stats['columns']])

score_v1 = mean_squared_error(y, preds_v1)
score_v2 = mean_squared_error(y, preds_v2)
score_v3 = mean_squared_error(y, preds_v3)
score_train = mean_squared_error(y_train, preds_train)
score_val = mean_squared_error(y_val, preds_val)


#%%

print('score_v1:', score_v1)
print('score_v2:', score_v2)
print('score_v3:', score_v3)
print('score_train:', score_train)
print('score_val:', score_val)





					