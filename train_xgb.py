# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:38:38 2019

@author: Alberto
"""

from xgboost import XGBRegressor
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

import os
import datetime
import json
import pickle


validation_split = 0.2
label = '72h'
#X_train, X_val = pd.read_pickle('datasets/X_train_0.2_298.pckl'), pd.read_pickle('datasets/X_val_0.2_298.pckl')
#y_train, y_val = pd.read_pickle('datasets/Y_train_0.2_pred_24h_298.pckl'), pd.read_pickle('datasets/Y_val_0.2_pred_24h_298.pckl')
X_train, X_val, y_train, y_val = pickle.load(
		open('datasets/XY_{}_pred_{}_298.pckl'.format(validation_split, label), 'rb'))


columns_step = 10

riesgo = X_train['RIESGO']
del X_train['RIESGO']; del X_val['RIESGO']
columns = X_train.columns.tolist()


# %%

columns = X_train.columns.tolist()
random.shuffle(columns)
count = 0
num_steps = int(len(columns))

errors = []
best_error = np.inf
best_model = None
best_columns = None


# %%

while len(columns) > 0:
	
	t = time.time()
	
	sample_weight = compute_class_weight(class_weight='balanced', classes=[True, False], y=riesgo)
	sample_weight = [ sample_weight[0] if r else sample_weight[1] for r in riesgo ]

	model = XGBRegressor(max_depth=7, n_estimators=400, learning_rate=0.05, n_jobs=6)
	model.fit(X_train, y_train,
			  eval_set = [(X_train, y_train), (X_val, y_val)],
			  verbose = False,
			  early_stopping_rounds = 10,
			  sample_weight = sample_weight)
	
	
	fi = model.feature_importances_
	fi_inds = sorted(range(len(fi)), key=lambda k: fi[k], reverse=True)
	
	results = model.evals_result()
	train_error = min(results['validation_0']['rmse'])
	val_error = min(results['validation_1']['rmse'])
	
	errors.append({
				'num_columns': len(columns),
				'error_train': train_error,
				'error_val': val_error,
				'training_time': (time.time()-t)/60
			})
	
	if val_error < best_error:
		best_error = val_error
		best_model = model
		best_columns = columns
		
	print('{}/{} | Num columns: {} | Train: {:.5f} - Val: {:.5f} | {:.2f} mins | {}'.format(
				count, num_steps//columns_step, len(columns),
				train_error, val_error,
				(time.time()-t)/60,
				datetime.datetime.now().strftime('%H:%M')
			))
	
	if len(columns) <= columns_step: break
	columns = np.random.choice(columns, size=len(columns)-columns_step, replace=False, p=fi)
#	columns = [ columns[i] for i in fi_inds[:-columns_step] ]
	X_train = X_train[columns]
	X_val = X_val[columns]

		
	count +=1
	

# %%

base_model_dir = 'models/'
folder_num = len(os.listdir(base_model_dir))
folder_path = base_model_dir + '{}_{}_model_{}/'.format(
		label, datetime.datetime.today().strftime('%m%d_%H%M'), folder_num)
os.makedirs(folder_path)

pickle.dump(best_model, open(folder_path + 'xgb.pckl', 'wb'))
json.dump({'val_error': best_error, 'columns': best_columns.tolist()},
		  open(folder_path + 'stats.json', 'w'))


# %%

#plt.figure(figsize=(12,7))
fig, ax1 = plt.subplots(figsize=(12,7))
ax2 = ax1.twinx()

ax1.plot([ d['num_columns'] for d in errors[:-1] ],
		 [ d['error_train'] for d in errors[:-1] ], 
		 c='m', label='Train')
ax1.plot([ d['num_columns'] for d in errors[:-1] ],
		 [ d['error_val'] for d in errors[:-1] ], 
		 c='g', label='Validation')

ax2.plot([ d['num_columns'] for d in errors ],
		 [ d['training_time'] for d in errors ], label='Training time')

ax1.set_ylabel('RMSE', color='g')
ax2.set_ylabel('Minutes', color='g')

ax1.legend()
ax2.legend()


# %%

if False:
	# %%
	
	t = time.time()
	model = XGBRegressor(max_depth=7, n_estimators=500, learning_rate=0.05, n_jobs=8)
	model.fit(X_train, y_train,
			  eval_set = [(X_train, y_train), (X_val, y_val)],
			  verbose = True,
			  early_stopping_rounds = 10,
#			  sample_weight = sample_weight
			  )
	print((time.time()-t)/60)


	# %%
	
	results = model.evals_result()
	plt.figure(figsize=(12,10))
	plt.plot(results['validation_0']['rmse'], label='Train')
	plt.plot(results['validation_1']['rmse'], label='Test')


