# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:38:38 2019

@author: Alberto
"""

from xgboost import XGBRegressor
import lightgbm as lgb

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
from sklearn.metrics import mean_squared_error


n_jobs = 8


def fit_predict_xgb(X_train, X_val, y_train, y_val):
	model = XGBRegressor(max_depth=7, n_estimators=100, learning_rate=0.05, n_jobs=n_jobs)
	model.fit(X_train, y_train,
			  eval_set = [(X_train, y_train), (X_val, y_val)],
			  verbose = False,
			  early_stopping_rounds = 10,
#			  sample_weight = sample_weight
			  )
	fi = model.feature_importances_
	
	results = model.evals_result()
	train_error = min(results['validation_0']['rmse'])
	val_error = min(results['validation_1']['rmse'])
	
	score = mean_squared_error(y_val, model.predict(X_val))
	
	return model, fi, train_error, val_error, score


# https://medium.com/@pushkarmandot/https-medium-com-pushkarmandot-what-is-lightgbm-how-to-implement-it-how-to-fine-tune-the-parameters-60347819b7fc
def fit_predict_lgb(X_train, X_val, y_train, y_val):
	
	model = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=5500, n_jobs=n_jobs)
	model.fit(X_train, y_train,
			   eval_set = [(X_train, y_train), (X_val, y_val)],
			   verbose = False,
			   early_stopping_rounds = 10
			   )

	fi = model.feature_importances_
	fi = fi/sum(fi)
	
	results = model.evals_result_
	train_error = min(results['training']['l2'])
	val_error = min(results['valid_1']['l2'])
	
	score = mean_squared_error(y_val, model.predict(X_val))
	
	return model, fi, train_error, val_error, score

# %%

# TODO: Meter en validación días/semanas enteras, no un random
# TODO: aplicar k-fold sobre feature grid search -> 3-4 modelos con features diferentes
# TODO: eliminar las X peores features no random

def train_model(dataset_filename, label, feat_mode, columns_step, early_stopping):


	X_train, X_val, y_train, y_val = pickle.load(
			open(dataset_filename, 'rb'))
	
	columns = X_train.columns.tolist()
	
	
	# %%
	
	count = 0
	count_es = 0
	num_steps = int(len(columns))
	
	errors = [] 
	best_error = np.inf
	best_score = np.inf
	best_model = None
	best_columns = None
	
	
	while len(columns) > 0:
		
		t = time.time()
		
	#	sample_weight = compute_class_weight(class_weight='balanced', classes=[True, False], y=riesgo)
	#	sample_weight = [ sample_weight[0] if r else sample_weight[1] for r in riesgo ]
	
	#	model, fi, train_error, val_error, score = fit_predict_xgb(X_train, X_val, y_train, y_val)
		model, fi, train_error, val_error, score = fit_predict_lgb(X_train, X_val, y_train, y_val)
	
		errors.append({
					'num_columns': len(X_train.columns),
					'error_train': train_error,
					'error_val': val_error,
					'training_time': (time.time()-t)/60,
					'feat_mode': feat_mode
				})
		
	#	score = mean_squared_error(y_val, model.predict(X_val))
	#	if val_error < best_error:
		if score < best_score:
			best_score = score
			best_error = val_error
			best_model = model
			best_columns = X_train.columns.tolist()
			count_es = 0
		else:
			count_es += 1
			
			
		print('{}/{} | Num columns: {} | Train: {:.7f} - Val: {:.7f} | Score: {:.7f} | {:.2f} mins | {} {}'.format(
					count, num_steps//columns_step, len(columns),
					train_error, val_error, score,
					(time.time()-t)/60,
					datetime.datetime.now().strftime('%H:%M'),
					'***' if score == best_score else ''
				))
	
		if count_es >= early_stopping: break
		
		if len(columns) <= columns_step: break
	
		if feat_mode == 'random':
			columns = np.random.choice(X_train.columns.tolist(), 
									size=len(columns)-columns_step, replace=False, p=fi)
		elif feat_mode == 'best':
			columns = X_train.columns.tolist()
			fi_inds = sorted(range(len(fi)), key=lambda k: fi[k], reverse=True)
			columns = [ columns[i] for i in fi_inds[:-columns_step] ]
			
		
	#	columns = [ columns[i] for i in fi_inds[:-columns_step] ]
		X_train = X_train[columns]
		X_val = X_val[columns]
	
			
		count +=1
		
	print(' ** Best score: {:.7f} | num_columns: {}'.format(best_score, len(best_columns)))
		
	
	# %%
	
	base_model_dir = 'models/'
	folder_num = len(os.listdir(base_model_dir))
	folder_path = base_model_dir + '{}_{}_model_{}/'.format(
			label, datetime.datetime.today().strftime('%m%d_%H%M'), folder_num)
	os.makedirs(folder_path)
	
	pickle.dump(best_model, open(folder_path + 'model.pckl', 'wb'))
	json.dump({'val_error': best_error, 
			   'score': best_score,
			   'columns': list(best_columns), 
			   'dataset_filename': dataset_filename},
			  open(folder_path + 'stats.json', 'w'))
	
	
	# %%
	
	if False:
		#%%
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
	
validation_split = 0.15
#label = '72h'
#num_columns = 120   # [499]   validation_0-rmse:0.048481	  validation_1-rmse:0.067338
#num_columns = 150   # [499]   validation_0-rmse:0.044235	  validation_1-rmse:0.060947
#num_columns = 168   # [499]   validation_0-rmse:0.043145	  validation_1-rmse:0.060269
#num_columns = 170   # [499]   validation_0-rmse:0.042698	  validation_1-rmse:0.056718
num_columns = 158   # [499]   validation_0-rmse:0.041888	  validation_1-rmse:0.056495




#X_train, X_val = pd.read_pickle('datasets/X_train_0.2_298.pckl'), pd.read_pickle('datasets/X_val_0.2_298.pckl')
#y_train, y_val = pd.read_pickle('datasets/Y_train_0.2_pred_24h_298.pckl'), pd.read_pickle('datasets/Y_val_0.2_pred_24h_298.pckl')
columns_step = 5
early_stopping = 10
num_models_per_label = 10
#gaussian_suffix = 'nw' 			# w3, nw
#feat_mode = 'random' 				# best, random


#gaussian_suffixes = [ 
#						'w0.5_sd6_sh3', 'w0.5_sd2_sh3', 
#						'w0.5_sd4_sh1', 'w0.5_sd4_sh5'
#						'w0.25_sd6_sh3', 'w0.25_sd2_sh3', 
#						'w0.25_sd4_sh1', 'w0.25_sd4_sh5',
#						'w0.75_sd6_sh3', 'w0.75_sd2_sh3', 
#						'w0.75_sd4_sh1', 'w0.75_sd4_sh5',
#					]

#gaussian_suffixes = [
#						'w0.75_sd6_sh1', 'w0.75_sd2_sh1', 
#						'w0.75_sd6_sh5', 'w0.75_sd2_sh5',
#						'w0.5_sd6_sh1', 'w0.5_sd2_sh1', 
#						'w0.5_sd6_sh5', 'w0.5_sd2_sh5',
#						'w0.25_sd6_sh1', 'w0.25_sd2_sh1', 
#						'w0.25_sd6_sh5', 'w0.25_sd2_sh5',
#					]

gaussian_suffixes = [ 'w{}_{}'.format(w, gs) for w in [0.5, 0.75, 0.25] 
						for gs in ['d6_h7', 'd6_h9', 'd8_h5', 'd8_h7', 'd8_h9']]

#dataset_filename = 'datasets/XY_{}_pred_{}_{}_{}.pckl'.format(
#		validation_split, label, num_columns, gaussian_suffix)
##		print('='*80)
##		print('|| {} | {}'.format('best', dataset_filename))
##		print('='*80)
##		train_model(dataset_filename, label, 'best', columns_step, early_stopping)
		

for i in range(num_models_per_label):
	iter_t = time.time()
	for label in ['72h', '24h', '48h']:
		label_t = time.time()
		for gaussian_suffix in gaussian_suffixes:
	#	for gaussian_suffix in ['w0.75']:
		#for gaussian_suffix in ['w0.25', 'w0.5', 'w1.5', 'w9']:
		#for gaussian_suffix in ['w1', 'w5', 'w7', 'w9']:

			t = time.time()
			dataset_filename = 'datasets/XY_{}_pred_{}_{}_{}.pckl'.format(
					validation_split, label, num_columns, gaussian_suffix)
			print('='*80)
			print('|| {}/{} | {}'.format(i+1, num_models_per_label, dataset_filename))
			print('='*80)
			train_model(dataset_filename, label, 'random', columns_step, early_stopping)
			print(' ** Time elapsed: {:.2f}'.format((time.time()-t)/60))
		
		print('*'*33)
		print('****** Label time: {:.2f} ****** '.format((time.time()-label_t)/60))
		print('*'*33)
	print('*'*41)
	print('*'*41)
	print('*********** Iter time: {:.2f} ************ '.format((time.time()-iter_t)/60))
	print('*'*41)
	print('*'*41)
	

# %%
if False:
	# %%
	
	scores = []
#	for ne in range(500, 6000, 200):
	for ne in [500]:
	
		def fit_predict_lgb(X_train, X_val, y_train, y_val):
		
			model = lgb.LGBMRegressor(learning_rate=0.05, n_estimators=ne,
								   )
			model.fit(X_train, y_train,
					   eval_set = [(X_train, y_train), (X_val, y_val)],
					   verbose = False,
					   early_stopping_rounds = 10
					   )
		
			fi = model.feature_importances_
			fi = fi/sum(fi)
			
			results = model.evals_result_
			train_error = min(results['training']['l2'])
			val_error = min(results['valid_1']['l2'])
			
			score = mean_squared_error(y_val, model.predict(X_val))
			
			return model, fi, train_error, val_error, score
		
		model, fi, train_error, val_error, score = fit_predict_lgb(X_train, X_val, y_train, y_val)
		scores.append(score)
		print(ne, '| Score:', score)
		
	
	# %%
	
	columns = X_train.columns.tolist()
	fi_inds = sorted(range(len(fi)), key=lambda k: fi[k], reverse=True)
#	columns = [ columns[i] for i in fi_inds[:-columns_step] ]
	
	plt.barh(range(len(columns)), fi[fi_inds])
	

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


	# %%
	
	from xgboost import plot_importance
	
	fig, ax = plt.subplots(1,1,figsize=(12,35))
	plot_importance(model, ax = ax, height=0.5);
	
	
	
# %%
	
	plt.bar(range(len(columns)), fi)
	
	# %%
	
	num_columns = 158
	perc = 0.07
	i = 0
	while num_columns > 0:
		step = int(num_columns*perc)
		if step == 0: break
		print(i, num_columns, step)
		num_columns -= step
		i += 1
