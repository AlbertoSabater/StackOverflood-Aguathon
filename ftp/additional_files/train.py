# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:38:38 2019

@author: Alberto
"""

import lightgbm as lgb

import time
import matplotlib.pyplot as plt
import numpy as np

import os
import datetime
import json
import pickle
from sklearn.metrics import mean_squared_error


n_jobs = 8


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
	
validation_split = 0.15
label = '24h'
num_columns = 158
gaussian_suffix = 'w0.5_sd10_sh3'

# Load dataset
dataset_filename = 'datasets/XY_{}_pred_{}_{}_{}.pckl'.format(
		validation_split, label, num_columns, gaussian_suffix)
X_train, X_val, y_train, y_val = pickle.load(open(dataset_filename, 'rb'))
columns = X_train.columns.tolist()

# Train model
model, fi, train_error, val_error, score = fit_predict_lgb(X_train, X_val, y_train, y_val)

# Create model folder
base_model_dir = 'models/'
folder_num = len(os.listdir(base_model_dir))
folder_path = base_model_dir + '{}_{}_model_{}/'.format(
		label, datetime.datetime.today().strftime('%m%d_%H%M'), folder_num)
os.makedirs(folder_path)

# Store model and stats
pickle.dump(model, open(folder_path + 'model.pckl', 'wb'))
json.dump({'val_error': val_error, 
		   'columns': list(columns), 
		   'dataset_filename': dataset_filename},
		  open(folder_path + 'stats.json', 'w'))
	
	
