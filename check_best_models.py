#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:41:33 2019

@author: asabater
"""

import os
import json
import numpy as np


# TODO: Explorar feature importances de los mejores modelos


models_dir = './models/'


def get_models():
	return { m:json.load(open(models_dir+m+'/stats.json', 'r')) for m in os.listdir(models_dir) }


# %%

label = '72h'
models = get_models()

best_score = np.inf
best_model = None
best_stats = None
for model, stats in models.items():
	if label in stats['dataset_filename']:
#		print(stats['score'])
		if stats['score'] < best_score:
			best_score = stats['score']
			best_model = model
			best_stats = stats

print(best_model, '|', best_stats['dataset_filename'])

# %%
			
# =============================================================================
# Print boxplots
# =============================================================================

import os
import json
import numpy as np
import time
import matplotlib.pyplot as plt

import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# TODO: Explorar feature importances de los mejores modelos
# TODO: plot boxplot por tipo


models_dir = './models/'
labels = ['24h', '48h', '72h']
#modes = { k:[] for k in  }



def get_models():
	return { m:json.load(open(models_dir+m+'/stats.json', 'r')) for m in os.listdir(models_dir) }

def get_models_by_key(models, keys):
	results = {}
	for model, stats in models.items():
		if all([ k in stats['dataset_filename'] for k in keys]):
			results[model] = stats
	return results



###############################################################################
###############################################################################
df = pd.read_csv('./ftp/ENTRADA/datos.csv', index_col=0)
df.index = pd.to_datetime(df.index)
r_inds = df[df.RIESGO].index

models = get_models()
models = { k:v for k,v in models.items() if 'score_val_r' not in v.keys() }

for model_folder, stats in tqdm(models.items(), total=len(models)):
	
	X_train, X_val, y_train, y_val = pickle.load(open(stats['dataset_filename'], 'rb'))
	X_val_r = X_val[X_val.index.isin(r_inds)]
	y_val_r = y_val[X_val.index.isin(r_inds)]
	X_val_r = X_val_r[stats['columns']]
	
	model = pickle.load(open('models/' + model_folder + '/model.pckl', 'rb'))
	val_r_preds = model.predict(X_val_r)
	score_val_r = mean_squared_error(y_val_r, val_r_preds)
	stats['score_val_r'] = score_val_r

	json.dump(stats, open('models/' + model_folder + '/stats.json', 'w'))
###############################################################################
###############################################################################	
	
	

#fig = plt.figure(figsize=(13,13))
fig, ax_list = plt.subplots(len(labels), 1, figsize=(13,13))

good_models = ['698', '691', '721', '693', '98', '657', '777', '425', '706', '411']
score_key = 'score_val_r' 		# score_val_r, score
for n_plot, label in enumerate(labels):
	
#	modes = [['w0.25'], ['w0.5'], ['w0.75'], ['w1'], ['w1.5'], ['w3'], ['w5'], ['w7'], ['w9'], ['nw']]
#	modes = [['w0.5', 'sd4_sh1'], ['w0.5', 'sd4_sh3'], ['w0.75', 'sd4_sh1'], ['w0.75', 'sd4_sh3']]
#	modes = { '_'.join(k):k for k in modes }
	models = get_models()
	modes = [ '_'.join(v['dataset_filename'][:-5].split('_')[-3:]) for k,v in models.items() 
				if '_old' not in k and
					'1.5' not in v['dataset_filename'] and 
					'nw' not in v['dataset_filename'] and 
#					'sd6_sh1' in v['dataset_filename'] and
#					float(v['dataset_filename'].split('_')[5][1:]) < 0.75 
					float(v['dataset_filename'].split('_')[5][1:]) == 0.5
			]
	modes = { k:[] for k in sorted(set(modes)) }

	models = { k:v for k,v in get_models().items() if '_old' not in k }
	for k,v in modes.items(): modes[k] = get_models_by_key(models, [label, k] )
	
	dots, xticks, colors = [], [], []
	best_score, best_model, best_dataset, best_num_columns = np.inf, None, None, None
	for i, (mode, models) in enumerate(modes.items()):
		xticks.append((i, mode))
		for model, stats in models.items():
			dots.append((i, stats[score_key]))
#			colors.append(1 if len(stats['columns']) == 158 else 0)
			if any([ 'model_{}'.format(gm) in model for gm in good_models ]):
				colors.append('r')
			else: colors.append('g' if len(stats['columns']) == 158 else 'b')
#			colors.append(1)
			if stats[score_key] < best_score:
				best_score = stats[score_key]
				best_model = model
				best_dataset = stats['dataset_filename']
				best_num_columns = len(stats['columns'])
	
	ax_list[n_plot].scatter([ d[0] for d in dots ], [ d[1] for d in dots ], color=colors, alpha=0.7);
	ax_list[n_plot].set_ylim(min([ d[1] for d in dots ])*0.98, max([ d[1] for d in dots ])*1.02);
	ax_list[n_plot].set_xticks([ xt[0] for xt in xticks ]);
	ax_list[n_plot].set_xticklabels([ xt[1] for xt in xticks ]);
	ax_list[n_plot].set_title('{} | {} | {:.7f}\n{} | {}'.format(label, best_model, best_score, best_dataset, best_num_columns));
	ax_list[n_plot].grid();
	ax_list[n_plot].xaxis.set_tick_params(rotation=30);
	plt.tight_layout();


	current_mode = None
	count = 0
	for mode in modes:
		mode = mode.split('_')[0]
		if mode != current_mode:
			current_mode = mode
			if count != 0: ax_list[n_plot].axvline(count - 0.5, c='b', linestyle='--')
	
		count += 1
		
plt.show();


models = get_models()
good_models_data = []
for model_folder, stats in models.items():
	if any([ 'model_{}'.format(gm) in model_folder for gm in good_models ]):
		good_models_data.append('{:<23} | {} | {}'.format(model_folder, 
						  len(stats['columns']), 
						  '_'.join(stats['dataset_filename'].split('_')[-2:])))

mode = '24'
for gm in sorted(good_models_data): 
	if gm[:2] != mode:
		print('='*45)
		mode = gm[:2]
	print(gm)


# %%

if False:
	# %%
	models = get_models()
	models = { k:v for k,v in models.items() if '72h_158_w0.5_sd6_sh3' in v['dataset_filename'] }
	#models = sorted(models.items(), key=lambda x: x[1]['dataset_filename'])
	models = sorted(models.items(), key=lambda x: x[0])
	for model_folder, stats in models:
		print(model_folder, stats['score'])

	





