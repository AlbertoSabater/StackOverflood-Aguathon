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
# TODO: plot boxplot por tipo


models_dir = './models/'


def get_models():
	return { m:json.load(open(models_dir+m+'/stats.json', 'r')) for m in os.listdir(models_dir) }


# %%

label = '24h'
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


# %%
			
# =============================================================================
# Print boxplots
# =============================================================================

import os
import json
import numpy as np
import time
import matplotlib.pyplot as plt


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



#fig = plt.figure(figsize=(13,13))
fig, ax_list = plt.subplots(len(labels), 1, figsize=(13,13))
scatters = []
for i in range(len(ax_list)):
	print('new_scatter')
	sp = ax_list[i].scatter([],[])
	scatters.append(sp)
	

for n_plot, label in enumerate(labels):
	
#	modes = [['w0.25'], ['w0.5'], ['w0.75'], ['w1'], ['w1.5'], ['w3'], ['w5'], ['w7'], ['w9'], ['nw']]
#	modes = [['w0.5', 'sd4_sh1'], ['w0.5', 'sd4_sh3'], ['w0.75', 'sd4_sh1'], ['w0.75', 'sd4_sh3']]
#	modes = { '_'.join(k):k for k in modes }
	models = get_models()
	modes = [ '_'.join(v['dataset_filename'][:-5].split('_')[-3:]) for k,v in models.items() 
			if '1.5' not in v['dataset_filename'] and 'nw' not in v['dataset_filename'] and \
			float(v['dataset_filename'].split('_')[5][1:]) < 3 ]
	modes = { k:[] for k in sorted(set(modes)) }

	models = get_models()
	for k,v in modes.items(): modes[k] = get_models_by_key(models, [label, k] )
	
	dots, xticks = [], []
	best_score, best_model, best_dataset = np.inf, None, None
	for i, (mode, models) in enumerate(modes.items()):
		xticks.append((i, mode))
		for model, stats in models.items():
			dots.append((i, stats['score']))
			if stats['score'] < best_score:
				best_score = stats['score']
				best_model = model
				best_dataset = stats['dataset_filename']
	
#		ax = plt.subplot(3, 1, n_plot+1)
#		ax = ax_list[n_plot]
#		ax_list[n_plot].scatter([ d[0] for d in dots ], [ d[1] for d in dots ]);
	scatters[n_plot].set_array(np.array([1]*len(dots)));
	scatters[n_plot].set_offsets(dots);
	ax_list[n_plot].set_ylim(min([ d[1] for d in dots ])*0.98, max([ d[1] for d in dots ])*1.02);
	ax_list[n_plot].set_xticks([ xt[0] for xt in xticks ]);
	ax_list[n_plot].set_xticklabels([ xt[1] for xt in xticks ]);
	ax_list[n_plot].set_title('{} | {} | {:.7f}\n{}'.format(label, best_model, best_score, best_dataset));
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
