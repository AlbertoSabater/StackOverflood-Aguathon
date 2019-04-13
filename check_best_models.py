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

import matplotlib.pyplot as plt

labels = ['24h', '48h', '72h']
modes = { k:[] for k in ['nw', 'w0.25', 'w0.5', 'w1', 'w1.5', 'w3', 'w5', 'w7', 'w9'] }

def get_models_by_key(models, keys):
	results = {}
	for model, stats in models.items():
		if all([ k in stats['dataset_filename'] for k in keys]):
			results[model] = stats
	return results


fig = plt.figure(figsize=(14,7))
for n_plot, label in enumerate(labels):
	
	models = get_models()
	for k in modes.keys(): modes[k] = get_models_by_key(models, [label, k] )
	
	dots, xticks = [], []
	best_score, best_model = np.inf, None
	for i, (mode, models) in enumerate(modes.items()):
		xticks.append((i, mode))
		for model, stats in models.items():
			dots.append((i, stats['score']))
			if stats['score'] < best_score:
				best_score = stats['score']
				best_model = model
	
	ax = plt.subplot(2, 2, n_plot+1)
	plt.scatter([ d[0] for d in dots ], [ d[1] for d in dots ])
	plt.ylim(min([ d[1] for d in dots ])*0.97, max([ d[1] for d in dots ])*1.03)
	plt.xticks([ xt[0] for xt in xticks ], [ xt[1] for xt in xticks ]);
	ax.set_title('{} | {} | {:.7f}'.format(label, best_model, best_score))
