#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 10:41:33 2019

@author: asabater
"""

import os
import json
import numpy as np


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

label ='24h'
modes = { k:[] for k in ['nw', 'w3'] }
models = get_models()


def get_models_by_key(models, keys):
	results = {}
	for model, stats in models.items():
		print(keys, model, stats['dataset_filename'])
		if all([ k in stats['dataset_filename'] for k in keys]):
			results[model] = stats
	return results


for k in modes.keys():
	modes[k] = get_models_by_key(models, [label, k] )


dots = []
xticks = []
for i, (mode, models) in enumerate(modes.items()):
	xticks.append((i, mode))
	for model, stats in models.items():
#		print(i, stats['score'], stats['dataset_filename'])
		dots.append((i, stats['score']))
#		plt.scatter(i, stats['score'])

plt.scatter([ d[0] for d in dots ], [ d[1] for d in dots ])
plt.ylim(min([ d[1] for d in dots ])*0.999, max([ d[1] for d in dots ])*1.001)
plt.xticks([ xt[0] for xt in xticks ], [ xt[1] for xt in xticks ])

