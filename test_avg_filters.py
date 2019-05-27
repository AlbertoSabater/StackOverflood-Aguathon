#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 09:50:22 2019

@author: asabater
"""

import matplotlib.pyplot as plt
#from numpy.fft import fft, fftshift
import numpy as np

window = np.hamming(7)
#window[2] = 0
window /= sum(window)
plt.plot(window)


# %%

import matplotlib.pyplot as plt
from scipy import signal

#window = signal.get_window('triang', 7)
window = signal.gaussian(7, 0.5)
#window[1] = 0
window /= sum(window)
plt.plot(window);


# %%

# TODO: check index de datasets

import pickle

nw24 = pickle.load(open('datasets/XY_0.15_pred_24h_158_nw.pckl', 'rb'))
nw48 = pickle.load(open('datasets/XY_0.15_pred_48h_158_nw.pckl', 'rb'))
nw72 = pickle.load(open('datasets/XY_0.15_pred_72h_158_nw.pckl', 'rb'))

w3_24 = pickle.load(open('datasets/XY_0.15_pred_24h_158_w3.pckl', 'rb'))
w3_48 = pickle.load(open('datasets/XY_0.15_pred_48h_158_w3.pckl', 'rb'))
w3_72 = pickle.load(open('datasets/XY_0.15_pred_72h_158_w3.pckl', 'rb'))

#print(all(nw24[0].index == nw48[0].index == nw72[0].index == w3_24[0].index == w3_48[0].index == w3_72[0].index))


# %%

import pickle

w075_sh1 = pickle.load(open('datasets/XY_0.15_pred_72h_158_w0.75_sd4_sh1.pckl', 'rb'))
w075_sh3 = pickle.load(open('datasets/XY_0.15_pred_72h_158_w0.75_sd4_sh3.pckl', 'rb'))
#w075 = pickle.load(open('datasets/XY_0.15_pred_72h_158_w0.75.pckl', 'rb'))


print(all(w075_sh1[0].index == w075_sh3[0].index))

print(all(w075_sh3[0].fillna(0) == w075_sh3[0].fillna(0)))


# %%

import os
import json

for model_folder in os.listdir('models'):
	stats = json.load(open('models/' + model_folder + '/stats.json'))
	dataset_filename = stats['dataset_filename']
#	if 'sd' in dataset_filename and 'sh' in dataset_filename:
#		print(dataset_filename)
#		stats['dataset_filename'] = dataset_filename.replace('.pckl', '_sd4_sh3.pckl')
#		json.dump(stats, open('models/' + model_folder + '/stats.json', 'w'))
	if len([ None for df in dataset_filename.split('_') if 'sd' in df ]) > 1:
		print(len([ None for df in dataset_filename.split('_') if 'sd' in df ]))
		print(dataset_filename)
		stats['dataset_filename'] = dataset_filename.replace('_sd4_sh3.pckl', '.pckl')
		print(stats['dataset_filename'])
		json.dump(stats, open('models/' + model_folder + '/stats.json', 'w'))
		
 

# %%

for dataset in os.listdir('datasets'):
	if 'XY' not in dataset: continue
	else:
		if not 'sd' in dataset and not 'sh' in dataset:
			print(dataset)
#			os.rename('datasets/' + dataset, 'datasets/' + dataset.replace('.pckl', '_sd4_sh3.pckl'))
			
			
			
			
			