# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:43:35 2019

@author: Alberto
"""

import pandas as pd
import numpy as np
import pickle


# TODO
# Tauste tiene nans


def get_dataset(dataset_filename, return_labels=True):
	df = pd.read_csv(dataset_filename, index_col=0)
	df.index = pd.to_datetime(df.index)
	
	label_columns = ['pred_24h', 'pred_48h', 'pred_72h']
	data_columns = ['ALAGON_NR', 'GRISEN_NR', 'NOVILLAS_NR', 'TAUSTE_NR',  'TUDELA_NR', 'ZGZ_NR']
	
	data = df.loc[:, data_columns]
	if return_labels: labels = df.loc[:, label_columns]
	else: labels = None
	del df
	
	
	# %%
	
	# Hour, day, month
	
	data = data.assign(day = data.index.day)
	data = data.assign(month = data.index.month)
	data = data.assign(year = data.index.year)
	
	
	# %%
	
	# 1-24 hours ago
	
	for h in list(range(1,24,1)) + [36, 60, 84]:
		print('Adding {} hours ago'.format(h))
		inds =  data.index - pd.DateOffset(hours=h)
		for col in data_columns:
			data = data.assign(**{'{}-h{}'.format(col, h): data.loc[inds, col].values})
			
	
	# %%
	
	# 1-7 days ago
	
	for d in range(1,8,1):
		print('Adding {} days ago'.format(d))
		inds =  data.index - pd.DateOffset(days=d)
		for col in data_columns:
			data = data.assign(**{'{}-d{}'.format(col, d): data.loc[inds, col].values})
	
	
	# %%
	
	# 7,14,28, 35 days ago
			
	for d in [14, 28, 35]:
		print('Adding {} days ago'.format(d))
		inds =  data.index - pd.DateOffset(days=d)
		for col in data_columns:
			data = data.assign(**{'{}-d{}'.format(col, d): data.loc[inds, col].values})
			
	
	# %%
	
	
	# 1-13 months ago
			
	for m in range(1, 13,1):
		print('Adding {} months ago'.format(m))
		inds =  data.index - pd.DateOffset(months=m)
		for col in data_columns:
			data = data.assign(**{'{}-m{}'.format(col, m): data.loc[inds, col].values})
		
	
# %%
			
	return data, labels

# %%
		
# Year, month, week, day trend?


# %%
		

def main(dataset_filename, validation_split):
	data, labels = get_dataset(dataset_filename, return_labels=True)
	
	
	data.to_pickle('datasets/X_{}_{}.pckl'.format(validation_split, len(data.columns)))
	for c in labels.columns:
		labels[[c]].to_pickle('datasets/Y_{}_{}.pckl'.format(validation_split, len(c)))
		
	
	num_val = int(len(data) * validation_split)
	num_train = len(data) - num_val
	
	inds = np.random.permutation(data.index)
	inds_train = inds[:num_train]
	inds_val = inds[num_train:]
	
	
	#data.loc[inds_train].to_pickle('datasets/X_train_{}_{}.pckl'.format(validation_split, len(data.columns)))
	#data.loc[inds_val].to_pickle('datasets/X_val_{}_{}.pckl'.format(validation_split, len(data.columns)))
	#for c in label_columns:
	#	labels.loc[inds_train, c].to_pickle('datasets/Y_train_{}_{}_{}.pckl'.format(validation_split, c, len(data.columns)))
	#	labels.loc[inds_val, c].to_pickle('datasets/Y_val_{}_{}_{}.pckl'.format(validation_split, c, len(data.columns)))
	
	
	X_train = data.loc[inds_train]
	X_val = data.loc[inds_val]
	for c in labels.columns:
		pickle.dump([
					X_train, X_val, labels.loc[inds_train, c], labels.loc[inds_val, c]
				],
				open('datasets/XY_{}_{}_{}.pckl'.format(validation_split, c, len(data.columns)), 'wb')
			)

if __name__ == '__main__':
	dataset_filename = './ftp/ENTRADA/datos.csv'
	validation_split = 0.2
	main(dataset_filename, validation_split)



