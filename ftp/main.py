# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 20:05:38 2019

@author: Alberto
"""

import pandas as pd
import pickle
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import time



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


def main():
	models = {
				'H24': {'model_folder': 'models/24h_0413_0359_model_86/'},
				'H48': {'model_folder': 'models/48h_0413_0508_model_94/'},
				'H72': {'model_folder': 'models/72h_0413_0604_model_100/'},
			}
	preds = {}
	
	
	dataset_filename = './ENTRADA/datos.csv'
	data = pd.read_csv(dataset_filename, index_col=0)
	data.index = pd.to_datetime(data.index)
	data_columns = ['ALAGON_NR', 'GRISEN_NR', 'NOVILLAS_NR', 'TAUSTE_NR',  'TUDELA_NR', 'ZGZ_NR']
	data = data.loc[:, data_columns]
	means = pickle.load(open('models/' + 'means_sd4_sh3_w0.5.pckl', 'rb'))
	
	
	columns = []
	for label, model in models.items():
		stats = json.load(open(model['model_folder'] + 'stats.json', 'r'))
		columns += stats['columns']
		models[label]['stats'] = stats
	columns = list(set(columns))
	
	print('Building prediction dataset')
	new_columns = Parallel(n_jobs=-1, max_nbytes=None)(delayed(get_column_data)(col, data, means) 
			for col in tqdm(columns, total=len(columns)))
	data = pd.DataFrame(dict(new_columns))[columns]
	
	
	for label, model in models.items():
		print('Predicting:', label)
		t = time.time()
		model_l = pickle.load(open(model['model_folder'] + 'model.pckl', 'rb'))
		preds[label] = model_l.predict(data[model['stats']['columns']])
		print('Time elapsed in prediction: {:.2f} mins.'.format((time.time()-t)/60))
		
	
	res = pd.DataFrame(preds, index=data.index)
	res.to_csv('./SALIDA/resultados.csv')
	print('Predictions stored')


if __name__ == '__main__':
	main()	

	
	
	
