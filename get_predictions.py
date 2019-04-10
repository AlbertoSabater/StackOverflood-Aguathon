# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:21:44 2019

@author: Alberto
"""

import pandas as pd
import test_data
import json
import pickle


def main():
	model_dirs = {
					'H24': './models/24h_0305_1952_model_0/',
					'H48': './models/48h_0305_2145_model_1/',
					'H72': './models/72h_0305_2321_model_2/'
				}
	
	
	# %%
	
	data_filename = './ftp/ENTRADA/datos.csv'
	
	data, _ = test_data.get_dataset(data_filename)
	
	
	# %%
	
	results = {}
	for k,folder in model_dirs.items():
		print(k, folder)
		stats = json.load(open(folder + 'stats.json', 'r'))
		model = pickle.load(open(folder + 'xgb.pckl', 'rb'))
	
		k_data = data[stats['columns']]
		results[k] = model.predict(k_data)
		
		
	# %%
		
	res = pd.DataFrame(results, index=k_data.index)
	res.to_csv('./ftp/SALIDA/resultados.csv')
	
	
