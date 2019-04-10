# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:03:39 2019

@author: Alberto
"""

import pandas as pd
from fbprophet import Prophet


dataset_filename = './ftp/ENTRADA/datos.csv'
df = pd.read_csv(dataset_filename, index_col=0)
df.index = pd.to_datetime(df.index)

label_columns = ['pred_24h', 'pred_48h', 'pred_72h']
data_columns = ['ALAGON_NR', 'GRISEN_NR', 'NOVILLAS_NR', 'TAUSTE_NR',  'TUDELA_NR', 'ZGZ_NR']

data = df.loc[:, data_columns]
del df


# %%

df = data[data.index.year == 2018]['ZGZ_NR'].reset_index()
df.columns = ['y', 'ds']
df.head()


import numpy as np

df['ds'] = np.log(df['ds'])
df = df.fillna(0)
m = Prophet(changepoint_prior_scale=0.01).fit(df)


future = m.make_future_dataframe(periods=365, freq='H')
forecast = m.predict(future)
fig2 = m.plot_components(forecast)


