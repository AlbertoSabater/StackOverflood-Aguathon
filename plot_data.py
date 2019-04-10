# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:00:21 2019

@author: Alberto
"""

import pandas as pd


dataset_filename = './ftp/ENTRADA/datos.csv'
df = pd.read_csv(dataset_filename, index_col=0)
df.index = pd.to_datetime(df.index)

label_columns = ['pred_24h', 'pred_48h', 'pred_72h']
data_columns = ['ALAGON_NR', 'GRISEN_NR', 'NOVILLAS_NR', 'TAUSTE_NR',  'TUDELA_NR', 'ZGZ_NR']

data = df.loc[:, data_columns]
del df


# %%

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot

init_notebook_mode(connected=True)

p = []
for c in data_columns: p.append(go.Scatter(x=data.index, y=data[c], name=c))
#p = [go.Scatter(x=data.index, y=data[data_columns[0]])]
plot(p, filename = 'time-series-simple')



