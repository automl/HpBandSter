import os
import glob
import json
import pickle

import numpy as np
import pandas as pd
#import matplotlib

import matplotlib.pyplot as plt

from util import get_incumbent_trajectories, plot_losses, fill_trajectories, save_pgf_data


from IPython import embed

datasets = ['adult','higgs', 'letter', 'mnist', 'optdigits', 'poker']
methods = ['bohb', 'hyperband', 'smac', 'tpe', 'robo']

all_trajectories = {}

for d in datasets:
	all_trajectories[d]={}



for d in datasets:
	directory = '../results/ParamNetSurrogates/'+d
	
	for m in methods:
		
		df = pd.DataFrame()
				
		for fn in glob.glob(directory+'/'+m+'*.pkl'):
			with open(fn, 'rb') as fh:
				datum = pickle.load(fh)
			

			times = np.array(datum['cummulative_budget'])
			#tmp = pd.DataFrame({fn: datum['losses']} , index = times)
			
			tmp = pd.DataFrame({fn: datum['losses']} , index = times)
			df = df.join(tmp, how='outer')
		
		
		df  = fill_trajectories(df)

		if df.empty:
			continue
			
		print(d,m,df.shape)

		all_trajectories[d][m] = {
						'time_stamps' : np.array(df.index),
						'losses': np.array(df.T)
		}




# renaming to convert names for plotting scripts


#save_pgf_data(all_trajectories, reduce_points=512, clip=(1e-6, 1))

for d in datasets:
	plot_losses(all_trajectories[d], '%s'%d, regret=True, show=False, plot_mean=True)

plt.show()
embed()
