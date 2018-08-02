import os
import glob
import json
import pickle

import numpy as np
import pandas as pd
#import matplotlib

import matplotlib.pyplot as plt

from util import get_incumbent_trajectories, plot_losses, merge_and_fill_trajectories, save_pgf_data


from IPython import embed

datasets = ['SVMSurrogate']
methods = ['bohb', 'entropy_search', 'ei', 'mtbo_4', 'fabolas', 'hyperband', 'randomsearch']

all_trajectories = {}

for d in datasets:
	all_trajectories[d]={}




for d in datasets:
	directory = '../data/svm_surrogate/'
	
	for m in methods:
		
		dfs = []
				
		for fn in glob.glob(directory+'/'+m+'*.pkl'):
			with open(fn, 'rb') as fh:
				datum = pickle.load(fh)
			
			times = np.array(datum['cummulative_cost'])
			df = pd.DataFrame({fn: datum['losses']} , index = times)
			#tmp = pd.DataFrame({fn: datum['losses']} , index = times)
			
			dfs.append(df)
		
		df  = merge_and_fill_trajectories(dfs, default_value=0.9)
		
		if df.empty:
			continue

		print(d,m,df.shape)

	
		all_trajectories[d][m] = {
						'time_stamps' : np.array(df.index),
						'losses': np.array(df.T)
		}





for d in datasets:
	directory = '../results/'+d

	for m in methods:
		dfs = []		
		for fn in glob.glob(directory+'/'+m+'*.pkl'):
			with open(fn, 'rb') as fh:
				datum = pickle.load(fh)
			
			if m in ['bohb', 'hyperband']:
				#embed()
				pass
			times = np.array(datum['cummulative_cost'])
			#tmp = pd.DataFrame({fn: datum['losses']} , index = times)
			tmp = pd.DataFrame({fn: datum['test_losses']} , index = times)
			dfs.append(tmp)

		df  = merge_and_fill_trajectories(dfs, default_value=0.9)

		if df.empty:
			continue
			
		print(d,m,df.shape)

		all_trajectories[d][m] = {
						'time_stamps' : np.array(df.index),
						'losses': np.array(df.T)
		}




# renaming to convert names for plotting scripts
#save_pgf_data(all_trajectories, reduce_points=512, clip=(1e-2, 1))

for d in datasets:
	plot_losses(all_trajectories[d], '%s'%d, regret=False, show=False, plot_mean=True)

plt.show()
embed()
