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


root = 'CountingOnes_1e-3/'
root = 'CountingOnes/'
datasets = [root+'4_4', root+'8_8', root+'16_16', root+'32_32']
#datasets +=['counting_ones/dims_8_8', 'counting_ones/dims_16_16', 'counting_ones/dims_32_32']#, 'counting_ones/64_64']
methods  = ['tpe', 'smac', 'randomsearch']


methods += ['bohb_run_%i'%i for i in [391479, 391480, 391481, 391482, 391542, 391657]]
methods += ['hyperband_run_%i'%i for i in [391474, 391475, 391476, 391478]]



print(methods)

all_trajectories = {}

for d in datasets:
	all_trajectories[d]={}


# loading mine
for d in datasets:
	dim = 2*int(d.split('_')[-1])
	
	directory = '../results/'+d
	
	
	for m in methods:
		dfs = []
				
		for fn in glob.glob(directory+'/'+m+'*.pkl'):
			try:
				with open(fn, 'rb') as fh:
					datum = pickle.load(fh)
				

				#print(m, datum['cummulative_budget'][-1],datum['budgets'][-1])
				#if m in ['bohb', 'hyperband', 'randomsearch']:
				try:
					times = np.array(datum['cummulative_budget'])/datum['HB_config']['max_budget']
				except: 
					times = np.array(datum['cummulative_budget'])/datum['budgets'][-1]
				#df = pd.DataFrame({fn: np.array(datum['losses'])/(dim)+1} , index = times)
				df = pd.DataFrame({fn: np.array(datum['test_losses'])/(dim)+1} , index = times)
				dfs.append(df)

			except:
				embed()
				#raise
				pass
		
		df  = merge_and_fill_trajectories(dfs)
		print(d,m,df.shape)
	
	
		all_trajectories[d][m] = {
						'time_stamps' : np.array(df.index),
						'losses': np.array(df.T)
		}




# renaming to convert names for plotting scripts


save_pgf_data(all_trajectories, reduce_points=512, clip=(1e-6, 1))

for d in datasets:
	plot_losses(all_trajectories[d], '%s'%d, regret=False, show=False, plot_mean=True)

plt.show()
embed()
