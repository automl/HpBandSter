import copy

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons,Button



def default_tool_tips(result_object, learning_curves, include_run_info=False):

	tool_tips = {}
	id2conf = result_object.get_id2config_mapping()
	
	for id in learning_curves.keys():
		
		config = id2conf[id]['config']
		config_info = id2conf[id]['config_info']

		all_runs =  result_object.get_runs_by_id(id)
		if len(all_runs) == 0:
			continue
		longest_run = all_runs[-1]
	
		while longest_run.loss is None:
			all_runs.pop()
			if len(all_runs) == 0: break
			longest_run = all_runs[-1]
			
		if len(all_runs) == 0: continue
		
		s = ['id: %s'%str(id), 'duration (sec): %f'%((longest_run['time_stamps']['finished'] - longest_run['time_stamps']['started']))]

		if not longest_run.loss is None:
			s += [str(k) + "=" +str(v) for k,v in sorted(id2conf[id]['config'].items()) ]
			try:
				s += [str(k) + "=" +str(v) for k,v in sorted(id2conf[id]['config_info'].items()) ]
			except:
				pass
			
			s += ['losses: {}'.format([r.loss for r in all_runs])]
			if include_run_info:
				s += ['longest run info: {}'.format(longest_run.info)]
			
		tool_tips[id] = "\n".join(s)
	return(tool_tips)

def concurrent_runs_over_time(runs, num_points = 512, show=False):
	
	data = np.array([(r.time_stamps['started'], r.time_stamps['finished']) for r in runs])
	ts = np.linspace(data.min(), data.max(), 512)
	n_workers = np.array([ ((data[:,0] <= t)*(data[:,1]>t)).sum() for t in ts])

	fig, ax = plt.subplots()
	ax.plot(ts, n_workers)
	ax.set_xlabel('time [s]')
	ax.set_ylabel('number of concurent runs')
	
	if show:
		plt.show()
	return(fig, ax)


def finished_runs_over_time(runs, show=False):
	budgets = set([r.budget for r in runs])
	
	times = {}
	for b in budgets:
		times[b] = [0]
	
	for r in runs:
		times[r.budget].append(r.time_stamps['finished'])
	
	for b in budgets:
		times[b].sort()
	
	
	fig, ax = plt.subplots()

	for b in budgets:
		ax.plot(times[b], np.arange(len(times[b])), label='b = %f'%b)


	ax.set_xlabel('time [s]')
	ax.set_ylabel('number of finished runs')
	ax.legend()
	
	if show:
		plt.show()
	return(fig,ax)

def performance_histogram_model_vs_random(runs, id2conf, show=False):
	model_based_runs = list(filter(lambda r: id2conf[r.config_id]['config_info']['model_based_pick'], runs))
	random_runs = list(filter(lambda r: not id2conf[r.config_id]['config_info']['model_based_pick'], runs))

	budgets = list(set([r.budget for r in runs]))
	budgets.sort()

	losses = {}
	for b in budgets:
		losses[b] = {'model_based': [], 'random': []}

	for r in model_based_runs:
		if r.loss is None or not np.isfinite(r.loss):
			continue
		losses[r.budget]['model_based'].append(r.loss)

	for r in random_runs:
		if r.loss is None or not np.isfinite(r.loss):
			continue
		losses[r.budget]['random'].append(r.loss)

	fig, axarr = plt.subplots(len(budgets), 2, sharey='row', sharex='row')
	plt.suptitle('Loss of model based configurations (left) vs. random configuration (right)')

	for i,b in enumerate(budgets):
		mbax, rax = axarr[i]
		mbax.hist(losses[b]['model_based'], label='b = %f \n n = %i'%(b,len(losses[b]['model_based'])))
		mbax.set_ylabel('frequency')
		mbax.legend()
		
		
		
		rax.hist(losses[b]['random'],label='b = %f \n n = %i'%(b,len(losses[b]['random'])))
		rax.legend()
		
		
		if i == len(budgets)-1:
			mbax.set_xlabel('loss')
			rax.set_xlabel('loss')
	if show:		
		plt.show()
		
	return(fig, axarr)


def correlation_across_budgets(results_object, show=False):
	
	runs = results_object.get_all_runs()
	id2conf = results_object.get_id2config_mapping()

	budgets = list(set([r.budget for r in runs]))
	budgets.sort()

	import itertools

	loss_pairs = {}
	for b in budgets[:-1]:
		loss_pairs[b] = {}

	for b1,b2 in itertools.combinations(budgets, 2):
		loss_pairs[b1][b2]= []

	for cid in id2conf.keys():
		runs = results_object.get_runs_by_id(cid)
		if len(runs) < 2: continue
		
		for r1,r2 in itertools.combinations(runs,2):
			if r1.loss is None or r2.loss is None: continue
			if not np.isfinite(r1.loss) or not np.isfinite(r2.loss): continue
			loss_pairs[float(r1.budget)][float(r2.budget)].append((r1.loss, r2.loss))
		
		

	rhos = np.eye(len(budgets)-1)
	rhos.fill(np.nan)

	ps = np.eye(len(budgets)-1)
	ps.fill(np.nan)

	for i in range(len(budgets)-1):
		for j in range(i+1,len(budgets)):
			spr = sps.spearmanr(loss_pairs[budgets[i]][budgets[j]])
			rhos[i][j-1] = spr.correlation
			ps[i][j-1] = spr.pvalue


	fig, ax = plt.subplots()

	cax = ax.matshow(rhos, vmin=-1, vmax=1)
	fig.colorbar(cax)


	ax.set_yticks( range(len(budgets)-1))
	ax.set_yticklabels(budgets[:-1],)

	ax.set_xticks( range(len(budgets)-1))
	ax.set_xticklabels(budgets[1:],)
	
	ax.set_title('Rank correlation of the loss across the budgets')

	for i in range(len(budgets)-1):
		for j in range(i+1,len(budgets)):
			plt.text(j-1,i, r'$\rho_{spearman}= %f$'%rhos[i][j-1] + '\n' + r'$p = %f$'%ps[i][j-1] + '\n' + r'$n = %i$'%len(loss_pairs[budgets[i]][budgets[j]]),
						horizontalalignment='center', verticalalignment='center')

	if show:
		plt.show()
	return(fig,ax)



def losses_over_time(runs, get_loss_from_run_fn = lambda r: r.loss, cmap = plt.get_cmap("tab10"), show=False):

	budgets = set([r.budget for r in runs])

	data = {}
	for b in budgets:
		data[b] = []

	for r in runs:
		if r.loss is None:
			continue
		b = r.budget
		t = r.time_stamps['finished']
		l = get_loss_from_run_fn(r)
		data[b].append((t,l))

	for b in budgets:
		data[b].sort()


	fig, ax = plt.subplots()

	for i, b in enumerate(budgets):
		data[b] = np.array(data[b])
		ax.scatter(data[b][:,0], data[b][:,1], color=cmap(i), label='b=%f'%b)
		
		ax.step(data[b][:,0], np.minimum.accumulate(data[b][:,1]), where='post')

	ax.set_title('Losses for different budgets over time')
	ax.set_xlabel('wall clock time [s]')
	ax.set_ylabel('loss')
	ax.legend()
	if show:
		plt.show()
	return(fig,ax)





def interactive_HBS_plot(learning_curves, tool_tip_strings=None,log_y=False, log_x=False, reset_times=False, color_map='Set3', colors_floats=None, title='', show=True):

	times, losses, config_ids, = [], [], []

	for k,v in learning_curves.items():
		for l in v:
			if len(l) == 0: continue
			tmp = list(zip(*l))
			try:
				times.append(tmp[0])
				losses.append(tmp[1])
				config_ids.append(k)
			except:
				import pdb; pdb.set_trace()



	num_curves = len(times)
	HB_iterations = [id[0] for id in config_ids]
	
	num_iterations = len(set(HB_iterations))
	
	cmap = plt.get_cmap(color_map)
	

	
	if reset_times:
		times = [np.array(ts) - ts[0] for ts in times]
	
	
	if colors_floats is None:
		color_floats = []
		for i in range(num_curves):
			seed = 100*np.abs(config_ids[i][0]) + 10*config_ids[i][1] + config_ids[i][2]
			np.random.seed(seed)
			color_floats.append(np.random.rand())

	fig, ax = plt.subplots()
	
	lines = [[] for i in range(num_iterations)]

	iteration_labels  = list(range(num_iterations))
	if HB_iterations[-1] == -1:
		iteration_labels[-1] = 'warmstart data'
	
	

	all_lines = []
	
	for i in range(num_curves):
		l, = ax.plot(times[i], losses[i], color=cmap(color_floats[i]), marker='o', gid=i, picker=True)
		lines[HB_iterations[i]].append(l)
		all_lines.append(l)

	if log_y:
		plt.yscale('log')

	ax.set_title(title)

	hover_annotation = ax.annotate("BLABLA", xy=(0,0), xytext=(20,20),textcoords="offset points",
					bbox=dict(boxstyle="round", fc="w"),
					arrowprops=dict(arrowstyle="->"))
	hover_annotation.set_visible(False)


	permanent_annotations = {}


	plt.subplots_adjust(left=0.2)
	rax = plt.axes([0.05, 0.1, 0.1, 0.8])

	
	axnone = plt.axes([0.05, 0, 0.05, 0.1])
	axall = plt.axes([0.1, 0, 0.05  , 0.1])
		
	check = CheckButtons(rax, iteration_labels, [True for i in range(num_iterations)])
	
	none_button = Button(axnone, 'None')
	all_button = Button(axall, 'All')
	

	def change_visibility(label, value=None):
		if label == 'warmstart data':
			index = -1
		else:
			index = int(label)
		
		if value is None:
			value = not lines[index][0].get_visible()
		[l.set_visible(value) for l in lines[index]]
		plt.draw()

		
	def show_all(event):
		for label in range(num_iterations):
			if not lines[label][0].get_visible():
				change_visibility(label, False)
				check.set_active(label)

	def hide_all(event):
		for label in range(num_iterations):
			if lines[label][0].get_visible():
				change_visibility(label, True)
				check.set_active(label)
			
	check.on_clicked(change_visibility)
	none_button.on_clicked(hide_all)
	all_button.on_clicked(show_all)


	def update_hover_annotation(line, annotation=hover_annotation):
		xdata = line.get_xdata()
		ydata = line.get_ydata()
		idx = line.get_gid()        
		annotation.xy = (xdata[-1], ydata[-1])

		if not tool_tip_strings is None:
			annotation.set_text(tool_tip_strings[config_ids[idx]])
		else:
			annotation.set_text(str(config_ids[idx]))

		fig.canvas.draw_idle()

	def onpick1(event):

		line = event.artist

		if not line in all_lines:
			return
		
		gid = line.get_gid()

		if gid in permanent_annotations:
			# remove permanent annotation
			permanent_annotations[gid].set_visible(False)
			del permanent_annotations[gid]
			fig.canvas.draw_idle()

		else:
			# add a new annotation

			xdata = line.get_xdata()
			ydata = line.get_ydata()
			idx = line.get_gid()

			if not tool_tip_strings is None:
				text = tool_tip_strings[config_ids[idx]]
			else:
				text = str(config_ids[idx])
			
			permanent_annotations[gid] = ax.annotate(text, copy.deepcopy((xdata[-1],ydata[-1])),
							xytext=(20,20),textcoords="offset points",
							bbox=dict(boxstyle="round", fc="w", linewidth=3),
							arrowprops=dict(arrowstyle="->"))
			permanent_annotations[gid].draggable()
			fig.canvas.draw_idle()            

	def hover(event):
		vis = hover_annotation.get_visible()
		if event.inaxes == ax:
			active_lines = list(filter(lambda l: l.contains(event)[0], all_lines))
			if len(active_lines) > 0:
				hover_annotation.set_visible(True)
				update_hover_annotation(active_lines[0])
			elif vis:
				hover_annotation.set_visible(False)
				fig.canvas.draw_idle()



	fig.canvas.mpl_connect('pick_event', onpick1)
	fig.canvas.mpl_connect("motion_notify_event", hover)


	if show:
		plt.show()
	return(fig, ax, check, none_button, all_button)
