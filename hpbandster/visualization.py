import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons,Button


def interactive_HB_plot(learning_curves, tool_tip_strings=None,log_y=False, log_x=False, reset_times=False, color_map='Set3', colors_floats=None, title='', show=True):

	times, losses, config_ids, = [], [], []

	for k,v in learning_curves.items():
		for l in v:
			tmp = list(zip(*l))
			times.append(tmp[0])
			losses.append(tmp[1])
			config_ids.append(k)

	num_curves = len(times)
	HB_iterations = [id[0] for id in config_ids]
	num_iterations = max(HB_iterations) + 1
	
	cmap = plt.get_cmap(color_map)
	

	
	if reset_times:
		times = [np.array(ts) - ts[0] for ts in times]
	
	
	if colors_floats is None:
		color_floats = []
		for i in range(num_curves):
			seed = 100*config_ids[i][0] + 10*config_ids[i][1] + config_ids[i][2]
			np.random.seed(seed)
			color_floats.append(np.random.rand())

	fig, ax = plt.subplots()
	
	lines = [[] for i in range(num_iterations)]

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
		
	check = CheckButtons(rax, range(num_iterations), [True for i in range(num_iterations)])
	
	none_button = Button(axnone, 'None')
	all_button = Button(axall, 'All')
	

	def change_visibility(label, value=None):
		if value is None:
			value = not lines[int(label)][0].get_visible()
		[l.set_visible(value) for l in lines[int(label)]]
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
