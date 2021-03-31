import os
import numpy as np
import matplotlib.pyplot as plt




def plot_singular_values(
	singular_values, title="", figsize=(12,8),
	path="CWD", filename=None, rank=None):
	"""
	Plots singular values of the SVD decomposition.

	:param ndarray singular_values: singular values.
	:param str title: if specified, title of the plot.
	:param tuple(int,int) figsize: size of the figure (width,height). \
		Default is (12,8).
	:param str path: if specified, the plot is saved at `path`. \
		Default is CWD.
	:param str filename: if specified, the plot is saved at `filename`. \
		Default is None.
	:param MPI_Rank rank: MPI rank for parallel SVD.
	"""
	if rank is not None:
		if rank == 0:
			plt.figure(figsize=figsize)
			plt.plot(singular_values)
			plt.legend()
			plt.title(title)
			plt.xlabel('Domain')
			plt.ylabel('U magnitude')

			# save or show plots
			if filename:
				if path == 'CWD': path = CWD
				plt.savefig(os.path.join(path, filename), dpi=200)
				plt.close()
			else:
				plt.show()



def plot_1D_modes(
	modes, idxs=[0], title="", figsize=(12,8),
	path="CWD", filename=None, rank=None, value='abs'):
	"""
	Plots modes of the SVD decomposition.

	:param ndarray modes: modes.
	:param str title: if specified, title of the plot.
	:param tuple(int,int) figsize: size of the figure \
		(width,height). Default is (12,8).
	:param str path: if specified, the plot is saved \
		at `path`. Default is CWD.
	:param str filename: if specified, the plot \
		is saved at `filename`. Default is None.
	:param MPI_Rank rank: MPI rank for parallel SVD.
	:param str value: whether to plot absolute \
		or real value of modes.
	"""

	if rank is not None:
		if rank == 0:
			plt.figure(figsize=figsize)
			if value.lower() == 'abs':
				for idx in idxs:
					plt.plot(np.abs(modes[:, idx]),
						label='mode '+str(idx))
			elif value.lower() == 'real':
				for idx in idxs:
					plt.plot(np.real(modes[:, idx]),
						label='mode '+str(idx))
			else:
				raise ValueError('`value` not recognized.')
			plt.legend()
			plt.title(title)
			plt.xlabel('Domain')
			plt.ylabel('U magnitude')

			# save or show plots
			if filename:
				if path == 'CWD': path = CWD
				plt.savefig(os.path.join(path, filename), dpi=200)
				plt.close()
			else:
				plt.show()
