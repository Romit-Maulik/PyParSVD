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



def plot_2D_modes(
	modes, num_rows, num_cols, num_ranks, idxs=[0], title="", figsize=(12,8),
	path="CWD", filename=None, rank=None, value='abs'):
	"""
	Plots modes of the SVD decomposition.

	:param ndarray modes: modes.
	:param int num_rows: number of rows (dimension 2) in the 3D dataset.
	:param int num_cols: number of columns (dimension 3) in the 3D dataset.
	:param int num_ranks: number of ranks for parallel SVD
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
				num_cols_rank = int(num_cols/num_ranks)
				dpr = num_rows*num_cols_rank

				for idx in idxs:
					plot_data = np.abs(modes[:dpr, idx]).reshape(num_rows,-1)
					for rank in range(1,num_ranks):
						if rank !=num_ranks-1:
							temp_data = np.abs(modes[rank*dpr:(rank+1)*dpr, idx]).reshape(num_rows,-1)
							plot_data = np.concatenate((plot_data,temp_data),axis=-1) # Stencil decomposition
						else:
							temp_data = np.abs(modes[rank*dpr:, idx]).reshape(num_rows,-1)
							plot_data = np.concatenate((plot_data,temp_data),axis=-1) # Stencil decomposition

					plt.imshow(plot_data)
						
			elif value.lower() == 'real':
				num_cols_rank = int(num_cols/num_ranks)
				dpr = num_rows*num_cols_rank
				for idx in idxs:
					plot_data = np.real(modes[:dpr, idx]).reshape(num_rows,-1)
					for rank in range(1,num_ranks):
						if rank !=num_ranks-1:
							temp_data = np.real(modes[rank*dpr:(rank+1)*dpr, idx]).reshape(num_rows,-1)
							plot_data = np.concatenate((plot_data,temp_data),axis=-1) # Stencil decomposition
						else:
							temp_data = np.real(modes[rank*dpr:, idx]).reshape(num_rows,-1)
							plot_data = np.concatenate((plot_data,temp_data),axis=-1) # Stencil decomposition

					plt.imshow(plot_data)

			else:
				raise ValueError('`value` not recognized.')
			plt.legend()
			plt.title(title)
			plt.xlabel('X')
			plt.ylabel('Y')

			# save or show plots
			if filename:
				if path == 'CWD': path = CWD
				plt.savefig(os.path.join(path, filename), dpi=200)
				plt.close()
			else:
				plt.show()
