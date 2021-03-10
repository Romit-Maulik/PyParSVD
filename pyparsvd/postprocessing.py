import os
import numpy as np
import matplotlib.pyplot as plt



def plot_1D_modes(
	modes, idxs=[0], title="", figsize=(12,8),
	path="CWD", filename=None, rank=None, value='abs'):

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
