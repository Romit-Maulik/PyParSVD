import os
import sys
import shutil
import pytest
import subprocess
import xarray as xr
import numpy  as np

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join(CFD, "../"))
sys.path.append(os.path.join(CFD, "../pyparsvd"))
from pyparsvd.parsvd_parallel import ParSVD_Parallel
from pyparsvd.parsvd_serial   import ParSVD_Serial



def test_serial_svd():
	ParSVD = ParSVD_Serial(K=10, ff=1.0, low_rank=True)
	initial_data = np.load(os.path.join(CFD, 'data', 'Batch_0_data.npy'))
	new_data = np.load(os.path.join(CFD, 'data', 'Batch_1_data.npy'))
	newer_data = np.load(os.path.join(CFD, 'data', 'Batch_2_data.npy'))
	newest_data = np.load(os.path.join(CFD, 'data', 'Batch_3_data.npy'))

	# Do first modal decomposition
	ParSVD.initialize(initial_data)

	# Incorporate new data
	ParSVD.incorporate_data(new_data)
	ParSVD.incorporate_data(newer_data)
	ParSVD.incorporate_data(newest_data)

	if ParSVD.rank == 0:
		# modes
		tol1 = 1e-10
		modes = ParSVD.modes
		assert((np.abs(modes[10,0])   < 0.0003926734521274234+tol1) & \
			   (np.abs(modes[10,0])   > 0.0003926734521274234-tol1))
		assert((np.abs(modes[100,0])  < 0.0039267345212739375+tol1) & \
			   (np.abs(modes[100,0])  > 0.0039267345212739375-tol1))
		assert((np.abs(modes[200,1])  < 0.005071521533114417 +tol1) & \
			   (np.abs(modes[200,1])  > 0.005071521533114417 -tol1))
		assert((np.abs(modes[300,3])  < 0.00325359321603978  +tol1) & \
			   (np.abs(modes[300,3])  > 0.00325359321603978  -tol1))
		assert((np.abs(modes[400,9])  < 0.001689203733107208 +tol1) & \
			   (np.abs(modes[400,9])  > 0.001689203733107208 -tol1))
		assert((np.abs(modes[1500,5]) < 0.045545239394491364 +tol1) & \
			   (np.abs(modes[1500,5]) > 0.045545239394491364 -tol1))
		assert((np.max(np.abs(modes)) < 0.05819275466587741  +tol1) & \
			   (np.max(np.abs(modes)) > 0.05819275466587741  -tol1))
		# singular values
		tol2 = 1e-6
		singular_values = ParSVD.singular_values
		assert((singular_values[0] < 98.2084703446786  +tol2) & \
			   (singular_values[0] > 98.2084703446786  -tol2))
		assert((singular_values[1] < 38.356267047792514+tol2) & \
			   (singular_values[1] > 38.356267047792514-tol2))
		assert((singular_values[2] < 21.282220344811638+tol2) & \
			   (singular_values[2] > 21.282220344811638-tol2))
		assert((singular_values[5] < 8.264440003962125 +tol2) & \
			   (singular_values[5] > 8.264440003962125 -tol2))
		assert((singular_values[9] < 3.851852948902795 +tol2) & \
			   (singular_values[9] > 3.851852948902795 -tol2))
		# getters
		assert(modes.shape      == (2048, 10))
		assert(ParSVD.K         == 10)
		assert(ParSVD.ff        == 1.0)
		assert(ParSVD.low_rank  == True)
		assert(ParSVD.iteration == 0)
		assert(ParSVD.n_modes   == 10)

		# Save results
		ParSVD.save()

		# Visualize modes
		ParSVD.plot_1D_modes(filename='parallel_1d_modes.png')



if __name__ == '__main__':
	test_serial_svd()
