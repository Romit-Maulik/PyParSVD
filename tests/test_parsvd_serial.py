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

	# modes
	if ParSVD.rank == 0:
		tol1 = 1e-10
		modes = ParSVD.modes
		assert((np.abs(modes[10,0])   < 0.00004899482868291135+tol1) & \
			   (np.abs(modes[10,0])   > 0.00004899482868291135-tol1))
		assert((np.abs(modes[100,0])  < 0.0004899482868291321 +tol1) & \
			   (np.abs(modes[100,0])  > 0.0004899482868291321 -tol1))
		assert((np.abs(modes[200,1])  < 0.0006310106406693729 +tol1) & \
			   (np.abs(modes[200,1])  > 0.0006310106406693729 -tol1))
		assert((np.abs(modes[300,3])  < 0.0004045576047363707 +tol1) & \
			   (np.abs(modes[300,3])  > 0.0004045576047363707 -tol1))
		assert((np.abs(modes[400,9])  < 0.00021061438553247674+tol1) & \
			   (np.abs(modes[400,9])  > 0.00021061438553247674-tol1))
		assert((np.abs(modes[7000,3]) < 0.005490309678018976  +tol1) & \
			   (np.abs(modes[7000,3]) > 0.005490309678018976  -tol1))
		assert((np.max(np.abs(modes)) < 0.0291383259914922    +tol1) & \
			   (np.max(np.abs(modes)) > 0.0291383259914922    -tol1))
		# singular values
		tol2 = 1e-6
		singular_values = ParSVD.singular_values
		assert((singular_values[0] < 393.04673638       +tol2) & \
			   (singular_values[0] > 393.04673638       -tol2))
		assert((singular_values[1] < 153.18879589       +tol2) & \
			   (singular_values[1] > 153.18879589       -tol2))
		assert((singular_values[2] <  84.94990906       +tol2) & \
			   (singular_values[2] >  84.94990906       -tol2))
		assert((singular_values[5] <  32.96249641       +tol2) & \
			   (singular_values[5] >  32.96249641       -tol2))
		assert((singular_values[9] <  15.341797626164364+tol2) & \
			   (singular_values[9] >  15.341797626164364-tol2))
		# getters
		assert(modes.shape      == (8192, 10))
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
