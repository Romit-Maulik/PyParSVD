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



@pytest.mark.mpi
def test_parallel_svd():
	iteration = 0
	ParSVD = ParSVD_Parallel(K=10, ff=1.0, low_rank=True)
	filename = 'points_rank_' + str(ParSVD.rank) + \
		'_batch_' + str(iteration) + '.npy'
	pathname = os.path.join(CFD, 'data', filename)
	data = np.load(pathname)

	# Do first modal decomposition
	ParSVD.initialize(data)

    # Incorporate new data
	for iteration in range(1, 4):
		filename = 'points_rank_' + str(ParSVD.rank) + \
			'_batch_' + str(iteration) + '.npy'
		pathname = os.path.join(CFD, 'data', filename)
		data = np.load(pathname)
		ParSVD.incorporate_data(data)

	if ParSVD.rank == 0:
		# modes
		tol1 = 1e-6
		modes = ParSVD.modes
		assert((np.abs(modes[10,0])   < 0.0003926734521274234+tol1) & \
			   (np.abs(modes[10,0])   > 0.0003926734521274234-tol1))
		assert((np.abs(modes[100,0])  < 0.003926734359882622 +tol1) & \
			   (np.abs(modes[100,0])  > 0.003926734359882622 -tol1))
		assert((np.abs(modes[200,1])  < 0.005071520463630745 +tol1) & \
			   (np.abs(modes[200,1])  > 0.005071520463630745 -tol1))
		assert((np.abs(modes[300,3])  < 0.0032535864808027884+tol1) & \
			   (np.abs(modes[300,3])  > 0.0032535864808027884-tol1))
		assert((np.abs(modes[400,9])  < 0.0016891690449827355+tol1) & \
			   (np.abs(modes[400,9])  > 0.0016891690449827355-tol1))
		assert((np.abs(modes[1500,5]) < 0.045545206908721145 +tol1) & \
			   (np.abs(modes[1500,5]) > 0.045545206908721145 -tol1))
		assert((np.max(np.abs(modes)) < 0.05819181847485311  +tol1) & \
			   (np.max(np.abs(modes)) > 0.05819181847485311  -tol1))
		# singular values
		tol2 = 1e-6
		singular_values = ParSVD.singular_values
		assert((singular_values[0] < 98.2084703446786  +tol2) & \
			   (singular_values[0] > 98.2084703446786  -tol2))
		assert((singular_values[1] < 38.356267047792514+tol2) & \
			   (singular_values[1] > 38.356267047792514-tol2))
		assert((singular_values[2] < 21.2822217532692+tol2) & \
			   (singular_values[2] > 21.2822217532692-tol2))
		assert((singular_values[5] < 8.264438631871547 +tol2) & \
			   (singular_values[5] > 8.264438631871547 -tol2))
		assert((singular_values[9] < 3.851847153126575 +tol2) & \
			   (singular_values[9] > 3.851847153126575 -tol2))
		# getters
		assert(modes.shape      == (2048, 10))
		assert(ParSVD.K         == 10)
		assert(ParSVD.ff        == 1.0)
		assert(ParSVD.low_rank  == True)
		assert(ParSVD.iteration == 3)
		assert(ParSVD.n_modes   == 10)

		# Save results
		ParSVD.save()

		# Visualize modes
		ParSVD.plot_1D_modes(filename='parallel_1d_modes.png')



if __name__ == '__main__':
	test_parallel_svd()
