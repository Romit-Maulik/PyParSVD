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

	# Save results
	ParSVD.save()

	# Visualize modes
	ParSVD.plot_1D_modes()



if __name__ == '__main__':
	test_parallel_svd()
