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
    ParSVD = ParSVD_Serial(K=10, ff=1.0, low_rank=False)
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

	# Visualize modes
    ParSVD.plot_1D_modes()



if __name__ == '__main__':
	test_serial_svd()
