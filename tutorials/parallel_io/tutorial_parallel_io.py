import os
import sys
import time
import numpy as np

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join("../../"))
from pyparsvd.parsvd_serial   import ParSVD_Serial
from pyparsvd.parsvd_parallel import ParSVD_Parallel

# Construct SVD objects
ParSVD = ParSVD_Parallel(K=10, ff=1.0, low_rank=True)

# Path to data
path = os.path.join(CFD, './data/')

# Batchwise data - note these are h5 files and do not need the "np.load"
initial_data = os.path.join(path, 'Batch_0_data.h5')
new_data = os.path.join(path, 'Batch_1_data.h5')
newer_data = os.path.join(path, 'Batch_2_data.h5')
newest_data = os.path.join(path, 'Batch_3_data.h5')

# Do first modal decomposition -- Parallel
s = time.time()
ParSVD.initialize(initial_data,dataset='dataset')

# Incorporate new data -- Parallel
ParSVD.incorporate_data(new_data,dataset='dataset')
ParSVD.incorporate_data(newer_data,dataset='dataset')
ParSVD.incorporate_data(newest_data,dataset='dataset')
if ParSVD.rank == 0: print('Elapsed time PARALLEL: ', time.time() - s, 's.')

# Basic postprocessing
if ParSVD.rank == 0:

    # Save results
    ParSVD.save()

    # Visualize singular values and modes modes
    ParSVD.plot_singular_values(filename='parallel_sv.png')
    ParSVD.plot_1D_modes(filename='parallel_1d_mode0.png')
    ParSVD.plot_1D_modes(filename='parallel_1d_mode2.png', idxs=[2])
