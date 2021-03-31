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
from pyparsvd.parsvd_parallel import ParSVD_Parallel

# Path to data
path = os.path.join(CFD, './data/')

# Do first modal decomposition -- Parallel
ParSVD = ParSVD_Parallel(K=10, ff=1.0, low_rank=True)

# Parallel data
initial_data_par = np.load(os.path.join(path, 'points_rank_' + str(ParSVD.rank) + '_batch_0.npy'))
new_data_par = np.load(os.path.join(path, 'points_rank_' + str(ParSVD.rank) + '_batch_1.npy'))
newer_data_par = np.load(os.path.join(path, 'points_rank_' + str(ParSVD.rank) + '_batch_2.npy'))
newest_data_par = np.load(os.path.join(path, 'points_rank_' + str(ParSVD.rank) + '_batch_3.npy'))

s = time.time()
ParSVD.initialize(initial_data_par)

# Incorporate new data -- Parallel
ParSVD.incorporate_data(new_data_par)
ParSVD.incorporate_data(newer_data_par)
ParSVD.incorporate_data(newest_data_par)
if ParSVD.rank == 0: print('Elapsed time PARALLEL: ', time.time() - s, 's.')

# Basic postprocessing
if ParSVD.rank == 0:

    # Save results
    ParSVD.save()

    # Visualize singular values and modes modes
    ParSVD.plot_singular_values(filename='parallel_sv.png')
    ParSVD.plot_1D_modes(filename='parallel_1d_mode0.png')
    ParSVD.plot_1D_modes(filename='parallel_1d_mode2.png', idxs=[2])
