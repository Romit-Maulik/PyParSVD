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

# Path to data
path = os.path.join(CFD, './data/')

# Construct SVD object - serial
SerSVD = ParSVD_Serial(K=10, ff=1.0)

# Serial data
initial_data_ser = np.load(os.path.join(path, 'Batch_0_data.npy'))
new_data_ser = np.load(os.path.join(path, 'Batch_1_data.npy'))
newer_data_ser = np.load(os.path.join(path, 'Batch_2_data.npy'))
newest_data_ser = np.load(os.path.join(path, 'Batch_3_data.npy'))

# Do first modal decomposition -- Serial
s = time.time()
SerSVD.initialize(initial_data_ser)

# Incorporate new data -- Serial
SerSVD.incorporate_data(new_data_ser)
SerSVD.incorporate_data(newer_data_ser)
SerSVD.incorporate_data(newest_data_ser)

print('Elapsed time SERIAL: ', time.time() - s, 's.')

# Basic postprocessing
# Save results
SerSVD.save()

# Visualize singular values and modes modes
SerSVD.plot_singular_values(filename='serial_sv.png')
SerSVD.plot_1D_modes(filename='serial_1d_mode0.png')
SerSVD.plot_1D_modes(filename='serial_1d_mode2.png', idxs=[2])
