import os
import sys
import time
import numpy as np
from mpi4py import MPI
import h5py

# Current, parent and file paths
CWD = os.getcwd()
CF  = os.path.realpath(__file__)
CFD = os.path.dirname(CF)

# Import library specific modules
sys.path.append(os.path.join("../../"))
from pyparsvd.parsvd_serial   import ParSVD_Serial
from pyparsvd.parsvd_parallel import ParSVD_Parallel

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

def load_h5(A,comm,rank,nprocs,dataset):
    h5_file = h5py.File(A, 'r', driver='mpio', comm=comm)
    dset = h5_file[dataset]
    ndof=dset.shape[0]

    num_rows_rank = int(ndof/nprocs)

    if rank != nprocs-1:
        rval =  dset[rank*num_rows_rank:(rank+1)*num_rows_rank,:].reshape(num_rows_rank,-1)
    else:
        num_dof_local = ndof - rank*num_rows_rank
        rval =  dset[rank*num_rows_rank:,:].reshape(num_dof_local,-1)

    h5_file.close()
    return rval

if __name__ == '__main__':
    # Path to data
    path = os.path.join(CFD, './data/')

    # Batchwise data - note these are h5 files
    data_path = os.path.join(path, 'Batch_0_data.h5')
    initial_data = load_h5(data_path,comm,rank,nprocs,'dataset')

    data_path = os.path.join(path, 'Batch_1_data.h5')
    new_data = load_h5(data_path,comm,rank,nprocs,'dataset')

    data_path = os.path.join(path, 'Batch_2_data.h5')
    newer_data = load_h5(data_path,comm,rank,nprocs,'dataset')

    data_path = os.path.join(path, 'Batch_3_data.h5')
    newest_data = load_h5(data_path,comm,rank,nprocs,'dataset')

    # Construct SVD objects
    ParSVD = ParSVD_Parallel(K=20, ff=1.0, low_rank=False)

    # Do first modal decomposition -- Parallel
    s = time.time()
    ParSVD.initialize(initial_data)

    # Incorporate new data -- Parallel
    ParSVD.incorporate_data(new_data)
    ParSVD.incorporate_data(newer_data)
    ParSVD.incorporate_data(newest_data)
    if ParSVD.rank == 0: print('Elapsed time PARALLEL: ', time.time() - s, 's.')

    # Basic postprocessing
    if ParSVD.rank == 0:

        # Save results
        ParSVD.save()

        # Visualize singular values and modes modes
        ParSVD.plot_singular_values(filename='parallel_sv.png')
        ParSVD.plot_1D_modes(filename='parallel_1d_mode0.png')
        ParSVD.plot_1D_modes(filename='parallel_1d_mode2.png', idxs=[2])
