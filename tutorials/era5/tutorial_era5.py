import os
import sys
import time
import numpy as np
from mpi4py import MPI
from netCDF4 import Dataset

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

def load_nc(A,comm,rank,nprocs,dataset):
    '''
    We need to load an nc file in parallel
    '''
    nc_file = Dataset(A,'r',format='NETCDF4',parallel=True)

    # print(nc_file[dataset].shape[0])     
    # print(nc_file[dataset].shape[1])     
    # print(nc_file[dataset].shape[2])     

    if rank != nprocs-1:
        
        num_rows_rank = int(nc_file[dataset].shape[1]/nprocs)
        num_cols_rank = int(nc_file[dataset].shape[2]/nprocs)

        rval =  nc_file[dataset][:,:,
                                rank*num_cols_rank:(rank+1)*num_cols_rank
                                ].reshape(-1,nc_file[dataset].shape[1]*num_cols_rank).T
        
    else:
        
        num_rows_rank = int(nc_file[dataset].shape[1]/nprocs)
        num_rows_local = nc_file[dataset].shape[1] - rank*num_rows_rank
        
        num_cols_rank = int(nc_file[dataset].shape[2]/nprocs)
        num_cols_local = nc_file[dataset].shape[2] - rank*num_cols_rank

        rval =  nc_file[dataset][:,:,
                                rank*num_cols_rank:].reshape(-1,nc_file[dataset].shape[1]*num_cols_local).T

    nc_file.close()
    return rval.filled()

if __name__ == '__main__':

    # Construct SVD objects
    ParSVD = ParSVD_Parallel(K=10, ff=1.0, low_rank=True)

    # Path to data
    data_path = os.path.join(CFD, './data/download.nc')

    # Data from nc file
    initial_data = load_nc(data_path,comm,rank,nprocs,'sp')

    # Do first modal decomposition -- Parallel
    s = time.time()
    ParSVD.initialize(initial_data)

    # Incorporate new data -- Parallel
    if ParSVD.rank == 0: 
        print('Elapsed time PARALLEL: ', time.time() - s, 's.')
        print('Using ',nprocs,'ranks')

    # Basic postprocessing
    if ParSVD.rank == 0:

        # Save results
        ParSVD.save()

        # Visualize singular values and modes modes
        num_rows, num_cols = 1801, 3600
        ParSVD.plot_singular_values(filename='parallel_sv.png')
        ParSVD.plot_2D_modes(num_rows,num_cols, filename='parallel_2d_mode0.png')
        ParSVD.plot_2D_modes(num_rows,num_cols, filename='parallel_2d_mode1.png', idxs=[2])
