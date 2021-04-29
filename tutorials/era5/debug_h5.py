import numpy as np
from mpi4py import MPI
from netCDF4 import Dataset
import h5py
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nprocs = comm.Get_size()

nc_data = Dataset('./data/download.nc','r',format='NETCDF4',parallel=True)

if rank !=nprocs-1:
    ppr = int(nc_data['sp'].shape[2]/nprocs)

    local_data = nc_data['sp'][:,:,rank*ppr:(rank+1)*ppr].T.reshape(ppr,-1)
    print(local_data.shape)
else:
    ppr = int(nc_data['sp'].shape[2]/nprocs)
    num_local_dof = nc_data['sp'].shape[2] - rank*ppr

    local_data = nc_data['sp'][:,:,rank*ppr:].T.reshape(num_local_dof,-1)
    print(local_data.shape)


