#!/bin/bash
#COBALT -n 128
#COBALT -t 01:00:00
#COBALT -q default 
#COBALT -A datascience

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules
#module load postgresql
#module load miniconda-3/latest
module load datascience/h5py-2.9.0
module load datascience/netCDF4-1.5.6
#source activate ae_search_env 
export PATH=/soft/libraries/mpi/mvapich2/gcc/bin/:$PATH

aprun -n 128 -N 1 -cc depth python tutorial_era5.py

