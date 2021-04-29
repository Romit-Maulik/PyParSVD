#!/bin/bash
#COBALT -n 2
#COBALT -t 01:00:00
#COBALT -q debug-cache-quad 
#COBALT -A datascience

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules
#module load postgresql
#module load miniconda-3/latest
module load datascience/h5py-2.9.0
#source activate ae_search_env 
export PATH=/soft/libraries/mpi/mvapich2/gcc/bin/:$PATH

aprun -n 2 -N 1 -cc depth python debug_h5.py

