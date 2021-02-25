import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

# Make sure you have this to ensure numpy doesn't automatically use multiple threads on a single compute node
# export OPENBLAS_NUM_THREADS=1

# Mode of the reduction (method of snapshots or SVD)
mos_mode = False

# Method of snapshots to accelerate
def generate_right_vectors_mos(Y):
    '''
    Y - Snapshot matrix - shape: NxS
    returns V - truncated right singular vectors
    '''
    new_mat = np.matmul(np.transpose(Y),Y)
    w, v = np.linalg.eig(new_mat)

    svals = np.sqrt(np.abs(w))
    rval = np.argmax(svals<0.0001) # eps0

    return v[:,:rval], np.sqrt(np.abs(w[:rval])) # Covariance eigenvectors, singular values

def generate_right_vectors_svd(Y):
    # Run a local SVD and threshold
    _, slocal, vt = np.linalg.svd(Y)
    rval = np.argmax(slocal<0.0001) # eps0

    slocal = slocal[:rval]
    vlocal = vt.T[:,:rval]

    return vlocal, slocal

if __name__ == '__main__':
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Check what data you have to grab
    # Here we assume that the snapshots are already segregated into different files
    # Should be (points per rank) x (snapshots) - total data matrix is nprocs*points x snapshots
    local_data = np.load('points_rank_'+str(rank)+'.npy') 

    if mos_mode: # Method of snapshots
        vlocal, slocal = generate_right_vectors_mos(local_data)
    else: # SVD
        vlocal, slocal = generate_right_vectors_svd(local_data)
    
    # Find Wr
    wlocal = np.matmul(vlocal,np.diag(slocal).T)

    # Gather data at rank 0:
    wglobal = comm.gather(wlocal,root=0)

    # perform SVD at rank 0:
    if rank == 0:
        temp = wglobal[0]
        for i in range(nprocs-1):
            temp = np.concatenate((temp,wglobal[i+1]),axis=-1)
        wglobal = temp

        x, s, y = np.linalg.svd(wglobal)
    else:
        x = None
        s = None
    
    x = comm.bcast(x,root=0)
    s = comm.bcast(s,root=0)

    # Find truncation threshold
    s_ratio = np.cumsum(s)/np.sum(s)
    rval = np.argmax(1.0-s_ratio<0.0001) # eps1

    # perform APMOS at each local rank
    phi_local = []
    for mode in range(rval):
        phi_temp = 1.0/s[mode]*np.matmul(local_data,x[:,mode:mode+1])
        phi_local.append(phi_temp)

    temp = phi_local[0]
    for i in range(rval-1):
        temp = np.concatenate((temp,phi_local[i+1]),axis=-1)
    phi_local = temp

    # Gather modes at rank 0
    # This is automatically in order
    phi_global = comm.gather(phi_local,root=0)

    if rank == 0:
        phi = phi_global[0]
        for i in range(nprocs-1):
            phi = np.concatenate((phi,phi_global[i+1]),axis=0)

        if mos_mode:
            np.save('APMOS_Basis_MOS.npy',phi)
        else:
            np.save('APMOS_Basis_SVD.npy',phi)









