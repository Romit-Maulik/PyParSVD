import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt
from mpi4py import MPI

if __name__ == '__main__':

    num_dof_rank = 4000
    num_snapshots = 100

    serial_mode = False
    
    As = np.random.uniform(size=(6*num_dof_rank,num_snapshots))

    if serial_mode:
        # The serial stuff
        qserial, rserial = np.linalg.qr(As)
        np.save('Q_Serial.npy',qserial)
        np.save('R_Serial.npy',rserial)

        print(qserial.shape)
        print(rserial.shape)
    else:

        # The parallel stuff
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        A = As[rank*num_dof_rank:(rank+1)*num_dof_rank]

        # Perform the local QR
        q, r = np.linalg.qr(A)
        rlocal_shape_0 = r.shape[0]
        rlocal_shape_1 = r.shape[1]

        # Gather data at rank 0:
        r_global = comm.gather(r,root=0)

        # perform SVD at rank 0:
        if rank == 0:
            temp = r_global[0]
            for i in range(nprocs-1):
                temp = np.concatenate((temp,r_global[i+1]),axis=0)
            r_global = temp

            qglobal, rfinal = np.linalg.qr(r_global)

            # For this rank
            qlocal = np.matmul(q,qglobal[:rlocal_shape_0])

            # send to other ranks
            for dest_rank in range(1,nprocs):
                comm.send(qglobal[rank*rlocal_shape_0:(rank+1)*rlocal_shape_0], dest=dest_rank, tag=dest_rank+10)

        else:
            # Receive qglobal slices from other ranks
            qglobal = comm.recv(source=0, tag=rank+10)

            # For this rank
            qlocal = np.matmul(q,qglobal)

        qfinal = comm.gather(qlocal,root=0)

        if rank == 0:
            temp1 = qfinal[0]
            for i in range(nprocs-1):
                temp1 = np.concatenate((temp1,qfinal[i+1]),axis=0)

            np.save('Q_Parallel.npy',temp1)
            np.save('R_Parallel.npy',rfinal)

            print(temp1.shape)
            print(rfinal.shape)
