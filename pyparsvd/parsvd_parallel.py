import os
import numpy as np
from mpi4py import MPI
from netCDF4 import Dataset
import h5py

# import PyParSVD as base class for ParSVD_Parallel
from pyparsvd.parsvd_base import ParSVD_Base

# Current, parent and file paths
CWD = os.getcwd()

# Fix random seed for reproducibility
np.random.seed(10)

# For shared memory deployment:
# `export OPENBLAS_NUM_THREADS=1`



class ParSVD_Parallel(ParSVD_Base):

    """
    PyParSVD parallel class.

    :param int K: number of modes to truncate.
    :param int ff: forget factor.
    :param bool low_rank: if True, it uses a low rank algorithm to speed up computations.
    :param str results_dir: if specified, it saves the results in `results_dir`. \
        Default save path is under a folder called `results` in current working path.
    """

    def __init__(self, K, ff, low_rank=False, results_dir='results'):
        super().__init__(K, ff, low_rank, results_dir)

    def initialize(self, A):

        """
        Initialize SVD computation with initial data.

        :param ndarray/str A: initial data matrix
        """        
        self.ulocal, self._singular_values = self.parallel_svd(A)
        self._gather_modes()

        return self


    def incorporate_data(self, A):
        """
        Incorporate new data in a streaming way for SVD computation.

        :param ndarray/str A: new data matrix.
        """
        self._iteration += 1
        ll = self._ff * np.matmul(self.ulocal, np.diag(self._singular_values))
        ll = np.concatenate((ll, A), axis=-1)
        qlocal, utemp, self._singular_values = self.parallel_qr(ll)
        self.ulocal = np.matmul(qlocal, utemp)
        self._gather_modes()

        return self

    def parallel_qr(self, A):

        # Perform local QR
        q, r = np.linalg.qr(A)
        rlocal_shape_0 = r.shape[0]
        rlocal_shape_1 = r.shape[1]

        # Gather data at rank 0:
        r_global = self.comm.gather(r, root=0)

        # perform SVD at rank 0:
        if self.rank == 0:
            temp = r_global[0]
            for i in range(self.nprocs-1):
                temp = np.concatenate((temp, r_global[i+1]), axis=0)
            r_global = temp

            qglobal, rfinal = np.linalg.qr(r_global)
            qglobal = -qglobal # Trick for consistency
            rfinal = -rfinal

            # For this rank
            qlocal = np.matmul(q, qglobal[:rlocal_shape_0])

            # send to other ranks
            for rank in range(1, self.nprocs):
                self.comm.send(qglobal[rank*rlocal_shape_0:\
                                (rank+1)*rlocal_shape_0],
                               dest=rank, tag=rank+10)

            # Step b of Levy-Lindenbaum - small operation
            if self._low_rank:
                # Low rank SVD
                unew, snew = low_rank_svd(rfinal, self._K)
            else:
                unew, snew, _ = np.linalg.svd(rfinal)

        else:
            # Receive qglobal slices from other ranks
            qglobal = self.comm.recv(source=0, tag=self.rank+10)

            # For this rank
            qlocal = np.matmul(q, qglobal)

            # To receive new singular vectors
            unew = None
            snew = None

        unew = self.comm.bcast(unew, root=0)
        snew = self.comm.bcast(snew, root=0)

        return qlocal, unew, snew



    def parallel_svd(self, A):

        vlocal, slocal = generate_right_vectors(A,self._K)

        # Find Wr
        wlocal = np.matmul(vlocal, np.diag(slocal).T)

        # Gather data at rank 0:
        wglobal = self.comm.gather(wlocal, root=0)

        # perform SVD at rank 0:
        if self.rank == 0:
            temp = wglobal[0]
            for i in range(self.nprocs-1):
                temp = np.concatenate((temp, wglobal[i+1]), axis=-1)
            wglobal = temp

            if self._low_rank:
                x, s = low_rank_svd(wglobal, self._K)
            else:
                x, s, y = np.linalg.svd(wglobal)
        else:
            x = None
            s = None

        x = self.comm.bcast(x, root=0)
        s = self.comm.bcast(s, root=0)

        # # Find truncation threshold
        # s_ratio = np.cumsum(s)/np.sum(s)
        # rval = np.argmax(1.0-s_ratio<0.0001) # eps1

        # perform APMOS at each local rank
        phi_local = []
        for mode in range(self._K):
            phi_temp = 1.0 / s[mode] * np.matmul(A, x[:,mode:mode+1])
            phi_local.append(phi_temp)

        temp = phi_local[0]
        for i in range(self._K - 1):
            temp = np.concatenate((temp, phi_local[i+1]), axis=-1)

        return temp, s[:self._K]



    def save(self):
        """
        Save data.
        """
        results_dir = os.path.join(CWD, self._results_dir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        pathname_sv = os.path.join(
            results_dir, 'parallel_singular_values.npy')
        np.save(pathname_sv, self._singular_values)
        pathname_m = os.path.join(
            results_dir, 'parallel_POD.npy')

        if self.rank == 0:
            np.save(pathname_m, self._modes)

        self._singular_values = pathname_sv
        self._modes = pathname_m



    def _gather_modes(self):
        # Gather modes at rank 0
        modes_global = self.comm.gather(self.ulocal, root=0)
        if self.rank == 0:
            self._modes = modes_global[0]
            for i in range(self.nprocs-1):
                self._modes = np.concatenate((
                    self._modes, modes_global[i+1]), axis=0)



def generate_right_vectors(A,K):
    """
    Method of snapshots.

    :param np.ndarray A: snapshot data matrix.

    :return: truncated right singular vectors `v`.
    :rtype: np.ndarray
    """
    new_mat = np.matmul(np.transpose(A), A)
    w, v = np.linalg.eig(new_mat)

    svals = np.sqrt(np.abs(w))
    # rval = np.argmax(svals < 0.0001) # eps0

    # Covariance eigenvectors, singular values
    return v[:,:K], np.sqrt(np.abs(w[:K]))



def low_rank_svd(A, K):
    """
    Performs randomized SVD.

    :param np.ndarray A: snapshot data matrix.
    :param int K: truncation.

    :return: singular values `unew` and `snew`.
    :rtype: np.ndarray, np.ndarray
    """
    M = A.shape[0]
    N = A.shape[1]

    omega = np.random.normal(size=(N, 2*K))
    omega_pm = np.matmul(A, np.transpose(A))
    Y = np.matmul(omega_pm, np.matmul(A,omega))

    Qred, Rred = np.linalg.qr(Y)

    B = np.matmul(np.transpose(Qred), A)
    ustar, snew, _ = np.linalg.svd(B)

    unew = np.matmul(Qred, ustar)

    unew = unew[:,:K]
    snew = snew[:K]

    return unew, snew



def check_orthogonality(modes, num_modes):
    """
    Check orthogonality of modes.

    :param np.ndarray modes: modes.
    :param int num_modes: number of modes.

    :return: True if orthogonality check passes.
    :rtype: bool
    """
    for m1 in range(num_modes):
        for m2 in range(num_modes):
            if m1 == m2:
                s_ = np.sum(modes[:,m1] * modes[:,m2])
                if not np.isclose(s_, 1.0):
                    print('Orthogonality check failed')
                    break
            else:
                s_ = np.sum(modes[:,m1] * modes[:,m2])
                if not np.isclose(s_, 0.0):
                    print('Orthogonality check failed')
                    break

    print('Orthogonality check passed successfully')
    return True
