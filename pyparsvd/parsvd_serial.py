import os
import numpy as np

# import PyParSVD as base class for ParSVD_Parallel
from pyparsvd.parsvd_base import ParSVD_Base

# Current, parent and file paths
CWD = os.getcwd()

# Fix random seed for reproducibility
np.random.seed(10)



class ParSVD_Serial(ParSVD_Base):

	"""
	PyParSVD serial class. 

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

		:param ndarray A: initial data matrix.
		"""

		# Computing R-SVD of the initial matrix - step 1 section II
		q, r = np.linalg.qr(A)

		# Compute SVD of r - v is already transposed  - step 2 section II
		# https://stackoverflow.com/questions/24913232/using-numpy-np-linalg-svd-for-singular-value-decomposition
		ui, self._singular_values, self.vit = np.linalg.svd(r)

		# Get back U and truncate
		self._modes = np.matmul(q, ui)[:,:self._K]  #- step 3 section II
		self._singular_values = self._singular_values[:self._K]

		return self



	def incorporate_data(self, A):
		"""
		Incorporate new data in a streaming way for SVD computation.

		:param ndarray A: new data matrix.
		"""

		# Section III B 3(a):
		m_ap = self._ff * np.matmul(self._modes, np.diag(self._singular_values))
		m_ap = np.concatenate((m_ap, A), axis=-1)
		udashi, ddashi = np.linalg.qr(m_ap)

		# Section III B 3(b):
		utildei, dtildei, vtildeti = np.linalg.svd(ddashi)

		# Section III B 3(c):
		max_idx = np.argsort(dtildei)[::-1][:self._K]
		self._singular_values = dtildei[max_idx]
		utildei = utildei[:,max_idx]
		self._modes = np.matmul(udashi, utildei)

		return self



	def save(self):
		"""
		Save data.
		"""

		results_dir = os.path.join(CWD, self._results_dir)
		if not os.path.exists(results_dir):
			os.makedirs(results_dir)
		pathname_sv = os.path.join(
			results_dir,
			'serial_singular_values.npy')
		np.save(pathname_sv, self._singular_values)
		pathname_m = os.path.join(
			results_dir,
			'serial_POD.npy')
		np.save(pathname_m, self._modes)
		self._singular_values = pathname_sv
		self._modes = pathname_m
