import numpy as np
from mpi4py import MPI

# Import custom Python packages
import pyparsvd.postprocessing as post

# For shared memory deployment:
# `export OPENBLAS_NUM_THREADS=1`



class ParSVD_Base(object):

	"""
	PyParSVD base class. It implements data and methods shared
	across the derived classes.

	:param int K: number of modes to truncate.
	:param int ff: forget factor.
	:param bool low_rank: if True, it uses a low rank algorithm to speed up computations.
	:param str results_dir: if specified, it saves the results in `results_dir`. \
		Default save path is under a folder called `results` in current working path.
	"""

	def __init__(self, K, ff, low_rank=False, results_dir='results'):

		self._K = K
		self._ff = ff
		self._low_rank = low_rank
		self._results_dir = results_dir
		self._iteration = 0

		# Initialize MPI
		self._comm = MPI.COMM_WORLD
		self._rank = self.comm.Get_rank()
		self._nprocs = self.comm.Get_size()



	# basic getters
	# ---------------------------------------------------------------------------

	@property
	def K(self):
		"""
		Get the number of modes to truncate.

		:return: number of modes to truncate.
		:rtype: int
		"""
		return self._K



	@property
	def ff(self):
		"""
		Get the forget factor.

		:return: forget factor.
		:rtype: int
		"""
		return self._ff



	@property
	def low_rank(self):
		"""
		Get the low rank behaviour.

		:return: low rank behaviour.
		:rtype: bool
		"""
		return self._low_rank



	@property
	def modes(self):
		"""
		Get the modes.

		:return: modes.
		:rtype: ndarray
		"""
		if self.rank == 0:
			if isinstance(self._modes, np.ndarray):
				return self._modes
			elif isinstance(self._modes, str):
				return np.load(self._modes)
			else:
				raise TypeError("type,", type(self._modes), "not available")



	@property
	def singular_values(self):
		"""
		Get the singular values.

		:return: singular values.
		:rtype: ndarray
		"""
		if self.rank == 0:
			if isinstance(self._singular_values, np.ndarray):
				return self._singular_values
			elif isinstance(self._singular_values, str):
				return np.load(self._singular_values)
			else:
				raise TypeError("type,", type(self._singular_values),
					"not available")



	@property
	def iteration(self):
		"""
		Get the number of data incorporation performed \
		in the streaming data ingestion.

		:return: iterations.
		:rtype: int
		"""
		return self._iteration



	@property
	def n_modes(self):
		"""
		Get the number of modes.

		:return: number of modes.
		:rtype: int
		"""
		return self.singular_values.shape[-1]



	@property
	def comm(self):
		"""
		Get the parallel MPI Communicator.

		:return: comm.
		:rtype: MPI_Comm
		"""
		return self._comm



	@property
	def rank(self):
		"""
		Get the parallel MPI Rank.

		:return: rank.
		:rtype: MPI_Rank
		"""
		return self._rank



	@property
	def nprocs(self):
		"""
		Get the number processors

		:return: processors.
		:rtype: int
		"""
		return self._nprocs
	# ---------------------------------------------------------------------------



	# plotting methods
	# ---------------------------------------------------------------------------

	def plot_singular_values(self, idxs=[0], title='', figsize=(12,8), filename=None):
		"""
		See method implementation in the postprocessing module.
		"""
		post.plot_singular_values(
			self.singular_values,
			title=title,
			figsize=figsize,
			path=self._results_dir,
			filename=filename,
			rank=self.rank)

	def plot_1D_modes(self, idxs=[0], title='', figsize=(12,8), filename=None):
		"""
		See method implementation in the postprocessing module.
		"""
		post.plot_1D_modes(
			self.modes,
			idxs=idxs,
			title=title,
			figsize=figsize,
			path=self._results_dir,
			filename=filename,
			rank=self.rank)

	def plot_2D_modes(self,num_rows,num_cols, idxs=[0], title='', figsize=(12,8), filename=None):
		"""
		See method implementation in the postprocessing module.
		"""
		post.plot_2D_modes(
			self.modes,
			num_rows,
			num_cols,
			self._nprocs,
			idxs=idxs,
			title=title,
			figsize=figsize,
			path=self._results_dir,
			filename=filename,
			rank=self.rank)

	# ---------------------------------------------------------------------------
