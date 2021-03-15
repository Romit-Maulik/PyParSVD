import numpy as np
from mpi4py import MPI

# Import custom Python packages
import pyparsvd.postprocessing as post

# For shared memory deployment:
# `export OPENBLAS_NUM_THREADS=1`



class ParSVD_Base(object):
	"""
	docstring for ParSVD_Base:
	K : Number of modes to truncate
	ff : Forget factor
	"""

	def __init__(self, K, ff, low_rank=False, results_dir='results'):

		# super(ParSVD_Base, self).__init__()
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
		return self._K



	@property
	def ff(self):
		return self._ff



	@property
	def low_rank(self):
		return self._low_rank



	@property
	def modes(self):
		if self.rank == 0:
			if isinstance(self._modes, np.ndarray):
				return self._modes
			elif isinstance(self._modes, str):
				return np.load(self._modes)
			else:
				raise TypeError("type,", type(self._modes), "not available")



	@property
	def singular_values(self):
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
		return self._iteration



	@property
	def n_modes(self):
		return self.singular_values.shape[-1]



	@property
	def comm(self):
		return self._comm



	@property
	def rank(self):
		return self._rank



	@property
	def nprocs(self):
		return self._nprocs
	# ---------------------------------------------------------------------------



	# plotting methods
	# ---------------------------------------------------------------------------

	def plot_singular_values(self, idxs=[0], title='', figsize=(12,8), filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_singular_values(
			self.singular_values,
			title=title,
			figsize=figsize,
			path=self._results_dir,
			filename=filename,
			rank=self.rank)

	def plot_1D_modes(self, idxs=[0], title='', figsize=(12,8), filename=None):
		'''
		See method implementation in the postprocessing module.
		'''
		post.plot_1D_modes(
			self.modes,
			idxs=idxs,
			title=title,
			figsize=figsize,
			path=self._results_dir,
			filename=filename,
			rank=self.rank)

	# ---------------------------------------------------------------------------
