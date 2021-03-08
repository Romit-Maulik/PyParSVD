"""PyParSVD init"""
__all__ = ['parsvd_base', 'parsvd_parallel', 'parsvd_serial']

from .parsvd_base     import ParSVD_Base
from .parsvd_parallel import ParSVD_Parallel
from .parsvd_serial   import ParSVD_Serial

import os
import sys
PACKAGE_PARENTS = ['..']
SCRIPT_DIR = os.path.dirname(os.path.realpath(
	os.path.join(os.getcwd(),
	os.path.expanduser(__file__))))
for P in PACKAGE_PARENTS:
	sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, P)))
