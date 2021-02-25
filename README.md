# Online SVD (serial and parallel)
Serial version deploys algorithm specified in:

"Sequential Karhunen–Loeve Basis Extraction and its Application to Images" by Avraham Levy and Michael Lindenbaum. IEEE TRANSACTIONS ON IMAGE PROCESSING, VOL. 9, NO. 8, AUGUST 2000.

Parallel version extends this to have a parallel SVD algorithm for the initialization and a parallel QR decomposition for the online update.
`Parallel_QR` folder valides the parallel QR algorithm.

To reproduce (needs atleast 6 available ranks):
`export OPENBLAS_NUM_THREADS=1` to ensure numpy does not multithread for this experiment.

1. Run `python data_splitter.py` to generate data etc.
2. Run `python online_svd_serial.py` for serial online update.
3. Run `mpirun -np 6 python online_svd_parallel.py`