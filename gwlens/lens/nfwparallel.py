import numpy as np
import h5py
import time
import os
import logging
from joblib import Parallel, delayed
from .nfw_old import NFW  # Ensure NFW(w, y, rs) is available

# Setup logger (you can customize the path or pass it in)
logger = logging.getLogger("nfw_interpolator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def compute_nfw_lens_test(w_values, y_values, rs_values, hdf5_filename, num_jobs=None):
    """
    Computes the NFW(w, y, rs) complex amplification factor over a 3D grid,
    and stores real and imaginary parts in an HDF5 file.

    Parameters:
    - w_values (np.ndarray): 1D array of dimensionless frequency
    - y_values (np.ndarray): 1D array of impact parameter
    - rs_values (np.ndarray): 1D array of scale radii
    - hdf5_filename (str): Path to output HDF5 file
    - num_jobs (int or None): Number of parallel jobs to use (default: all cores)
    """

    W_len, Y_len, R_len = len(w_values), len(y_values), len(rs_values)
    total = W_len * Y_len * R_len
    logger.info(f"Preparing {total} NFW evaluations ({W_len}×{Y_len}×{R_len})")
    logger.info(f"Saving to: {hdf5_filename}")

    # Start timer
    t0 = time.time()

    # Flatten all grid points as tasks
    tasks = [(i, j, k, w_values[i], y_values[j], rs_values[k])
             for i in range(W_len)
             for j in range(Y_len)
             for k in range(R_len)]

    def _compute_point(i, j, k, w, y, rs):
        val = NFW(w, y, rs)
        return (i, j, k, val.real, val.imag)

    # Run in parallel
    logger.info(f"Running in parallel with {num_jobs or os.cpu_count()} cores...")
    results = Parallel(n_jobs=num_jobs or -1, backend="loky", verbose=10)(
        delayed(_compute_point)(i, j, k, w, y, rs) for (i, j, k, w, y, rs) in tasks
    )

    # Allocate result array
    result_array = np.zeros((W_len, Y_len, R_len, 2), dtype=np.float64)
    for i, j, k, re, im in results:
        result_array[i, j, k, 0] = re
        result_array[i, j, k, 1] = im

    # Save everything to HDF5
    with h5py.File(hdf5_filename, "w") as f:
        f.create_dataset("w_values", data=w_values)
        f.create_dataset("y_values", data=y_values)
        f.create_dataset("rs_values", data=rs_values)
        f.create_dataset("results", data=result_array, chunks=True, compression="gzip")

    logger.info(f"✅ Completed in {time.time() - t0:.2f} seconds.")
    logger.info("💾 Interpolation grid saved and ready.")
