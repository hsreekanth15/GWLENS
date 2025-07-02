import numpy as np
import time
import multiprocessing as mpr
import h5py
import os
import configparser
import logging
from mpmath import mp, mpc, exp, pi, gamma, factorial, hyp1f1, nstr

# Set desired decimal precision (increase if needed)
mp.dps = 100  # digits of precision

def calculate_F(w, y, n_lim, epsilon=1e-30):
    # Convert inputs to mpmath-compatible complex numbers
    w = mpc(w)
    y = mpc(y)
    
    const = exp(0.5j * w * y**2)
    result = mpc(0)

    for n in range(n_lim):
        try:
            gamma_term = gamma(1 + n/2) / factorial(n)
            exponent = (2 * w * exp(pi * 3j / 2))**(n/2)
            hyp_term = hyp1f1(1 + n/2, 1, -0.5j * w * y**2)
            term = gamma_term * exponent * hyp_term
            result += term

            if abs(term) < epsilon:
                break  # early convergence
        except Exception as e:
            print(f"Error at n={n}: {e}")
            return -1
    
    return result * const




def _worker(args):
    i, j, w, y,n_lim = args
    val = calculate_F(w, y, n_lim)
    return i, j, val.real, val.imag


def compute_sis_lens_grid(w_values, y_values, hdf5_filename, n_lim = 1000 ,num_processes=None):
    """
    Computes the Point(w, y) complex amplification factor over a grid
    and stores the result in an HDF5 file.

    Parameters:
    - w_values (1D np.array): dimensionless frequency array
    - y_values (1D np.array): impact parameter array
    - hdf5_filename (str): path to output HDF5 file
    - num_processes (int or None): number of CPU cores to use (default: all)
    """

        #Read the config file
    config = configparser.ConfigParser()
    config.read('parameters.ini')

    # Setup logging
    logger = logging.getLogger("compute_logger")
    logger.setLevel(logging.INFO)


    #Store result
    output_folder = config['Output']['outdir']

    # Subfolder for interpolation outputs
    log_folder = os.path.join(output_folder, 'logs')
    os.makedirs(log_folder, exist_ok=True)  # Create if it doesn't exist

    log_file = os.path.join(log_folder,'interpolation.log')


    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    W_len = len(w_values)
    Y_len = len(y_values)
    total_points = W_len * Y_len

    # Create task list
    tasks = [(i, j, w_values[i], y_values[j],n_lim ) for i in range(W_len) for j in range(Y_len)]

    logger.info(f" Output file: {hdf5_filename}")
    logger.info(f" Grid size: {W_len} × {Y_len} → {total_points} points")
    logger.info(f"Launching parallel computation with {num_processes} tasks")

    start_time = time.time()

    with h5py.File(hdf5_filename, "w") as h5file:
        h5file.create_dataset("w_values", data=w_values)
        h5file.create_dataset("y_values", data=y_values)
        dset = h5file.create_dataset(
            "results",
            shape=(W_len, Y_len, 2),
            dtype="f8",
            chunks=(1, Y_len, 2)  # optimize for row-wise access
        )

        with mpr.Pool(processes=num_processes or mpr.cpu_count()) as pool:
            for count, (i, j, re, im) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=100), 1):
                dset[i, j, 0] = re
                dset[i, j, 1] = im

                if count % 10000 == 0 or count == total_points:
                    logger.info(f" {count}/{total_points} results written...")

        h5file.flush()

    end_time = time.time()
    logger.info(f" Done in {end_time - start_time:.2f} seconds")
    logger.info(f" Saved to: {hdf5_filename}")
