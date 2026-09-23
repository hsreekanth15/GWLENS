import numpy as np
import time
import multiprocessing as mpr
from importlib.metadata import version, PackageNotFoundError  
from datetime import datetime, timezone                        
import h5py
import os
import configparser
import logging
from mpmath import mp, mpc, sqrt, exp, besselj, quad, log

mp.dps = 50  # set precision to 50 decimal places (arbitrary precision)
eps = 1e-3  # relative accuracy tolerance

    
def psi_plummer(x, core):
    # x is u = sqrt(2*integration_var); core = xc #check https://arxiv.org/pdf/astro-ph/0602368
    return 0.5 * core**2 * log(1 + x**2 / core**2)

def func_plummer(x, w, y, core):
    u = sqrt(2 * x)

    return (
        besselj(0, w * y * u)
        * exp(-1j * w * psi_plummer(u, core))
    )

def func2_plummer(x, w, y, core):
    return func_plummer(x, w, y, core) * exp(1j * w * x)

def dfunc_plummer(x, w, y, core):
    u = sqrt(2 * x)
    psi = psi_plummer(u, core)
    #dpsi_dx = core**2 / (core**2 + u**2)   # <- don't need
    J1 = besselj(1, w * y * u)
    return (-w*y/u* J1* exp(-1j*w*psi)- (1j*w/(2*x))* func_plummer(x, w, y, core))

def ddfunc_plummer(x, w, y, core):
    u = sqrt(2 * x)
    psi = psi_plummer(u, core)
    #dpsi_dx = core**2 / (core**2 + u**2)          # <- don't need 
    #d2psi_dx2 = -2 * core**2 / (core**2 + u**2)**2   # <- don't need

    term1 = (
        (w*y)/(2*x*u)
        * (2 + 1j*w)
        * besselj(1, w*y*u)
        * exp(-1j*w*psi)
    )

    term2 = (
        -1/(2*x)
        * (w**2*y**2 - 1j*w/x)
        * func_plummer(x, w, y, core)
    )

    term3 = (
        -1j*w/(2*x)
        * dfunc_plummer(x, w, y, core)
    )

    return term1 + term2 + term3

def PlummerAmplification(w, y, core):

    a = 1e-5

    if w < 1:
        b = 100 / w
    elif w < 500:
        b = 1000 / w
    else:
        b = 10000 / w

    zz = quad(
        lambda x: func2_plummer(x, w, y, core),
        [a, b],
        error=True,
        maxdegree=10
    )

    zz_val = zz[0]

    tail = (-func_plummer(b, w, y, core) / (1j * w) * exp(1j * w * b)
                - dfunc_plummer(b, w, y, core) / (w**2) * exp(1j * w * b)
                + ddfunc_plummer(b, w, y, core) / (1j * w**3) * exp(1j * w * b))
    
    zz_val += tail
    
    return -1j * w * exp(0.5 * 1j * w * y**2) * zz_val


def _worker(args):
    i, j, w, y,core = args
    val = PlummerAmplification(w, y, core)
    return i, j, val.real, val.imag



def compute_plummer_lens_grid(w_values, y_values, core , hdf5_filename,num_processes=None):
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
    tasks = [(i, j, w_values[i], y_values[j], core) for i in range(W_len) for j in range(Y_len)]

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

                if count % 100 == 0 or count == total_points:
                    logger.info(f" {count}/{total_points} results written...")

        h5file.flush()

    end_time = time.time()
    logger.info(f" Done in {end_time - start_time:.2f} seconds")
    logger.info(f" Saved to: {hdf5_filename}")
    