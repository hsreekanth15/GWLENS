import numpy as np
import time
import multiprocessing as mpr
import h5py
import os
import configparser
import logging
from mpmath import mp, mpc, sqrt, exp, besselj, quad

mp.dps = 50  # set precision to 50 decimal places (arbitrary precision)
eps = 1e-3  # relative accuracy tolerance

def psi_powerlaw(x, amp, core, p):
    # x, amp, core, p are mpf/mpc types
    return amp * (x**2 + core**2)**(p / 2) - amp * core**p

def func_powerlaw(x, w, y, amp, core, p):
    # x: real integration variable (mpf)
    # w, y, amp, core, p: mpc or mpf
    sqrt2x = sqrt(2 * x)
    arg_bessel = w * y * sqrt2x
    bessel_val = besselj(0, arg_bessel)
    psi_val = psi_powerlaw(sqrt2x, amp, core, p)
    return bessel_val * exp(-1j * w * psi_val)

def func2_powerlaw(x, w, y, amp, core, p):
    return func_powerlaw(x, w, y, amp, core, p) * exp(1j * w * x)

def dfunc_powerlaw(x, w, y, amp, core, p):
    sqrt2x = sqrt(2 * x)
    psi_val = psi_powerlaw(sqrt2x, amp, core, p)
    J1 = besselj(1, w * y * sqrt2x)
    prefactor = -w * y / sqrt2x
    return prefactor * J1 * exp(-1j * w * psi_val) - (1j * w / (2 * x)) * func_powerlaw(x, w, y, amp, core, p)

def ddfunc_powerlaw(x, w, y, amp, core, p):
    sqrt2x = sqrt(2 * x)
    psi_val = psi_powerlaw(sqrt2x, amp, core, p)
    denom = (sqrt2x**2 + core**2)**(1 - p/2)
    dpsi = amp * p * sqrt2x / denom
    
    # derivative squared term:
    d2psi = amp * p * (1 - p / 2) * (2 * x)**(-0.5) * (1 - 2 * x / (2 * x + core**2)) / (2 * x + core**2)**(1 - p/2)
    
    term1 = (w * y) / (2 * x * sqrt2x) * (2 + 1j * w) * besselj(1, w * y * sqrt2x) * exp(-1j * w * psi_val)
    term2 = -1 / (2 * x) * (w**2 * y**2 - 1j * w / x) * func_powerlaw(x, w, y, amp, core, p)
    term3 = -1j * w / (2 * x) * dfunc_powerlaw(x, w, y, amp, core, p)
    return term1 + term2 + term3

def PowerLawAmplification(w, y, amp, core, p):
    a = 0.00001

    if w < 1 :
        b = 100 / w
    elif 1 < w < 500:
        b = 1000/w 
    else: b = 10000/w   
    #zzp = mpc(-1)
    
    
    # mpmath.quad with complex integrand
    zz = quad(lambda x: func2_powerlaw(x, w, y, amp, core, p), [a, b], error=True, maxdegree=10)
    zz_val = zz[0]  # integral value
    
    # Add tail correction terms (at b)
    tail = (-func_powerlaw(b, w, y, amp, core, p) / (1j * w) * exp(1j * w * b)
            - dfunc_powerlaw(b, w, y, amp, core, p) / (w**2) * exp(1j * w * b)
            + ddfunc_powerlaw(b, w, y, amp, core, p) / (1j * w**3) * exp(1j * w * b))
    
    zz_val += tail
        
    return -1j * w * exp(0.5 * 1j * w * y**2) * zz_val


def _worker(args):
    i, j, w, y, amp,core,p = args
    val = PowerLawAmplification(w, y, amp,core,p)
    return i, j, val.real, val.imag



def compute_powerlaw_lens_grid(w_values, y_values, amp, core, p , hdf5_filename,num_processes=None):
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
    tasks = [(i, j, w_values[i], y_values[j], amp, core, p ) for i in range(W_len) for j in range(Y_len)]

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