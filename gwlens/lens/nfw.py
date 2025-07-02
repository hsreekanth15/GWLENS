import numpy as np
import time
import multiprocessing as mpr
import h5py
import os
import configparser
import logging
from joblib import Parallel, delayed
from mpmath import mp, mpc, sqrt, exp, besselj, quad,log, atan,atanh



# Setup logger (you can customize the path or pass it in)
logger = logging.getLogger("nfw_interpolator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)



mp.dps = 70  # set precision to 50 decimal places (arbitrary precision)
eps = 1e-3  # relative accuracy tolerance

def psi(x, rs):
    if x == 0:
        return 0
    elif x < 1:
        return rs/2 * ((log(x/2))**2 - atanh(sqrt(1 - x**2))**2)
    else:
        return rs/2 * ((log(x/2))**2 +  atan(sqrt(x**2 - 1))**2)
    

def func(x, w, y,rs):
    # x: real integration variable (mpf)
    # w, y, amp, core, p: mpc or mpf
    sqrt2x = sqrt(2 * x)
    arg_bessel = w * y * sqrt2x
    bessel_val = besselj(0, arg_bessel)
    psi_val = psi(sqrt2x, rs)
    return bessel_val * exp(-1j * w * psi_val)

def func2(x, w, y, rs):
    return func(x, w, y, rs) * exp(1j * w * x)

def dfunc(x, w, y, rs):
    sqrt2x = sqrt(2 * x)
    psi_val = psi(sqrt2x, rs)
    J1 = besselj(1, w * y * sqrt2x)
    prefactor = -w * y / sqrt2x
    return prefactor * J1 * exp(-1j * w * psi_val) - (1j * w / (2 * x)) * func(x, w, y,rs)

def ddfunc(x, w, y,rs):
    sqrt2x = sqrt(2 * x)
    psi_val = psi(sqrt2x, rs)
    #denom = (sqrt2x**2 + core**2)**(1 - p/2)
    #dpsi = amp * p * sqrt2x / denom
    
    # derivative squared term:
    #d2psi = amp * p * (1 - p / 2) * (2 * x)**(-0.5) * (1 - 2 * x / (2 * x + core**2)) / (2 * x + core**2)**(1 - p/2)
    
    term1 = (w * y) / (2 * x * sqrt2x) * (2 + 1j * w) * besselj(1, w * y * sqrt2x) * exp(-1j * w * psi_val)
    term2 = -1 / (2 * x) * (w**2 * y**2 - 1j * w / x) * func(x, w, y, rs)
    term3 = -1j * w / (2 * x) * dfunc(x, w, y,rs)
    return term1 + term2 + term3

def NFW(w, y, rs):

    #w = mpc(w)
    #y = mpc(y)
    #rs = mpc(rs)
    a = 0.00001

    if w < 1 :
        b = 100 / w
    elif 1 < w < 500:
        b = 1000/w 
    else: b = 10000/w   
    #zzp = mpc(-1)
    
    
    # mpmath.quad with complex integrand
    zz = quad(lambda x: func2(x, w, y,rs), [a, b], error=True, maxdegree=10)
    zz_val = zz[0]  # integral value
    
    # Add tail correction terms (at b)
    tail = (-func(b, w, y, rs) / (1j * w) * exp(1j * w * b)
            - dfunc(b, w, y, rs) / (w**2) * exp(1j * w * b)
            + ddfunc(b, w, y, rs) / (1j * w**3) * exp(1j * w * b))
    
    zz_val += tail
        
    return -1j * w * exp(0.5 * 1j * w * y**2) * zz_val


def _worker(args):
    i, j, w, y, rs = args
    val = NFW(w, y, rs)
    return i, j, val.real, val.imag



def compute_nfw_lens_grid(w_values, y_values, rs , hdf5_filename,num_processes=None):
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

    w_values = [mpc(w) for w in w_values]
    y_values = [mpc(y) for y in y_values]

    # Create task list
    tasks = [(i, j, w_values[i], y_values[j], rs ) for i in range(W_len) for j in range(Y_len)]

    logger.info(f" Output file: {hdf5_filename}")
    logger.info(f" Grid size: {W_len} × {Y_len} → {total_points} points")
    logger.info(f"Launching parallel computation with {num_processes} tasks")

    start_time = time.time()
    w_array = np.array([float(w) for w in w_values], dtype='f8')
    y_array = np.array([float(y) for y in y_values], dtype='f8')
    with h5py.File(hdf5_filename, "w") as h5file:
        h5file.create_dataset("w_values", data=w_array)
        h5file.create_dataset("y_values", data=y_array)
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

                if count % 1000 == 0 or count == total_points:
                    logger.info(f" {count}/{total_points} results written...")

        h5file.flush()

    end_time = time.time()
    logger.info(f" Done in {end_time - start_time:.2f} seconds")
    logger.info(f" Saved to: {hdf5_filename}")

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