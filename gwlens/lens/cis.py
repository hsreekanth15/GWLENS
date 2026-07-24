import numpy as np
import time
import multiprocessing as mpr
import h5py
import os
import configparser
import logging
from mpmath import mp, mpc, sqrt, exp, besselj, quad, log

mp.dps = 50  # set precision to 50 decimal places (arbitrary precision)
eps = 1e-3  # relative accuracy tolerance


def psi_cis(x, amp, core):
    # x is u = sqrt(2*integration_var); amp = psi0, core = xc
    s = sqrt(core**2 + x**2)
    return amp * s + core * amp * log(2 * core / (s + core))

def func_cis(x, w, y, amp, core):
    # x: real integration variable (mpf)
    # w, y, amp, core: mpc or mpf
    sqrt2x = sqrt(2 * x)
    arg_bessel = w * y * sqrt2x
    bessel_val = besselj(0, arg_bessel)
    psi_val = psi_cis(sqrt2x, amp, core)
    return bessel_val * exp(-1j * w * psi_val)

def func2_cis(x, w, y, amp, core):
    return func_cis(x, w, y, amp, core) * exp(1j * w * x)

def dfunc_cis(x, w, y, amp, core):
    u = sqrt(2 * x)
    s = sqrt(core**2 + u**2)
    psi_val = amp * s + core * amp * log(2 * core / (s + core))
    dpsi_dx = amp / (s + core)

    J1 = besselj(1, w * y * u)
    prefactor = -w * y / u
    return prefactor * J1 * exp(-1j * w * psi_val) \
        - 1j * w * dpsi_dx * func_cis(x, w, y, amp, core)

def ddfunc_cis(x, w, y, amp, core):
    """
    Uses the exact identity: J1'(z) = J0(z) - J1(z)/z
    which is equivalent to: J1'(z) = (J0(z) - J2(z))/2 via the Bessel recurrence: J2(z) = 2 J1(z)/z - J0(z).
    """
    u = sqrt(2 * x)
    s = sqrt(core**2 + u**2)
    psi_val = amp * s + core * amp * log(2 * core / (s + core))
    dpsi_dx = amp / (s + core)
    d2psi_dx2 = -amp / (s * (s + core)**2) 

    A = -1j * w * dpsi_dx
    Aprime = -1j * w * d2psi_dx2

    z = w * y * u
    J1 = besselj(1, z)

    f = func_cis(x, w, y, amp, core)    
    df = dfunc_cis(x, w, y, amp, core)

    extra_term = 2 * w * y / u**3 * J1 * exp(-1j * w * psi_val)
    coeff_f = Aprime - A**2 - (w * y)**2 / u**2

    return coeff_f * f + 2 * A * df + extra_term

def CISAmplification(w, y, amp, core):
    a = 0.00001

    if w < 1:
        b = 100 / w
    elif 1 < w < 500:
        b = 1000 / w
    else:
        b = 10000 / w
    #zzp = mpc(-1)

    # mpmath.quad with complex integrand
    zz = quad(lambda x: func2_cis(x, w, y, amp, core), [a, b], error=True, maxdegree=10)
    zz_val = zz[0]  # integral value

    # Add tail correction terms (at b)
    tail = (-func_cis(b, w, y, amp, core) / (1j * w) * exp(1j * w * b)
            - dfunc_cis(b, w, y, amp, core) / (w**2) * exp(1j * w * b)
            + ddfunc_cis(b, w, y, amp, core) / (1j * w**3) * exp(1j * w * b))

    zz_val += tail

    return -1j * w * exp(0.5 * 1j * w * y**2) * zz_val


def _worker(args):
    i, j, w, y, amp,core = args
    val = CISAmplification(w, y, amp,core)
    return i, j, val.real, val.imag



def compute_cis_lens_grid(w_values, y_values, amp, core , hdf5_filename,num_processes=None):
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
    tasks = [(i, j, w_values[i], y_values[j], amp, core) for i in range(W_len) for j in range(Y_len)]

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
    