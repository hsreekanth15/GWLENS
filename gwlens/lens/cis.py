import numpy as np
import time
import multiprocessing as mpr
import h5py
import os
import configparser
from importlib.metadata import version, PackageNotFoundError   
from datetime import datetime, timezone                        
import logging
from mpmath import mp, mpf, sqrt, exp, besselj, quad, log
from multiprocessing import Pool, cpu_count

mp.dps = 50  # set precision to 50 decimal places (arbitrary precision)
eps = 1e-3  # relative accuracy tolerance


def psi(x, core):
    # x is u = sqrt(2*integration_var); amp = psi0, core = xc
    s = sqrt(core**2 + x**2)
    return s + core * log(2 * core / (s + core))

def func(x, w, y, core):
    # x: real integration variable (mpf)
    # w, y, amp, core: mpc or mpf
    sqrt2x = sqrt(2 * x)
    arg_bessel = w * y * sqrt2x
    bessel_val = besselj(0, arg_bessel)
    psi_val = psi(sqrt2x, core)
    return bessel_val * exp(-1j * w * psi_val)

def func2(x, w, y,core):
    return func(x, w, y,core) * exp(1j * w * x)

def dfunc(x, w, y, core):
    sqrt2x = sqrt(2 * x)
    #s = sqrt(core**2 + u**2)
    #psi_val = amp * s + core * amp * log(2 * core / (s + core))
    psi_val=psi(sqrt2x, core)
    #dpsi_dx = amp / (s + core)

    J1 = besselj(1, w * y * sqrt2x)
    prefactor = -w * y / sqrt2x
    return prefactor * J1 * exp(-1j * w * psi_val) -(1j * w / (2 * x)) * func(x, w, y,core)

def ddfunc(x, w, y,core):
    """
    Uses the exact identity: J1'(z) = J0(z) - J1(z)/z
    which is equivalent to: J1'(z) = (J0(z) - J2(z))/2 via the Bessel recurrence: J2(z) = 2 J1(z)/z - J0(z).
    """
    sqrt2x = sqrt(2 * x)
    #u = sqrt(2 * x)
    #s = sqrt(core**2 + u**2)
    psi_val = psi(sqrt2x, core)
    #dpsi_dx = amp / (s + core)
    #d2psi_dx2 = -amp / (s * (s + core)**2) 

    #A = -1j * w * dpsi_dx
    #Aprime = -1j * w * d2psi_dx2

    #z = w * y * u
    #J1 = besselj(1, z)

    #f = func(x, w, y, amp, core)    
    #df = dfunc(x, w, y, amp, core)

    #extra_term = 2 * w * y / u**3 * J1 * exp(-1j * w * psi_val)
    #coeff_f = Aprime - A**2 - (w * y)**2 / u**2
    term1 = (w * y) / (2 * x * sqrt2x) * (2 + 1j * w) * besselj(1, w * y * sqrt2x) * exp(-1j * w * psi_val)
    term2 = -1 / (2 * x) * (w**2 * y**2 - 1j * w / x) * func(x, w, y,core)
    term3 = -1j * w / (2 * x) * dfunc(x, w, y,core)
    return term1 + term2 + term3


def CISAmplification(w, y,core):
    a = 0.00001
    # Convert inputs to mpmath types
    w = mpf(w)        # real number
    y = mpf(y)        # real number
    core = mpf(core)      # real number
    a = mpf('0.00001')


    # Convert inputs to mpmath types
    w = mpf(w)        # real number
    y = mpf(y)        # real number
    core = mpf(core)      # real number
    a = mpf('0.00001')

    if w < 1:
        b = mpf('100') / w
    elif 1 < w < 500:
        b = mpf('1000') / w
    else:
        b = mpf('10000') / w
    #zzp = mpc(-1)

    # mpmath.quad with complex integrand
    zz = quad(lambda x: func2(x, w, y, core), [a, b], error=True, maxdegree=10)
    zz_val = zz[0]  # integral value

    # Add tail correction terms (at b)
    tail = (-func(b, w, y, core) / (1j * w) * exp(1j * w * b)
            - dfunc(b, w, y, core) / (w**2) * exp(1j * w * b)
            + ddfunc(b, w, y, core) / (1j * w**3) * exp(1j * w * b))

    zz_val += tail

    return -1j * w * exp(0.5 * 1j * w * y**2) * zz_val




def _worker(args):
    i, j, k,w, y,core = args
    val = CISAmplification(w, y,core)
    return i, j, k, val.real, val.imag

def _as_1d(x):
    """Scalar -> length-1 array; remembers whether it was scalar."""
    arr = np.atleast_1d(np.asarray(x, dtype=object) if isinstance(x, str) else np.asarray(x))
    return arr, np.ndim(x) == 0

def CISparallel(w, y, core, processes=None, chunksize=None, squeeze=True):
    """
    Evaluate CISAmplification over the full grid w x y x core in parallel.

    Returns a complex ndarray of shape (len(w), len(y), len(core)).
    With squeeze=True, axes that came from scalar inputs are dropped.
    """
    wa, w_scalar = _as_1d(w)
    ya, y_scalar = _as_1d(y)
    ca, c_scalar = _as_1d(core)

    nw, ny, nc = len(wa), len(ya), len(ca)
    total = nw * ny * nc
    if total == 0:
        return np.empty((nw, ny, nc), dtype=complex)

    tasks = [
        (i, j, k, wa[i], ya[j], ca[k])
        for i in range(nw)
        for j in range(ny)
        for k in range(nc)
    ]

    processes = min(processes or cpu_count(), total)
    if chunksize is None:
        chunksize = max(1, total // (processes * 4))

    out = np.empty((nw, ny, nc), dtype=complex)

    with Pool(processes) as pool:
        for i, j, k, re, im in pool.imap_unordered(_worker, tasks, chunksize=chunksize):
            out[i, j, k] = complex(re, im)

    if squeeze:
        axes = tuple(ax for ax, s in enumerate((w_scalar, y_scalar, c_scalar)) if s)
        if axes:
            out = out.reshape([d for ax, d in enumerate(out.shape) if ax not in axes])
        if out.ndim == 0:
            return complex(out)

    return out

def compute_cis_lens_grid(w_values, y_values, core_values , hdf5_filename,num_processes=None):
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
    C_len = len(core_values)
    total_points = W_len * Y_len * C_len

    # Create task list
    tasks = [(i, j, k, w_values[i], y_values[j],core_values[k]) for i in range(W_len) for j in range(Y_len) for k in range(C_len)]

    logger.info(f" Output file: {hdf5_filename}")
    logger.info(f" Grid size: {W_len} × {Y_len} × {C_len} → {total_points} points")
    logger.info(f"Launching parallel computation with {num_processes} tasks")

    start_time = time.time()

    with h5py.File(hdf5_filename, "w") as h5file:
        h5file.create_dataset("w_values", data=w_values)
        h5file.create_dataset("y_values", data=y_values)
        h5file.create_dataset("core_values",data=core_values)

        dset = h5file.create_dataset(
            "results",
            shape=(W_len, Y_len,C_len,2),
            dtype="f8",
            chunks=(1, Y_len, 1 , 2)  # optimize for row-wise access
        )

        with mpr.Pool(processes=num_processes or mpr.cpu_count()) as pool:
            for count, (i, j,k, re, im) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=100), 1):
                dset[i, j,k, 0] = re
                dset[i, j, k,1] = im

                if count % 1000 == 0 or count == total_points:
                    logger.info(f" {count}/{total_points} results written...")

        h5file.flush()

    end_time = time.time()
    logger.info(f" Done in {end_time - start_time:.2f} seconds")
    logger.info(f" Saved to: {hdf5_filename}")
    
def compute_cis_lens_grid_mpi(w_values, y_values, core_values,
                              hdf5_filename, t_start=None,
                              compression="gzip"):
    """
    CIS(w, y, rs) over a 3D grid, computed with MPI.

    Results are stored as (W_len, Y_len, R_len, 2), where the last
    axis is [real, imag].
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    W_len = len(w_values)
    Y_len = len(y_values)
    C_len = len(core_values)
    total_points = W_len * Y_len * C_len

    if t_start is not None:
        max_startup = comm.allreduce(time.time() - t_start, op=MPI.MAX)
    else:
        max_startup = -1.0

    # ---- logging: rank 0 only ----
    logger = None
    if rank == 0:
        logger = logging.getLogger("compute_logger")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        try:
            config = configparser.ConfigParser()
            config.read("parameters.ini")
            output_folder = config["Output"]["outdir"]
        except Exception:
            output_folder = "."

        log_folder = os.path.join(output_folder, "logs")
        os.makedirs(log_folder, exist_ok=True)

        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                                "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(os.path.join(log_folder, "interpolation.log"))
        sh = logging.StreamHandler()
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)

        logger.info(f"Ranks: {size}")
        logger.info(f"Grid: {W_len} x {Y_len} x {C_len} = {total_points} points")
        logger.info(f"Tasks per rank: ~{total_points // size}")
        logger.info(f"mpmath precision: mp.dps = {mp.dps}")
        logger.info(f"Slowest-rank startup: {max_startup:.1f} s")

    t0 = time.time()

    # ---- mpmath inputs, rebuilt locally on every rank ----
    w_mp = [mpf(float(w)) for w in w_values]
    y_mp = [mpf(float(y)) for y in y_values]
    core_mp = [mpf(float(c)) for c in core_values]

    # ---- flatten the 3D index space, then stride ----
    tasks = [(i, j, k)
             for i in range(W_len)
             for j in range(Y_len)
             for k in range(C_len)]
    local_tasks = tasks[rank::size]

    local_results = []
    for i, j, k in local_tasks:
        amp = CISAmplification(w_mp[i], y_mp[j], core_mp[k])      # mpc
        local_results.append((i, j, k, float(amp.real), float(amp.imag)))

    t_compute = time.time() - t0

    # ---- balance diagnostics ----
    t_max = comm.allreduce(t_compute, op=MPI.MAX)
    t_min = comm.allreduce(t_compute, op=MPI.MIN)
    t_sum = comm.allreduce(t_compute, op=MPI.SUM)

    all_results = comm.gather(local_results, root=0)

    if rank == 0:
        logger.info(f"Compute  slowest rank: {t_max:.1f} s")
        logger.info(f"Compute  fastest rank: {t_min:.1f} s")
        logger.info(f"Compute  mean rank:    {t_sum/size:.1f} s")
        logger.info(f"Balance efficiency: {100*(t_sum/size)/t_max:.1f} %")

        flat = [item for sub in all_results for item in sub]
        if len(flat) != total_points:
            logger.warning(f"Expected {total_points} results, got {len(flat)}")

        # NaN-fill so a missing point is visibly wrong, not silently zero
        grid = np.full((W_len, Y_len, C_len, 2), np.nan, dtype="f8")
        for i, j, k, re_, im_ in flat:
            grid[i, j, k, 0] = re_
            grid[i, j, k, 1] = im_

        t_write = time.time()
        with h5py.File(hdf5_filename, "w") as h5:
            h5.create_dataset("w_values",  data=np.asarray(w_values,  dtype="f8"))
            h5.create_dataset("y_values",  data=np.asarray(y_values,  dtype="f8"))
            h5.create_dataset("core_values", data=np.asarray(core_values, dtype="f8"))
            dset = h5.create_dataset(
                "results",
                data=grid,
                chunks=(1, Y_len, C_len, 2),
                compression=compression,
            )
            dset.attrs["axes"] = "w, y, core, [real|imag]"
            dset.attrs["mp_dps"] = mp.dps

        logger.info(f"HDF5 write: {time.time() - t_write:.1f} s")
        logger.info(f"Saved to {hdf5_filename}")