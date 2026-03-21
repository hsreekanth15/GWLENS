"""
author: Sreeakanth Harikumar
"""
import os
import numpy as np
import multiprocessing as mpr
from mpmath import hyp1f1,exp, mp,sqrt,mpc,pi,log,gamma, besselj
import logging
import mpmath
from astropy.constants import c,G,M_sun
import configparser
import time
import h5py
from .base import Lens

mp.dps= 60

#Define constants   
c = c.value
G = G.value
M_sun = M_sun.value



class PML(Lens):

    def potential(self,x):
        return np.log(np.abs(x))

    def time_delay(self,x1,x2,y1,y2):
        x = np.sqrt(x1**2 + x2**2)
        y = np.sqrt(y1**2 + y2**2)
        return 0.5*np.abs(x-y)**2 - self.potential(x)

    def hessian_potential(self, x1, x2):

        r2 = x1**2 + x2**2
        r4 = r2**2

        psi_xx = (r2 - 2*x1**2)/r4
        psi_yy = (r2 - 2*x2**2)/r4
        psi_xy = (-2*x1*x2)/r4

        return psi_xx, psi_xy, psi_yy


    def DimensionlessFrequency(lensmass,f):
    
        const = 8*np.pi*G/c**3

        w = const*lensmass*M_sun*f

        return w



    def Point_multi(w, x):
        """
        A point mass lens analytical amplification factor from Takahashi(2003). The function is suitable for 
        microlensing (fixed w and time varyinig impact paramater)

        Parameter :
        w : Dimensionless frequency
        y : Impact parameter

        returns : the magnitude of the amplification factor
        """
        Freal = []
        Fimag = []
        y = np.abs(x)
        for i in range(len(w)):
            
            for j in range(len(y)):
                #y[i] = np.abs(y[i])
                xm = (y[j] + np.sqrt(y[j]**2 + 4)) / 2
                phim = (xm - y[j])**2 / 2 - np.log(xm)
                expon = np.exp(np.pi * w[i] / 4 + 1j * w[i] / 2 * (np.log(w[i] / 2) - 2 * phim))
                mp_w = mpmath.mpc(w[i])
                mp_y = mpmath.mpc(y[j])
                f = expon * mpmath.gamma(1 - 1j / 2 *w) * hyp1f1(1j / 2 * w, 1, 1j / 2 * w * y**2,100)
                Freal.append(f.real)
                Fimag.append(f.imag)   
        return Freal, Fimag


    @staticmethod
    def Point(w,y):

        w = mpc(w)
        y = mpc(y)
        xm = (y + sqrt(y**2 + 4)) / 2
        phim = (xm - y)**2 / 2 - log(xm)
        expon = exp(pi * w / 4 + 1j * w / 2 * (log(w / 2) - 2 * phim))
        #mp_w = mpmath.mpc(w)
        #mp_y = mpmath.mpc(y)
        f = expon *gamma(1 - 1j / 2 * w) * hyp1f1(1j / 2 * w, 1, 1j / 2 * w * y**2, maxterms=1e+9)
        #Fan.append(complex(f.real, f.imag)) 
        #F = complex(f.real, f.imag)
        return f



    
    def psi(x):
        
        return log(x)
        

    def func(x, w, y):
        # x: real integration variable (mpf)
        # w, y, amp, core, p: mpc or mpf
        sqrt2x = sqrt(2 * x)
        arg_bessel = w * y * sqrt2x
        bessel_val = besselj(0, arg_bessel)
        psi_val = psi(sqrt2x)
        return bessel_val * exp(-1j * w * psi_val)

    def func2(x, w, y):
        return self.func(x, w, y) * exp(1j * w * x)

    def dfunc(x, w, y):
        sqrt2x = sqrt(2 * x)
        psi_val = self.psi(sqrt2x)
        J1 = besselj(1, w * y * sqrt2x)
        prefactor = -w * y / sqrt2x
        return prefactor * J1 * exp(-1j * w * psi_val) - (1j * w / (2 * x)) * func(x, w, y)

    def ddfunc(x, w, y):
        sqrt2x = sqrt(2 * x)
        psi_val = self.psi(sqrt2x)
        #denom = (sqrt2x**2 + core**2)**(1 - p/2)
        #dpsi = amp * p * sqrt2x / denom
        
        # derivative squared term:
        #d2psi = amp * p * (1 - p / 2) * (2 * x)**(-0.5) * (1 - 2 * x / (2 * x + core**2)) / (2 * x + core**2)**(1 - p/2)
        
        term1 = (w * y) / (2 * x * sqrt2x) * (2 + 1j * w) * besselj(1, w * y * sqrt2x) * exp(-1j * w * psi_val)
        term2 = -1 / (2 * x) * (w**2 * y**2 - 1j * w / x) * self.func(x, w, y)
        term3 = -1j * w / (2 * x) * self.dfunc(x, w, y)
        return term1 + term2 + term3

    def TakPoint(w, y):

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
        zz = quad(lambda x: func2(x, w, y), [a, b], error=True, maxdegree=10)
        zz_val = zz[0]  # integral value
        
        # Add tail correction terms (at b)
        tail = (-self.func(b, w, y) / (1j * w) * exp(1j * w * b)
                - self.dfunc(b, w, y) / (w**2) * exp(1j * w * b)
                + self.ddfunc(b, w, y) / (1j * w**3) * exp(1j * w * b))
        
        zz_val += tail
            
        return -1j * w * exp(0.5 * 1j * w * y**2) * zz_val
    


def _worker(args):
    i, j, w, y = args
    val = PML.Point(w, y)
    return i, j, val.real, val.imag


def compute_point_lens_grid(w_values, y_values, hdf5_filename, num_processes=None):
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
    tasks = [(i, j, w_values[i], y_values[j]) for i in range(W_len) for j in range(Y_len)]

    logger.info(f" Output file: {hdf5_filename}")
    logger.info(f" Grid size: {W_len} × {Y_len} → {total_points} points")
    logger.info("Launching parallel computation...")

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



