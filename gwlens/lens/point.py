"""
author: Sreeakanth Harikumar
"""
import os
import numpy as np
import multiprocessing as mpr
from mpmath import hyp1f1,exp, mp,sqrt,mpc,pi,log,gamma, besselj
from importlib.metadata import version, PackageNotFoundError   
from datetime import datetime, timezone                        
import logging
import mpmath
from astropy.constants import c,G,M_sun
from scipy.optimize import fsolve
import configparser
import time
import h5py
from .base import Lens
from gwlens.lens import solver
from gwlens.tools import logutils

#for new method
from scipy.special import jv, roots_legendre, roots_laguerre
NGL = 20
GL_X, GL_W = roots_legendre(NGL)
NLAG = 60
LAG_X, LAG_W = roots_laguerre(NLAG)
########
mp.dps= 60





class PML(Lens):

    def potential(self,x):
        return np.log(np.abs(x))

    def Fermat_Potential(self,x1,x2,y1,y2):
        x = np.sqrt(x1**2 + x2**2)
        
        return 0.5* ( (x1-y1)**2 + (x2-y2)**2) - self.potential(x)

    def Hessian(self, x1, x2):

        r2 = x1**2 + x2**2
        r4 = r2**2

        psi_xx = (r2 - 2*x1**2)/r4
        psi_yy = (r2 - 2*x2**2)/r4
        psi_xy = (-2*x1*x2)/r4

        return psi_xx, psi_xy, psi_yy
    
    def grad_potential(self, x1, x2):
        r2 = x1**2 + x2**2
        return x1 / r2, x2 / r2
    
    def deflection(self, x1, x2):
       return self.grad_potential(x1, x2)
    
    def magnification(self, x1, x2):

        psi_xx, psi_xy, psi_yy = self.Hessian(x1, x2)

        A = np.array([
            [1 - psi_xx, -psi_xy],
            [-psi_xy, 1 - psi_yy]
        ])

        detA = np.linalg.det(A)

        return 1.0 / detA
    
    def Time_Delay(self,y1, y2, Ml):

        """
        
        Computes the physical timedelay between images

        Returns: timedelay between signals in seconds

        """

        prefactor = 4*G*Ml*M_sun/c**3

        sol = self.solve_images(y1,y2,method="analytic")
        td = []
        for img in sol:
            dimensionlenss_td = img["dimensionless_time_delay"]
            img["timedelay"] = prefactor*dimensionlenss_td
            td.append(img["timedelay"])

        return td
 
    
    def Image_count(self):
        """
        Total image count is fixed ( 2 images- Minimum and Saddle) for PML model.
        But it can also be calculated directly from the method
        solve_images. The number of independent solutions correspond to number of images.
        
        """

        return 2.0
    
    def image_type(self, x1, x2):

        psi_xx, psi_xy, psi_yy = self.Hessian(x1, x2)

        A = np.array([
            [1 - psi_xx, -psi_xy],
            [-psi_xy, 1 - psi_yy]
        ])

        eigvals = np.linalg.eigvals(A)

        if np.all(eigvals > 0):
            return {"type":"minimum", "Morse": 0}
        elif np.all(eigvals < 0):
            return {"type": "maximum", "Morse": 2}
        else:
            return {"type": "saddle","Morse":1}
        

    def solve_images(self,y1, y2, method="analytic", tol=1e-6):
        """
        Solve for image positions, magnifications, and types.

        Parameters
        ----------
        y1, y2 : float
            Source position
        method : str
            "analytic", "numeric", or "both"
        tol : float
            Tolerance for removing duplicate numerical solutions

        Returns
        -------
        results : list of dict
            Each dict contains:
                - position: (x1, x2)
                - magnification: float
                - type: "minimum" / "saddle" / "maximum"
                - method: "analytic" or "numeric"
        """



        results = []

        # -------------------------
        # ANALYTICAL SOLUTION
        # -------------------------
        if method in ["analytic", "both"]:

            y = np.sqrt(y1**2 + y2**2)

            if y == 0:
                raise ValueError("Degenerate case: Einstein ring")

            # radial solutions
            x_plus = 0.5 * (y + np.sqrt(y**2 + 4))
            x_minus = 0.5 * (y - np.sqrt(y**2 + 4))

            # direction
            yhat1, yhat2 = y1 / y, y2 / y

            analytic_positions = [
                (x_plus * yhat1, x_plus * yhat2),
                (x_minus * yhat1, x_minus * yhat2)
            ]

            for x1, x2 in analytic_positions:
                info = self.image_type(x1, x2)
                results.append({
                    "position": (x1, x2),
                    "magnification": self.magnification(x1, x2),
                    "type": info["type"],
                    "Morse": info["Morse"],
                    "method": "analytic"
                })

        # -------------------------
        # NUMERICAL SOLUTION
        # -------------------------
        if method in ["numeric", "both"]:

            def lens_eq(x):
                x1, x2 = x
                gx, gy = self.grad_potential(x1, x2)
                return [
                    x1 - gx - y1,
                    x2 - gy - y2
                ]

            guesses = [
                [y1, y2],
                [-y1, -y2],
                [1.0, 0.0],
                [-1.0, 0.0]
            ]

            sols = []

            for guess in guesses:
                sol, info, ier, _ = fsolve(lens_eq, guess, full_output=True)

                if ier == 1:
                    # remove duplicates
                    if not any(np.linalg.norm(sol - s) < tol for s in sols):
                        sols.append(sol)

            for sol in sols:
                x1, x2 = sol
                info = self.image_type(x1, x2)
                results.append({
                    "position": (x1, x2),
                    "magnification": self.magnification(x1, x2),
                    "type": info["type"],
                    "Morse": info["Morse"],
                    "method": "numeric"
                })

        # compute Fermat potential for each image
        for img in results:
            x1, x2 = img["position"]
            phi = self.Fermat_Potential(x1, x2, y1, y2)
            img["fermat_potential"] = phi

        # find minimum (first arrival)
        phi_min = min(img["fermat_potential"] for img in results)

        # compute relative delays
        for img in results:
            img["dimensionless_time_delay"] = img["fermat_potential"] - phi_min
            

              

        return results
    
    def dimensionless_time_delay(self, y1, y2, method="analytic"):
        """
        Compute relative time delays between images.
        First arriving image is normalized to 0.
        """

        images = self.solve_images(y1, y2, method=method)

        # compute Fermat potential for each image
        for img in images:
            x1, x2 = img["position"]
            phi = self.Fermat_Potential(x1, x2, y1, y2)
            img["fermat_potential"] = phi

        # find minimum (first arrival)
        phi_min = min(img["fermat_potential"] for img in images)

        # compute relative delays
        for img in images:
            img["dimensionless_time_delay"] = img["fermat_potential"] - phi_min

        return images


    @staticmethod
    def Dimensionless_Frequency(lensmass,f):

        """
        Computes the dimensionss frequency mentioned in Takahashi 2003

        Parameters:

        lensmass: The mass of the lens in solar units
        f : Frequency of the gravitational wave source in units of Hz

        Returns: The dimensionless frequency 
        
        """
    
        const = 8*np.pi*G/c**3

        w = const*lensmass*M_sun*f

        return w
    
    @staticmethod
    def Analytical_Magnification(y, type="both"):
        """
        Calculates the Point Mass Lens analytical magnification for both images

        parameters:

        y: Impact paraemeter of the lens
        type: nature of the image: minimum, saddle and maximum
        """
        term = (y**2 +2)/(2*y*np.sqrt(y**2 + 4))
        mu_plus = 0.5+term
        mu_minus = 0.5-term
        
        mapping = {
            "minimum": mu_plus,
            "saddle":mu_minus,
            "both": (mu_plus,mu_minus),
        }
        
        try:
            return mapping[type]
        except KeyError:
            raise ValueError(f"Invalid type '{type}'. Expected one of {list(mapping.keys())}") 
        
    
    def Amplification(self,w,y,type="both"):
        """
        Computes geometric optics amplification factor for Point Mass Lens

        Parameters:
        
        w: Dimensionless frequency
        y: Impact parameter

        type: Nature of the image: minimum, saddle, both

        """

        
        y1 = y
        y2 = 0

        sol = self.solve_images(y1,y2, method="analytic")

        mu = []
        td = []
        n = []
        
        for img in sol:
            mag = img["magnification"]
            dimensionless_td = img["dimensionless_time_delay"]
            MorseIndex = img["Morse"]
            mu.append(mag)
            td.append(dimensionless_td)
            n.append(MorseIndex)

        #magnification = self.Analytical_Magnification(y,type=type)


        return solver.GO(w,td,mu,n)

    @staticmethod
    def Point_multi(w, x):
        """
        A point mass lens analytical amplification factor from Takahashi(2003). The function is suitable for 
        microlensing (fixed w and time varyinig impact paramater). A numpy implementation

        Parameter :
        w : Dimensionless frequency
        y : Impact parameter

        returns : The complex ampliification factor

        Caveat: The precision is very poor and hence may fail at higher values of w or wy
        Use Point(w,y) for arbitrary precison to get a better result. 
        The Evaluation is slover but you will get better result.
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
                f = expon * mpmath.gamma(1 - 1j / 2 *w) * hyp1f1(1j / 2 * w, 1, 1j / 2 * w * y**2,100)
                Freal.append(f.real)
                Fimag.append(f.imag)   
        return Freal, Fimag


    @staticmethod
    def Point(w,y):  
        """
        A arbitrary precision mpmath implementation of the Point Mass Lens Analytical Amplification Factor
        The precision can be globally set by calling mp.dps from mpmath library.
        Parameters:

        w: Dimensionless Frequency
        y: Dimensionless Impact parameter

        Returns : Complex amplification factor of chosen precision.

        """

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

# New method##
def f(u, w, y):
    """J0(w y x) * x^{-i w}, x = sqrt(2u); principal branch, fine for Re u > 0."""
    x = np.sqrt(2 * np.asarray(u, dtype=complex))
    return jv(0, w * y * x) * np.exp(-1j * w * np.log(x))

def split_point(y):
    """x_b beyond the outer image x+ = (y + sqrt(y^2+4))/2, with (y + psi'(x_b))/x_b <= 1/2.
    For psi' = 1/x this is x_b >= y + sqrt(y^2 + 2) (~2y at large y; the head costs ~w x_b^2,
    so this is ~2x cheaper than 3y with the same accuracy)."""
    return max(3.0, y + np.sqrt(y * y + 2))

def head_edges(w, y, ub, umin):
    """Edges in [umin, ub]: one cycle of |d phase| + J0 phase per piece."""
    n_log = int(3000 + 8 * (w / 2) * np.log(ub / umin) / (2 * np.pi))
    n_lin = int(max(3000, 8 * w * ub / (2 * np.pi)))
    ug = np.unique(np.concatenate([np.geomspace(umin, ub, n_log), np.linspace(umin, ub, n_lin)]))
    phase = w * (ug - 0.5 * np.log(2 * ug))                     # exact phase of e^{iwu} x^{-iw}
    V = np.concatenate([[0], np.cumsum(np.abs(np.diff(phase)))]) + w * y * np.sqrt(2 * ug)
    V -= V[0]
    edges = np.interp(np.arange(0, V[-1], 2 * np.pi), V, ug)
    # x^{-iw} = (2u)^{-iw/2} varies on a ln-u scale near 0 even when w is small, so also
    # cap every piece at a factor e^{0.5} in u (Gauss-Legendre cannot span decades of u).
    geo = np.exp(np.arange(np.log(umin), np.log(ub), 0.5))
    return np.unique(np.concatenate([edges, geo, np.linspace(umin, ub, 5), [ub]]))


def PointMass(w, y, xb=None):
    w, y = float(w), float(y)
    xb = split_point(y) if xb is None else float(xb)
    # At small w the tail decays over t ~ 1/w, much longer than u_b, and f(u_b + it) changes
    # at t ~ u_b -- too close to t = 0 for Gauss-Laguerre. Pushing u_b out to ~20/w fixes
    # that and is nearly free: the head has few oscillations when w is small.
    ub = max(xb**2 / 2, 20 / w)
    xb = np.sqrt(2 * ub)

    # [0, umin] exactly. Dropping J0 and e^{iwu} there costs ~ w (1 + y^2) umin^2 in the
    # integral, i.e. ~ w^2 (1 + y^2) umin^2 in F; this umin keeps that below ~1e-16.
    umin = 1e-8 / (max(w, 1.0) * np.sqrt(1 + y * y))
    sliver = 2**(-0.5j * w) * umin**(1 - 0.5j * w) / (1 - 0.5j * w)

    # head [umin, ub]
    e = head_edges(w, y, ub, umin)
    mid, half = (e[1:] + e[:-1]) / 2, (e[1:] - e[:-1]) / 2
    uu = (mid[:, None] + half[:, None] * GL_X[None, :]).ravel()
    ww = (half[:, None] * GL_W[None, :]).ravel()
    head = sliver + np.sum(ww * f(uu, w, y) * np.exp(1j * w * uu))

    # tail along ub + i t, t = tau / a
    a = w * (1 - (y + 1 / xb) / xb)
    tail = 1j * np.exp(1j * w * ub) / a * np.sum(
        LAG_W * f(ub + 1j * LAG_X / a, w, y) * np.exp(-(w / a - 1) * LAG_X))

    return -1j * w * np.exp(0.5j * w * y**2) * (head + tail)

def PointMass_many(ws, y, xb=None):
    """F on arrays of frequencies and/or y values."""
    ws, y = np.broadcast_arrays(np.atleast_1d(ws), np.atleast_1d(y))
    return np.array([PointMass(w, yi, xb) for w, yi in zip(ws.flat, y.flat)]).reshape(ws.shape)

def F_analytic(w, y):
    """Exact point-lens F (same phase convention), via mpmath; for checking."""
    from mpmath import mpf, exp, log, gamma, hyp1f1, pi
    w, y = mpf(w), mpf(y)
    return complex(exp(pi * w / 4) * exp(0.5j * w * log(w / 2)) * gamma(1 - 0.5j * w)
                * hyp1f1(0.5j * w, 1, 0.5j * w * y**2, maxterms=10**6))


# New method##
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

    #Store result
    output_folder = config['Output']['outdir']


    # Setup logging
    logger = logutils.setup_logger(output_folder, logger_name="point_lens_logger")


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


#Old Asymptotic series code.
#Now replaced with the new method

""" 
def psi(x):
"""
#Arbitrary precision mpmath version of the lensing potential
"""

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

        """