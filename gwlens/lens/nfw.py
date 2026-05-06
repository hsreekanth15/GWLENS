import numpy as np
import time
import multiprocessing as mpr
import h5py
import os
import configparser
import logging
from .base import Lens
from gwlens.lens import solver
from scipy.optimize import fsolve
from astropy.constants import c,G,M_sun
from joblib import Parallel, delayed
from mpmath import mp,mpf, mpc, sqrt, exp, besselj, quad,log, atan,atanh



# Setup logger (you can customize the path or pass it in)
logger = logging.getLogger("nfw_interpolator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


mp.dps = 200  # set precision to 50 decimal places (arbitrary precision)
eps = 1e-3  # relative accuracy tolerance


class NFW1(Lens):
    """
    Navarro Frenk White Class, a class inherited from the base class Lens
    """
    def potential(self,rs, x):
        scale_radius = rs
        u = scale_radius*x

        if u <= 1:
            return 0.5*(  (np.log(u/2))**2 - (np.arctanh(np.sqrt(1-u**2)))**2 )
        else:
            return 0.5*( (np.log(u/2))**2 + (np.arctan(np.sqrt(u**2 -1)))**2   )
        
    def grad_potential(self, x1, x2, rs):
        """
        Gradient of the NFW lensing potential (= deflection angle vector).

        For a circularly-symmetric lens:
            ∇ψ = α(r)/r · (x1, x2)

        where the scalar deflection angle is:
            α(r) = (1/r) · [ln(u/2) + h(u)],   u = rs·r

        The returned components are:
            ∂ψ/∂x1 = α(r)/r · x1
            ∂ψ/∂x2 = α(r)/r · x2
        """
        r = np.sqrt(x1**2 + x2**2)
        u = rs * r

        h = self._h(u)                             # atanh / atan branch

        # α(r)/r — avoids a separate division by r
        alpha_over_r = (np.log(u / 2.0) + h) / r**2

        return alpha_over_r * x1, alpha_over_r * x2
    

    def deflection(self, x1, x2, rs):
        """
        NFW deflection angle vector  α = ∇ψ.

        For a circularly-symmetric lens the deflection points radially:

            α₁ = α(r)/r · x1,   α₂ = α(r)/r · x2

        where the scalar deflection magnitude is:

            α(r) = (1/r) · [ln(u/2) + h(u)],   u = rs · r

        This is identical to grad_potential — the deflection angle IS
        the gradient of the lensing potential.

        Parameters
        ----------
        x1, x2 : float  – image-plane coordinates
        rs     : float  – dimensionless NFW scale radius

        Returns
        -------
        alpha1, alpha2 : float  – deflection angle components
        """
        return self.grad_potential(x1, x2, rs)  
    
    def Time_Delay(self,y1, y2,rs, Ml):

        """
        
        Computes the physical timedelay between images

        Returns: timedelay between signals in seconds

        """

        prefactor = 12*G*Ml*M_sun/c**3

        sol = self.solve_images(y1,y2,rs,method="numeric")
        td = []
        for img in sol:
            dimensionlenss_td = img["dimensionless_time_delay"]
            img["timedelay"] = prefactor*dimensionlenss_td
            td.append(img["timedelay"])

        return td
 

    def Fermat_Potential(self,x1,x2,y1,y2,rs):
        x = np.sqrt(x1**2 + x2**2)
        
        return 0.5* ( (x1-y1)**2 + (x2-y2)**2) - self.potential(rs,x)
    def _h(self, u):
        """
        Auxiliary angular function h(u) that appears in deflection and
        mean-convergence expressions.
            h(u) = arctanh(sqrt(1-u²)) / sqrt(1-u²)   for u < 1
            h(u) = 1                                    for u = 1
            h(u) = arctan(sqrt(u²-1))  / sqrt(u²-1)   for u > 1
        """
        if abs(u - 1.0) < 1e-7:
            return 1.0
        elif u < 1.0:
            return np.arctanh(np.sqrt(1.0 - u**2)) / np.sqrt(1.0 - u**2)
        else:
            return np.arctan(np.sqrt(u**2 - 1.0)) / np.sqrt(u**2 - 1.0)
    def _convergence(self, r, rs):
        """
        NFW convergence (surface-mass density in units of Σ_cr):
            κ(r) = (rs²/2) · [1/(u²-1) + h(u)/(1-u²)]   u < 1
            κ(r) = rs² / 6                                 u = 1  (analytic limit)
            κ(r) = (rs²/2) · [1/(u²-1) - h(u)/(u²-1)]   u > 1

        Derived by applying the 2-D Poisson equation to the NFW
        potential: κ = (1/2r) d(r·α)/dr.
        """
        u = rs * r

        if abs(u - 1.0) < 1e-7:              # analytic limit  u → 1
            return rs**2 / 6.0

        elif u < 1.0:
            # 1/(u²-1) is negative here; atanh term compensates → κ > 0
            return (rs**2 / 2.0) * (
                1.0 / (u**2 - 1.0)
                + np.arctanh(np.sqrt(1.0 - u**2)) / (1.0 - u**2)**1.5
            )
        else:
            return (rs**2 / 2.0) * (
                1.0 / (u**2 - 1.0)
                - np.arctan(np.sqrt(u**2 - 1.0)) / (u**2 - 1.0)**1.5
            )
    def _mean_convergence(self, r, rs):
        """
        Mean convergence within projected radius r:
            κ̄(r) = α(r)/r = (1/r²) · [ln(u/2) + h(u)]

        This equals twice the enclosed projected mass divided by π r²
        (in units of Σ_cr), and is directly the deflection angle α = κ̄·r.
        """
        u = rs * r
        return (1.0 / r**2) * (np.log(u / 2.0) + self._h(u))


    def Hessian(self, x1, x2, rs):
        """
        Second derivatives of the NFW lensing potential ψ(x1, x2).

        For a circularly-symmetric lens the Hessian decomposes as:
            ψ_xx = κ + γ cos2φ
            ψ_yy = κ − γ cos2φ        cos2φ = (x1²−x2²)/r²
            ψ_xy =     γ sin2φ         sin2φ = 2 x1 x2 / r²

        where the shear magnitude is  γ = κ̄(r) − κ(r).

        Parameters
        ----------
        x1, x2 : float  – lens-plane coordinates
        rs      : float  – dimensionless NFW scale radius

        Returns
        -------
        psi_xx, psi_xy, psi_yy : float
        """
        r = np.sqrt(x1**2 + x2**2)

        kappa   = self._convergence(r, rs)        # local convergence
        kappa_bar = self._mean_convergence(r, rs) # mean convergence
        gamma   = kappa_bar - kappa               # shear magnitude

        r2 = r**2
        cos2phi = (x1**2 - x2**2) / r2           # = cos(2φ)
        sin2phi = 2.0 * x1 * x2 / r2             # = sin(2φ)

        psi_xx = kappa + gamma * cos2phi
        psi_yy = kappa - gamma * cos2phi
        psi_xy = gamma * sin2phi

        return psi_xx, psi_xy, psi_yy
    def image_type(self, x1, x2, rs):
        """
        Classify an image position using the eigenvalues of the lensing
        Jacobian  A = I − ∇∇ψ.

        Morse index n:
            n = 0   → minimum   (both eigenvalues of A > 0, det A > 0, tr A > 0)
            n = 1/2 → saddle    (det A < 0  ↔  one eigenvalue negative)
            n = 1   → maximum   (both eigenvalues < 0, det A > 0, tr A < 0)
                      (rare for NFW since there is no central core)
        """
        psi_xx, psi_xy, psi_yy = self.Hessian(x1, x2, rs)

        A = np.array([
            [1.0 - psi_xx, -psi_xy],
            [-psi_xy,       1.0 - psi_yy]
        ])
        eigvals = np.linalg.eigvalsh(A)   # always real for symmetric A

        if eigvals[0] * eigvals[1] < 0.0:
            return {"type": "saddle",   "Morse": 0.5}
        elif eigvals[0] > 0.0:
            return {"type": "minimum",  "Morse": 0}
        else:
            return {"type": "maximum",  "Morse": 1}
    
    def _nfw_guesses(self, y1, y2, rs):
        """
        Build a set of starting points for fsolve that covers where NFW
        images actually form.

        NFW images lie:
          • Along the source azimuth φ_s  (same side, larger r)
          • Opposite  azimuth φ_s + π     (counter-image, smaller r)
          • Near the tangential critical curve  r ~ 1/rs  (Einstein ring scale)

        Strategy: radial grid in both directions × angular perturbations.
        """
        y     = np.hypot(y1, y2)
        phi_s = np.arctan2(y2, y1) if y > 1e-8 else 0.0

        # Characteristic radii:  0.2 … 5 × θ_E  (θ_E ~ 1/rs in these units)
        r_scale = 1.0 / rs if rs > 0 else 1.0
        radii   = r_scale * np.array([0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0])

        guesses = []

        # Along and opposite the source direction (NFW main images)
        for r in radii:
            for dphi in [0.0, np.pi]:
                phi = phi_s + dphi
                guesses.append([r * np.cos(phi), r * np.sin(phi)])

        # Tangential arc: ring at r_scale with 8 equally-spaced angles
        for phi in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            guesses.append([r_scale * np.cos(phi), r_scale * np.sin(phi)])

        # Slight off-axis nudges (helps for nearly-aligned configurations)
        if y > 1e-8:
            for sign in [+1.0, -1.0]:
                guesses.append([y1 + sign * 0.5, y2])
                guesses.append([y1,               y2 + sign * 0.5])

        return guesses
    

    def solve_images(self, y1, y2, rs, method="numeric", tol=1e-6):
        """
        Solve for NFW image positions, magnifications, Morse indices,
        and dimensionless time delays.

        Parameters
        ----------
        y1, y2 : float  – source position (lens-plane units)
        rs     : float  – dimensionless NFW scale radius
        method : str    – only "numeric" is meaningful for NFW
                          (no closed-form image positions exist)
        tol    : float  – duplicate-removal threshold in the image plane

        Returns
        -------
        results : list of dict, each containing:
            position               – (x1, x2)
            magnification          – float (signed)
            type                   – "minimum" / "saddle" / "maximum"
            Morse                  – 0 / 0.5 / 1
            fermat_potential       – φ(x) = (x-y)²/2 − ψ(x)
            dimensionless_time_delay – φ(x) − φ_min
            method                 – "numeric"
        """
        results = []

        # ── lens equation: x − ∇ψ(x) = y ───────────────────────────────
        def lens_eq(x):
            gx, gy = self.grad_potential(x[0], x[1], rs)  # <-- rs threaded
            return [x[0] - gx - y1,
                    x[1] - gy - y2]

        # ── collect unique, well-converged roots ─────────────────────────
        found = []      

        for guess in self._nfw_guesses(y1, y2, rs):
            try:
                sol, _, ier, _ = fsolve(lens_eq, guess, full_output=True)
            except Exception:
                continue

            if ier != 1:                               # fsolve did not converge
                continue

            residual = np.max(np.abs(lens_eq(sol)))
            if residual > np.sqrt(tol):                # converged to a non-root
                continue

            if any(np.linalg.norm(sol - s) < tol for s in found):  # duplicate
                continue

            found.append(sol)

        # ── classify each image ──────────────────────────────────────────
        for sol in found:
            x1, x2 = sol
            itype = self.image_type(x1, x2, rs)       # <-- rs threaded
            results.append({
                "position":      (x1, x2),
                "magnification": self.magnification(x1, x2, rs),  # <-- rs
                "type":          itype["type"],
                "Morse":         itype["Morse"],
                "method":        "numeric",
            })

        # ── Fermat potential & relative time delays ──────────────────────
        for img in results:
            x1, x2 = img["position"]
            img["fermat_potential"] = self.Fermat_Potential(x1, x2, y1, y2, rs)  # <-- rs

        if results:
            phi_min = min(img["fermat_potential"] for img in results)
            for img in results:
                img["dimensionless_time_delay"] = img["fermat_potential"] - phi_min

        return results


    def magnification(self, x1, x2,rs):

        psi_xx, psi_xy, psi_yy = self.Hessian(x1, x2,rs)

        A = np.array([
            [1 - psi_xx, -psi_xy],
            [-psi_xy, 1 - psi_yy]
        ])

        detA = np.linalg.det(A)

        return 1.0 / detA
    
    def solve_images(self, y1, y2, rs, method="numeric", tol=1e-6):
        """
        Solve for NFW image positions, magnifications, Morse indices,
        and dimensionless time delays.

        Parameters
        ----------
        y1, y2 : float  – source position (lens-plane units)
        rs     : float  – dimensionless NFW scale radius
        method : str    – only "numeric" is meaningful for NFW
                          (no closed-form image positions exist)
        tol    : float  – duplicate-removal threshold in the image plane

        Returns
        -------
        results : list of dict, each containing:
            position               – (x1, x2)
            magnification          – float (signed)
            type                   – "minimum" / "saddle" / "maximum"
            Morse                  – 0 / 0.5 / 1
            fermat_potential       – φ(x) = (x-y)²/2 − ψ(x)
            dimensionless_time_delay – φ(x) − φ_min
            method                 – "numeric"
        """
        results = []

        # ── lens equation: x − ∇ψ(x) = y ───────────────────────────────
        def lens_eq(x):
            gx, gy = self.grad_potential(x[0], x[1], rs)  # <-- rs threaded
            return [x[0] - gx - y1,
                    x[1] - gy - y2]

        # ── collect unique, well-converged roots ─────────────────────────
        found = []       # list of accepted solution vectors

        for guess in self._nfw_guesses(y1, y2, rs):
            try:
                sol, _, ier, _ = fsolve(lens_eq, guess, full_output=True)
            except Exception:
                continue

            if ier != 1:                               # fsolve did not converge
                continue

            residual = np.max(np.abs(lens_eq(sol)))
            if residual > np.sqrt(tol):                # converged to a non-root
                continue

            if any(np.linalg.norm(sol - s) < tol for s in found):  # duplicate
                continue

            found.append(sol)

        # ── classify each image ──────────────────────────────────────────
        for sol in found:
            x1, x2 = sol
            itype = self.image_type(x1, x2, rs)       # <-- rs threaded
            results.append({
                "position":      (x1, x2),
                "magnification": self.magnification(x1, x2, rs),  # <-- rs
                "type":          itype["type"],
                "Morse":         itype["Morse"],
                "method":        "numeric",
            })

        # ── Fermat potential & relative time delays ──────────────────────
        for img in results:
            x1, x2 = img["position"]
            img["fermat_potential"] = self.Fermat_Potential(x1, x2, y1, y2, rs)  # <-- rs

        if results:
            phi_min = min(img["fermat_potential"] for img in results)
            for img in results:
                img["dimensionless_time_delay"] = img["fermat_potential"] - phi_min

        return results
                
    
    def Amplification(self,w,y,rs,method="numeric"):
        """
        Computes geometric optics amplification factor for Point Mass Lens

        Parameters:
        
        w: Dimensionless frequency
        y: Impact parameter

        type: Nature of the image: minimum, saddle, both

        """

        
        y1 = y
        y2 = 0

        sol = self.solve_images(y1,y2,rs, method= method)

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
    def Dimensionless_Frequency(lensmass,f):

        """
        Computes the dimensionss frequency mentioned in Takahashi 2003

        Parameters:

        lensmass: The mass of the lens in solar units
        f : Frequency of the gravitational wave source in units of Hz

        Returns: The dimensionless frequency 
        
        """
    
        const = 24*np.pi*G/c**3

        w = const*lensmass*M_sun*f

        return w

def psi(x,rs):
    """
    NFW lensing potential defined as per Villarrubia-Rojo et. al. 2024
    The constant k = xi_0/r_s

    To avoid replacing all the r_s with k. I have used k = rs in this form of lensing potential. This is only for testing purpose and hence will be replaced later.


    """

    u = rs*x
    if u <= 1:

        return 0.5*(  (log(u/2))**2 - (atanh(sqrt(1-u**2)))**2 )
    else:

        return 0.5*( (log(u/2))**2 + (atan(sqrt(u**2 -1)))**2   )


def psi_2(x, rs):
    if x == 0:
        return 0
    elif x < 1:
        return rs/2 * ((log(x/2))**2 - (atanh(sqrt(1 - x**2)))**2)
    else:
        return rs/2 * ((log(x/2))**2 +  (atan(sqrt(x**2 - 1)))**2)

#def psi(x,rs):
#
#    bterm = mpf('2.01')   

#    return rs*( (log(x/2*bterm))**2 - (atanh(sqrt(1 - ((x**2) /bterm**2)**2)))**2)

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

    # Convert inputs to mpmath types
    w = mpf(w)        # real number
    y = mpf(y)        # real number
    rs = mpf(rs)      # real number
    a = mpf('0.00001')

    if w <= 1:
        b = mpf('100') / w
    elif 1 < w < 400000:
        b = mpf('1000') / w
    elif 1000 < w < 500000:
        b = mpf('100000') / w
    elif 500000 < w < 10000000:
        b = mpf('1000000') / w
    else:
        b = mpf('100000000') / w
    
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
