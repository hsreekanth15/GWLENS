import numpy as np
import time
import multiprocessing as mpr
import h5py
import os
import configparser
import logging
from .base import Lens
from  gwlens.lens import solver
from astropy.constants import c,G
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from scipy.optimize import fsolve
from mpmath import mp, mpc, exp, pi, gamma, factorial, hyp1f1, nstr

# Set desired decimal precision (increase if needed)
mp.dps = 50  # digits of precision


class SIS(Lens):
    def potential(self,x):
        return np.abs(x)
    
    def Fermat_Potential(self, x1, x2,y1,y2):
        x = np.sqrt(x1**2 + x2**2)
        
        return 0.5* ( (x1-y1)**2 + (x2-y2)**2) - self.potential(x)
    
    def Hessian(self, x1, x2):

        r2 = x1**2 + x2**2
        r = np.sqrt(r2)
        r3 = r2*r

        psi_xx = (1/r) - (x1**2)/r3
        psi_yy = (1/r) - (x2**2)/r3
        psi_xy = -(x1*x2)/r3

        return psi_xx, psi_xy, psi_yy
    
    def grad_potential(self, x1, x2):
        r = np.sqrt(x1**2 + x2**2)
        if r < 1e-12:
            raise ValueError("Singularity at origin (SIS)")
        return x1 / r, x2 / r
    
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
    
    def Effective_Lens_Mass(self,sigma, zL, zS):

        from astropy.cosmology import FlatLambdaCDM              
        cosmo = FlatLambdaCDM(H0=69.7, Om0=0.306, Tcmb0=2.725)


        DL         = cosmo.angular_diameter_distance(zL)
        DS         = cosmo.angular_diameter_distance(zS)
        DLS        = cosmo.angular_diameter_distance_z1z2(zL, zS)
        D          = (DL*DLS)/DS
        
        Deff = D.to(u.km)
        sigma = sigma* (u.km/u.s)
        c = 3e+5 * (u.km/u.s) # in km/s

        G_new = G.to(u.km**3 / (u.M_sun * u.s**2)) 
        
        mlz = (4*np.pi**2 * sigma**4)*Deff*(1+ zL)/(c**2  * G_new) 

        return mlz



    
    def Time_Delay(self,y1, y2, sigma, zL, zS):

        """
        
        Computes the physical timedelay between images

        Parameters:

        y1,y2 :  Impact parameters
        sigma : 1-D velocity dispersion of the lens  

        Returns: timedelay between signals in seconds

        """

        G_new = G.to(u.km**3 / (u.M_sun * u.s**2)) 
        c = 3e+5 * (u.km/u.s)

        prefactor = 4*G_new*self.Effective_Lens_Mass(sigma,zL,zS)/(c**3)

        sol = self.solve_images(y1,y2,method="analytic")
        td = []
        for img in sol:
            dimensionlenss_td = img["dimensionless_time_delay"]
            img["timedelay"] = prefactor*dimensionlenss_td
            td.append(img["timedelay"])

        return td

    def Image_count(self, y1, y2, method = "analytic"):
        """
        Total image count is not fixed ( 2 images maximum) for SIS lens model.
        But it can also be calculated directly from the method
        solve_images. The number of independent solutions correspond to number of images.
        
        """

        return len(self.solve_images(y1,y2,method="analytic"))
    
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
    
    def Dimensionless_Frequency(self,f, sigma, Dl, Ds, Dls,zl=None, zs=None):

        #Dl = Dl* (u.kpc)
        #Ds = Ds * (u.kpc)
        #Dls= Ds-Dl
        
        Deff = self.effective_distance(Dl=Dl,Dls=Dls, Ds= Ds, zl= zl,zs=zs)

        sigma_v = sigma* (u.km/u.s)

        c = 3e+5 * (u.km/u.s) # in km/s  
        f = f*(1/u.s)

        w  = 32* np.pi**3  * sigma_v**4 *Deff * f/(c**5)
        return w.value

    def effective_distance(self, Dl=None, Dls=None, Ds=None, zs=None, cosmo=None):
        """
        Compute the effective lensing distance: Deff = Dl * Dls / Ds

        Two modes:
        ─────────────────────────────────────────────────────────────────
        1. NEARBY  (zl = 0 or low-z): provide distances explicitly.
        All three must be astropy Quantities with distance units.

        Example:
            halo.effective_distance(Dl=100*u.kpc, Dls=50*u.kpc, Ds=130*u.kpc)

        2. COSMOLOGICAL (zl > 0): provide only the source redshift zs.
        Angular diameter distances are computed from self.zl and zs.
        Optionally pass a custom astropy cosmology object.

        Example:
            halo.effective_distance(zs=1.5)
            halo.effective_distance(zs=1.5, cosmo=my_cosmo)
        ─────────────────────────────────────────────────────────────────
        Returns: Deff in kpc
        """

        if Dl is not None or Dls is not None or Ds is not None:
            # ── Mode 1: Explicit distances ──────────────────────────────
            missing = [n for n, v in [("Dl", Dl), ("Dls", Dls), ("Ds", Ds)] if v is None]
            if missing:
                raise ValueError(f"Explicit mode requires all three distances. Missing: {missing}")
            
            for name, val in [("Dl", Dl), ("Dls", Dls), ("Ds", Ds)]:
                if not isinstance(val, u.Quantity):
                    raise TypeError(f"{name} must be an astropy Quantity (e.g. 100*u.kpc). Got: {type(val)}")
            
            return (Dl.to(u.kpc) * Dls.to(u.kpc) / Ds.to(u.kpc)).to(u.kpc)

        elif zs is not None:
            # ── Mode 2: Redshift-based ──────────────────────────────────
            if self.zl == 0:
                raise ValueError(
                    "zl = 0: angular diameter distances are undefined. "
                    "Provide Dl, Dls, Ds explicitly instead."
                )
            if zs <= self.zl:
                raise ValueError(f"Source redshift zs={zs} must be greater than lens redshift zl={self.zl}")

            if cosmo is None:
                cosmo = FlatLambdaCDM(
                    H0=self.H0.to(u.km / u.s / u.Mpc).value,
                    Om0=0.3
                )

            Dl  = cosmo.angular_diameter_distance(self.zl).to(u.kpc)
            Ds  = cosmo.angular_diameter_distance(zs).to(u.kpc)
            Dls = cosmo.angular_diameter_distance_z1z2(self.zl, zs).to(u.kpc)

            return (Dl * Dls / Ds).to(u.kpc)

        else:
            raise ValueError(
                "Provide either:\n"
                "  - Explicit distances: Dl, Dls, Ds (as astropy Quantities)\n"
                "  - Source redshift:    zs (float), with zl set in the constructor"
            )

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
    
    def solve_images(self, y1, y2, method="analytic", tol=1e-6):

        results = []

        y = np.sqrt(y1**2 + y2**2)

        if y == 0:
            raise ValueError("Degenerate case: Einstein ring")

        # direction
        yhat1, yhat2 = y1 / y, y2 / y

        # -------------------------
        # ANALYTICAL SOLUTION
        # -------------------------
        if method in ["analytic", "both"]:

            # always one outer image
            x_plus = y + 1

            positions = [
                (x_plus * yhat1, x_plus * yhat2)
            ]

            # second image only if y < 1
            if y < 1:
                x_minus = y - 1  # negative → opposite direction
                positions.append(
                    (x_minus * yhat1, x_minus * yhat2)
                )

            for x1, x2 in positions:
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

        # -------------------------
        # Fermat + delays (reuse your pipeline)
        # -------------------------
        for img in results:
            x1, x2 = img["position"]
            phi = self.Fermat_Potential(x1, x2, y1, y2)
            img["fermat_potential"] = phi

        phi_min = min(img["fermat_potential"] for img in results)

        for img in results:
            img["dimensionless_time_delay"] = img["fermat_potential"] - phi_min

        return results
    
    def sample(self,N=10000,y_max=2.0,y_min=1e-3,sigma_v_range=(150, 300), zL=0.5,zS=2.0,seed=None):
        """
        Sample SIS lensing observables.

        Returns:
            dict with:
                y              : impact parameter
                mu1, mu2       : magnifications
                x1, x2         : image positions
                delta_t        : time delay (arbitrary units)
                sigma_v        : velocity dispersion
        """

        rng = np.random.default_rng(seed)

        # --- 1. Sample impact parameter (uniform in area → p(y) ∝ y)
        rand = rng.random(N)
        y = np.sqrt(rand) * y_max
        y = np.clip(y, y_min, None)

        # --- 2. Image positions
        x1 = y + 1.0
        x2 = y - 1.0   # negative side

        # --- 3. Magnifications
        mu1 = 1.0 + 1.0 / y
        mu2 = 1.0 - 1.0 / y

        # signed magnification for image 2 (can be negative parity)
        mu2_signed = mu2

        # absolute magnification (what you often use observationally)
        mu2_abs = np.abs(mu2)

        # --- 4. Fermat potential (dimensionless)
        def fermat(x, y):
            return 0.5 * (x - y)**2 - np.abs(x)

        phi1 = fermat(x1, y)
        phi2 = fermat(x2, y)

        delta_phi = phi2 - phi1   # dimensionless time delay

        # --- 5. Sample velocity dispersion
        sigma_v = rng.uniform(sigma_v_range[0], sigma_v_range[1], N)

        G_new = G.to(u.km**3 / (u.M_sun * u.s**2)) 
        c = 3e+5 * (u.km/u.s)

        prefactor = 4*G_new*self.Effective_Lens_Mass(sigma_v,zL,zS)/(c**3)

        # --- 6. Physical scaling of time delay
        # Δt ∝ (σ_v^4) * Δφ   (SIS scaling)
        delta_t = prefactor*delta_phi

        return {
            "y": y,
            "x1": x1,
            "x2": x2,
            "mu1": mu1,
            "mu2_signed": mu2_signed,
            "mu2_abs": mu2_abs,
            "delta_t": delta_t,
            "sigma_v": sigma_v
        }
            
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
