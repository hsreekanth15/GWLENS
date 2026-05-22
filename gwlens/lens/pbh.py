#author : Sreekanth Harikumar
#A class for Primodial DM mini halo 
# See https://arxiv.org/pdf/2303.17601 and references there in for more details

from .base import Lens
from astropy import units as u
from gwlens.lens import solver
from mpmath import mp,log,sqrt
from astropy import constants as const
from astropy import units as u
from gwlens.tools import logutils
import h5py
import configparser
import time
import multiprocessing as mpr

    
class PBH:
    """
    Dressed PBH gravitational lens: point mass + dark matter minihalo.

    Parameters
    ----------
    m_PBH : float     — PBH mass in solar masses
    f_PBH : float     — fraction of dark matter in PBHs  (0 < f_PBH ≤ 1)
    Dl    : Quantity  — observer–lens distance   (e.g. 1.0 * u.kpc)
    Ds    : Quantity  — observer–source distance
    Dls   : Quantity  — lens–source distance
    """

    def __init__(self, m_PBH, f_PBH, Dl= 1 *(u.kpc), Ds=10*(u.kpc), Dls=9*(u.kpc)):
        self.m_PBH = float(m_PBH)   # numerical value in M_sun
        self.f_PBH = float(f_PBH)
        self.Dl    = Dl.to(u.m)
        self.Ds    = Ds.to(u.m)
        self.Dls   = Dls.to(u.m)

        # Derived — computed once at construction
        self.Rh  = self.proper_size()     # halo radius        [Quantity, pc]
        self.xi0 = self.einstein_radius() # physical Einstein radius [Quantity, pc]

    # ── Halo structure ─────────────────────────────────────────────────────

    def proper_size(self):
        """
        Virial radius of the PBH minihalo R_h  [parsec].
        Urrutia+2023.
        """
        return (14.0 * self.m_PBH**(1/3)
                * min(1.0, 1e-3 * self.f_PBH**(-4/3))
                * u.pc)

    def minihalo_mass(self):
        """Minihalo mass M_h  [solar masses, float]."""
        return ((1.0 - self.f_PBH) * self.m_PBH
                * min(50.0, 0.28 / self.f_PBH))

    # ── Einstein radius ────────────────────────────────────────────────────

    def einstein_radius(self):
        """
        Physical Einstein radius  ξ₀ = D_L · θ_E  [parsec], full units.

        G=c=1  →  θ_E² = 4M · D_LS / (D_L D_S)
        Full   →  θ_E² = (4GM/c²) · D_LS / (D_L D_S)   ← used here

        The factor G/c² ≈ 7.4×10⁻²⁸ m kg⁻¹ is the only place the unit
        system enters; everything else is already dimensionless.
        """
        m_kg   = (self.m_PBH * u.Msun).to(u.kg)
        xi0_sq = (4.0 * const.G * m_kg / const.c**2) * (self.Dl * self.Dls / self.Ds)
        xi0_sq_decomposed = xi0_sq.decompose()
        xi0_value = sqrt(xi0_sq_decomposed.value)
        return (xi0_value * xi0_sq_decomposed.unit**0.5).to(u.pc)

    # ── Lensing potentials ─────────────────────────────────────────────────

    def psi_point(self, x):
        """
        x will come in as mpf from mp.quad().
        Keep it in mpmath throughout.
        """
        # Don't convert — x is already mpf from quad()
        if x <= 0:  # Works fine with mpf
            raise ValueError("x must be strictly positive")
        return log(x)  # ✓ Returns high-precision result

    def psi_halo(self, x):
        """Same — x is already mpf from quad()"""

    
        m_ratio = self.m_PBH                                     # m_PBH / M_sun
        R_ratio = (self.Rh / (1.0 * u.pc)).decompose().value    # R_h   / pc

        prefactor = (12.0 * (1.0 - self.f_PBH)
                     * m_ratio**(1/4)
                     * R_ratio**(3/4))

        # Use mpmath conditionals instead of NumPy where()
        inner = (x * self.xi0 / self.Rh).decompose().value
        
        if inner <= 0:
            bracket = mp.mpf(0)
        elif inner <= 1:
            bracket = inner**(mp.mpf(3)/mp.mpf(4))
        else:
            bracket = 1 + (mp.mpf(3)/mp.mpf(4)) * log(inner)
        
        return prefactor * bracket  # ✓ High precision preserved

    def psi(self, x):
        """
        Total lensing potential  ψ(x) = ψ_PM(x) + ψ_h(x)  [dimensionless].
        """
        return self.psi_point(x) + self.psi_halo(x)
    
    def amplification(self, w, y):
        """
        Arbitrary precision calculation of the amplification factor F(w, y) using mpmath's quad()
        """
        amp = solver.Solver_AS(w, y, self.psi)
        return amp


def compute_pbh_lens_grid(w_values, y_values,f_pbh_values,hdf5_filename, num_processes=None):
    """
    Computes the PBH lensing amplification factor over a grid
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
    logger = logutils.setup_logger(output_folder, logger_name="Dressed_PBH_amplification_logger")


    W_len = len(w_values)
    Y_len = len(y_values)
    F_PBH_len = len(f_pbh_values)
    total_points = W_len * Y_len * F_PBH_len

    # Create task list
    tasks = [(i, j, k, w_values[i], y_values[j], f_pbh_values[k]) for i in range(W_len) for j in range(Y_len) for k in range(F_PBH_len)]
    logger.info(f" Output file: {hdf5_filename}")
    logger.info(f" Grid size: {W_len} × {Y_len} × {F_PBH_len} → {total_points} points")
    logger.info("Launching single node parallel computation...")
    logger.warning("This may take a while for large grids. Monitor progress in the logs.")
    logger.warning("MPI parallelization is not implemented in this function; it uses multiprocessing on a single node.")
    
    start_time = time.time()

    def _worker(i, j, k, w, y, f_pbh):
        pbh = PBH(m_PBH=100,f_PBH=f_pbh)
        val = pbh.amplification(w, y)
        return (i, j, k, val.real, val.imag)

    with h5py.File(hdf5_filename, "w") as h5file:
        h5file.create_dataset("w_values", data=w_values)
        h5file.create_dataset("y_values", data=y_values)
        h5file.create_dataset("f_pbh_values", data=f_pbh_values)
        dset = h5file.create_dataset(
            "results",
            shape=(W_len, Y_len,F_PBH_len, 2),
            dtype="f8",
            chunks=(1, Y_len,1, 2)  # optimize for row-wise access
        )

        with mpr.Pool(processes=num_processes or mpr.cpu_count()) as pool:
            for count, (i, j, k, re, im) in enumerate(pool.imap_unordered(_worker, tasks, chunksize=100), 1):
                dset[i, j, k, 0] = re
                dset[i, j, k, 1] = im

                if count % 10000 == 0 or count == total_points:
                    logger.info(f" {count}/{total_points} results written...")

        h5file.flush()

    end_time = time.time()
    logger.info(f" Done in {end_time - start_time:.2f} seconds")
    logger.info(f" Saved to: {hdf5_filename}")
