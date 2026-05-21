#author : Sreekanth Harikumar
#A class for Primodial DM mini halo 
# See https://arxiv.org/pdf/2303.17601 and references there in for more details

from .base import Lens
from astropy import units as u
from gwlens.lens import solver
from mpmath import mp,log
from astropy import constants as const
from astropy import units as u

    
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

    def __init__(self, m_PBH, f_PBH, Dl, Ds, Dls):
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






