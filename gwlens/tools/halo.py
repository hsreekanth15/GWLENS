import numpy as np
from astropy import units as u
from astropy.constants import G,c
from astropy.cosmology import FlatLambdaCDM

class NFW:
    """
    Class to represent an NFW halo and compute related quantities.
    All distances in kpc, masses in solar masses to be consistent with CW analysis.
    Mpc are also accepted but the results are printed in astropy Quantity allowing conversions
    """
    def __init__(self, M200, H0=70.0,zl= 0):
        """
        Parameters:
        - M200: Virial mass in solar masses (e.g. 1e14)
        - c: concentration parameter (e.g. 5)
        - H0: Hubble constant in km/s/Mpc (default 70)
        - Redshift of the lens
        """
        self.M200 = M200 * u.Msun
        self.zl = zl
        self.cp = self.concentration_parameter()
        self.H0 = H0 * (u.km/u.s/u.Mpc)
        #self.G = 4.302e-6  # Gravitational constant in (kpc * km^2) / (s^2 * Msun)
        self.G = G.to(u.kpc * (u.km/u.s)**2  /u.Msun)
        self.c = c.to(u.km/u.s)
        self.rho_c = self.compute_critical_density()
        self.r200 = self.compute_r200()
        self.rs = self.r200 / self.cp
        self.rhos = self.compute_rhos()

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

    def concentration_parameter(self, h= 0.7):
        term = (self.M200/(1e+14 * u.Msun) * h)**(-0.13)
        return 8/(1+self.zl) * term

    def compute_critical_density(self):
        """Compute critical density of the Universe at z = 0 (in Msun/kpc^3)"""
        #H0_kpc = self.H0 / 1000  # Convert to km/s/kpc
        rho_c = (3 * self.H0**2 / (8 * np.pi * self.G))
        return rho_c.to(u.Msun/u.kpc**3)

    def compute_r200(self):
        """Compute r_200 in kpc"""
        return (3 * self.M200 / (4 * np.pi * 200 * self.rho_c))**(1/3)

    def compute_rhos(self):
        """Compute the NFW scale density rho_s (in Msun/kpc^3)"""
        f_c = np.log(1 + self.cp) - self.cp / (1 + self.cp)
        return (200 / 3) * self.rho_c * (self.cp**3) / f_c

    def mass_enclosed(self, r):
        """
        Compute enclosed mass at radius r (in kpc)
        Returns: Mass in Msun
        """
        if not isinstance(r, u.Quantity):
            raise TypeError(f"r must be an astropy Quantity (e.g. r*u.kpc). Got: {type(r)}")
        r = r.to(self.rs.unit)
        x = r / self.rs
        f_x = np.log(1 + x) - x / (1 + x)
        return (4 * np.pi * self.rhos * self.rs**3 * f_x).to(u.Msun)

    def mass_enclosed_at_xrs(self, x):
        """
        Compute enclosed mass at r = x * rs
        Parameters:
        - x: multiple of scale radius (e.g. x = 100 for r = 100 rs)
        """
        r = x * self.rs
        return self.mass_enclosed(r)

    def Dimensionless_Frequency(self,f):
        """
        Dimensionless freuquency for the GW lensing studies

        Normalizations are according to GLoW package

        Mass is calculated at the scale radius and measured in solar Mass.
        frequency is in Hertz.

        """
        if not isinstance(f, u.Quantity):
            f = f*u.Hz 

        M = self.mass_enclosed(self.rs)
        
        f = f.to(1/u.s)

        return (24*np.pi*self.G*M*f/self.c**3).decompose()

    def scaling_ratio(self, zs=None, Dl=None, Dls=None, Ds=None, cosmo=None):
        """
        Compute the scaling ratio used in GLoW to produce u.
        Deff = Dl * Dls / Ds is computed via effective_distance().
        """
        Deff = self.effective_distance(Dl=Dl, Dls=Dls, Ds=Ds, zs=zs, cosmo=cosmo)
        constant = 12 * self.G / self.c**2
        return (np.sqrt(constant * self.mass_enclosed(self.rs) * Deff) / self.rs).decompose()

    

    def summary(self):
        print(f"NFW Halo Summary:")
        print(f"  M_200       = {self.M200:.2e} ")
        print(f"  c           = {self.cp}")
        print(f"  H_0         = {self.H0} ")
        print(f"  rho_c       = {self.rho_c:.5f} ")
        print(f"  r_200       = {self.r200:.5f}")
        print(f"  r_s         = {self.rs:.5f} ")
        print(f"  rho_s       = {self.rhos:.2e} ")
        print()



#  Usage
if __name__ == "__main__":
    halo = NFW(M200=1e+6)
    halo.summary()
    print("Dim. freq")
    w= halo.Dimensionless_Frequency(f = 1)
    print(w)

    print("Scaling ratio", halo.scaling_ratio(0.5*u.kpc))


    x = 1
    m100rs = halo.mass_enclosed_at_xrs(x) 
    print(f"Mass enclosed within {x} r_s = {x * halo.rs:.2f} kpc:")
    print(f"  M(<{x} r_s) = {m100rs:.2e} Msun")



