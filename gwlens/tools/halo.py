import numpy as np

class NFW:
    """
    Class to represent an NFW halo and compute related quantities.
    All distances in kpc, masses in solar masses to be consistent with CW analysis.
    """
    def __init__(self, M200, H0=70.0):
        """
        Parameters:
        - M200: Virial mass in solar masses (e.g. 1e14)
        - c: concentration parameter (e.g. 5)
        - H0: Hubble constant in km/s/Mpc (default 70)
        """
        self.M200 = M200
        self.c = self.concentration_parameter()
        self.H0 = H0  # in km/s/Mpc
        self.G = 4.302e-6  # Gravitational constant in (kpc * km^2) / (s^2 * Msun)

        self.rho_c = self.compute_critical_density()
        self.r200 = self.compute_r200()
        self.rs = self.r200 / self.c
        self.rhos = self.compute_rhos()

    def concentration_parameter(self,zl=0, h= 0.7):
        term = (self.M200*h/(1e+14))**(-0.13)
        return 8/(1+zl) * term

    def compute_critical_density(self):
        """Compute critical density of the Universe at z = 0 (in Msun/kpc^3)"""
        H0_kpc = self.H0 / 1000  # Convert to km/s/kpc
        rho_c = 3 * H0_kpc**2 / (8 * np.pi * self.G)
        return rho_c

    def compute_r200(self):
        """Compute r_200 in kpc"""
        return (3 * self.M200 / (4 * np.pi * 200 * self.rho_c))**(1/3)

    def compute_rhos(self):
        """Compute the NFW scale density rho_s (in Msun/kpc^3)"""
        f_c = np.log(1 + self.c) - self.c / (1 + self.c)
        return (200 / 3) * self.rho_c * (self.c**3) / f_c

    def mass_enclosed(self, r):
        """
        Compute enclosed mass at radius r (in kpc)
        Returns: Mass in Msun
        """
        x = r / self.rs
        f_x = np.log(1 + x) - x / (1 + x)
        return 4 * np.pi * self.rhos * self.rs**3 * f_x

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

        return 3.71e-4 * self.mass_enclosed(self.rs)*f

    def scaling_ratio(self,D):

        """
        The ratio used in GLoW to produce u.
        
        """
        return 2.4e-8 * np.sqrt(self.mass_enclosed(self.rs)*D)/self.rs

    

    def summary(self):
        print(f"NFW Halo Summary:")
        print(f"  M_200       = {self.M200:.2e} Msun")
        print(f"  c           = {self.c}")
        print(f"  H_0         = {self.H0} km/s/Mpc")
        print(f"  rho_c       = {self.rho_c:.5f} Msun/kpc^3")
        print(f"  r_200       = {self.r200:.5f} kpc")
        print(f"  r_s         = {self.rs:.5f} kpc")
        print(f"  rho_s       = {self.rhos:.2e} Msun/kpc^3")
        print()



#  Usage
if __name__ == "__main__":
    halo = NFW(M200=1e+8)
    halo.summary()
    print("Dim, freq")
    w= halo.Dimensionless_Frequency(f = 100)
    print(w)

    print("Scaling ratio", halo.scaling_ratio(0.5))


    x = 69
    m100rs = halo.mass_enclosed_at_xrs(x)
    print(f"Mass enclosed within {x} r_s = {x * halo.rs:.2f} kpc:")
    print(f"  M(<{x} r_s) = {m100rs:.2e} Msun")



