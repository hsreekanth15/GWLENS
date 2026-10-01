"""
Defining your own lens for lens2d_general.py: three worked examples.

Recipe
  1. write psi(x1, x2) with analytic numpy operations (it is also called with complex arrays)
  2. list the points where psi is not smooth (point masses, cusps, very small cores)
  3. lens = G.Lens(psi, singular=[...])            # optional: q_inf = kappa + |gamma|
  4. F = G.amplification_many(ws, y, lens)          # or G.amplification(w, y, lens)
     F, err = G.amplification_many(ws, y, lens, estimate_error=True)   # self-check
"""
import time
import numpy as np
import lens2d_general as G


# ---------------------------------------------------------------- 1. SIE + external shear
def sie_plus_shear(b=1.0, q=0.7, phi=0.3, gamma1=0.05, gamma2=-0.03, x0=(0.0, 0.0)):
    """Singular isothermal ellipsoid (Keeton & Kochanek form) rotated by phi, plus external shear."""
    c, s, sq = np.cos(phi), np.sin(phi), np.sqrt(1 - q * q)
    a1, a2 = x0

    def psi(x1, x2):
        u = c * (x1 - a1) + s * (x2 - a2)                 # coordinates along the ellipse axes
        v = -s * (x1 - a1) + c * (x2 - a2)
        om = np.sqrt(q * q * u * u + v * v)
        ax = b * q / sq * np.arctan(sq * u / om)
        ay = b * q / sq * np.arctanh(sq * v / om)
        return u * ax + v * ay + 0.5 * (gamma1 * (x1 * x1 - x2 * x2) + 2 * gamma2 * x1 * x2)
    return G.Lens(psi, singular=[x0], q_inf=np.hypot(gamma1, gamma2))


# ---------------------------------------------------------------- 2. pseudo-elliptical NFW
def elliptical_nfw(k=2.0, psi0=1.0, e=0.15):
    """NFW potential evaluated at an elliptical radius (pseudo-elliptical NFW)."""
    def psi(x1, x2):
        R = np.sqrt((1 - e) * x1 * x1 + (1 + e) * x2 * x2 + 0j)
        v = k * R
        return 0.5 * psi0 * (np.log(v / 2)**2 + np.arccos(1 / v)**2)
    return G.Lens(psi, singular=[(0.0, 0.0)])


# ---------------------------------------------------------------- 3. host halo + subhalos (built from components)
def host_with_subhalos():
    host = G.cis(psi0=1.0, rc=0.1)
    subs = [(0.01, (1.2, 0.3)), (0.005, (-0.8, 0.9)), (0.02, (0.4, -1.1))]
    lens = host
    for m, pos in subs:
        lens = lens + G.point_mass(m, pos)
    return lens


if __name__ == "__main__":
    ws = np.linspace(0.2, 100.0, 500)
    for name, lens, y in (("SIE (q=0.7) + external shear", sie_plus_shear(), (0.15, 0.1)),
                          ("pseudo-elliptical NFW", elliptical_nfw(), (0.2, 0.1)),
                          ("cored halo + 3 point-mass subhalos", host_with_subhalos(), (0.3, 0.2))):
        mism = lens.check_analytic()
        t = time.perf_counter()
        F, err = G.amplification_many(ws, y, lens, estimate_error=True)
        dt = time.perf_counter() - t
        print(f"{name:<36} analytic check {mism:.1e} | {ws.size} frequencies, |F| from {abs(F).min():.3f} to {abs(F).max():.3f}"
              f" | self-check worst {err.max():.1e} | {dt:.1f} s incl. self-check", flush=True)