"""
Fast amplification factor for an ARBITRARY axisymmetric lens psi(x), rotated-tail method.

    F(w, y) = -i w e^{i w y^2/2} Int_0^inf f(u) e^{i w u} du,   u = x^2/2,
    f(u)    = J0(w y x) exp(-i w psi(x)),                       x = sqrt(2u)

  head  [0, u_b]       Gauss-Legendre on pieces holding <= one cycle of the full phase,
                       real arithmetic (scipy j0, real psi)
  tail  u_b + i t      exact rotation (Cauchy), scaled Gauss-Laguerre with the true decay rate
                       a = w [1 - (y + psi'(x_b)) / x_b]

What you supply
---------------
  psi(x)   the lensing potential as a function of the radius x. It is called with REAL arrays
           (head) and with COMPLEX arrays (tail), so write it with analytic numpy operations
           only: + - * / **, np.log, np.sqrt, np.exp, np.arccos, np.arcsinh, ...
           NOT np.abs, np.hypot, np.maximum, np.where, np.real, or if/else on x.
           If the formula needs complex intermediates on the real axis (e.g. arccos of numbers
           > 1, as for NFW inside the scale radius), cast to complex inside; the head uses the
           real part.
  psi_real (optional) a faster real-arithmetic version of psi for real x > 0.
  q_inf    (optional) lim psi'(x)/x at large x (a mass sheet kappa); estimated if omitted.

Usage
-----
    lens = Potential(lambda x: 0.5 * np.log(x**2 + 0.01))          # any analytic psi(x)
    F    = amplification(10.0, 0.3, lens)                           # one frequency
    Fs   = amplification_many(np.geomspace(1e-2, 1e3, 500), 0.3, lens)   # grid reused
    solver = Solver1D(lens, y=0.3, wmin=1e-2, wmax=1e3); solver(w)      # keep the grid yourself

Ready-made potentials (add them with +): point_mass, sis, cis, nfw, mass_sheet, power_law.
"""
import warnings
import numpy as np
from scipy.special import j0, jv, roots_legendre, roots_laguerre


# ================================================================== potentials
class Potential:
    """An axisymmetric lensing potential psi(x).

    psi       callable, analytic (accepts complex arrays)
    psi_real  optional callable for real x > 0 in real arithmetic (speed)
    q_inf     optional lim psi'(x)/x (external convergence); None = estimate
    """

    def __init__(self, psi, psi_real=None, q_inf=None):
        self._psi = [psi]
        self._psi_real = [psi_real]
        self.q_inf = q_inf
        self._checked = False

    def __call__(self, x):
        out = self._psi[0](x)
        for f in self._psi[1:]:
            out = out + f(x)
        return out

    def real(self, x):
        """psi on the real axis (x > 0), real arithmetic when available."""
        out = 0.0
        for f, fr in zip(self._psi, self._psi_real):
            out = out + (fr(x) if fr is not None else np.real(f(x)))
        return out

    def __add__(self, other):
        new = Potential.__new__(Potential)
        new._psi = self._psi + other._psi
        new._psi_real = self._psi_real + other._psi_real
        new.q_inf = None if (self.q_inf is None or other.q_inf is None) else self.q_inf + other.q_inf
        new._checked = False
        return new

    def dpsi(self, x, h=1e-20):
        """psi'(x) for real x > 0 by the complex step (exact to rounding)."""
        x = np.asarray(x, dtype=float)
        return np.imag(self(x + 1j * h)) / h

    def check_analytic(self, verbose=False):
        """Compare complex-step and finite-difference derivatives; warn if psi is not analytic."""
        x = np.geomspace(0.03, 30, 60)
        try:
            with np.errstate(all="ignore"):
                d_cs = self.dpsi(x)
        except TypeError as err:
            raise TypeError("psi(x) must accept complex arrays (it is evaluated on the rotated "
                            f"tail); use analytic numpy operations only. ({err})") from None
        h = 1e-6 * x
        with np.errstate(all="ignore"):
            d_fd = (self.real(x + h) - self.real(x - h)) / (2 * h)
            im = np.imag(self(x + 0j))
            re = np.real(self(x + 0j))
            r_ok = np.max(np.abs(self.real(x) - re) / (1 + np.abs(re)))
        mism = np.max(np.abs(d_cs - d_fd) / (1 + np.abs(d_fd)))
        imag = np.max(np.abs(im) / (1 + np.abs(re)))
        if not np.isfinite(mism) or mism > 1e-5 or imag > 1e-10 or r_ok > 1e-10:
            warnings.warn(
                f"psi does not look analytic (derivative mismatch {mism:.1e}, Im psi on the real "
                f"axis {imag:.1e}, psi_real vs psi {r_ok:.1e}). Use only analytic numpy operations "
                "(no abs/hypot/where/real/if-else on x); otherwise the rotated tail is invalid.")
        if verbose:
            print(f"check_analytic: derivative mismatch {mism:.1e}, Im psi {imag:.1e}, psi_real {r_ok:.1e}")
        self._checked = True
        return mism


def point_mass(m=1.0):
    """psi = m ln x."""
    m = float(m)
    return Potential(lambda x: m * np.log(x), lambda x: m * np.log(x), q_inf=0.0)


def sis(psi0=1.0):
    """psi = psi0 x."""
    p = float(psi0)
    return Potential(lambda x: p * x, lambda x: p * x, q_inf=0.0)


def cis(psi0=1.0, rc=0.05):
    """Cored isothermal sphere (GLoW convention): psi0 [r + rc ln(2 rc/(r + rc))], r = sqrt(x^2 + rc^2)."""
    p, rc = float(psi0), float(rc)

    def f(x):
        r = np.sqrt(x * x + rc * rc)
        return p * (r + rc * np.log(2 * rc / (r + rc)))
    return Potential(f, f, q_inf=0.0)


def nfw(k=1.0, psi0=1.0):
    """NFW: 0.5 psi0 [ln^2(v/2) + arccos^2(1/v)], v = k x  (k = xi0 / r_s)."""
    p, k = float(psi0), float(k)

    def f(x):
        v = k * x + 0j
        return 0.5 * p * (np.log(v / 2)**2 + np.arccos(1 / v)**2)

    def f_real(x):
        v = k * np.asarray(x, dtype=float)
        out = np.empty_like(v)
        lo, hi = v < 1, v > 1
        L = np.log(v / 2)
        out[lo] = 0.5 * (L[lo]**2 - np.arccosh(1 / v[lo])**2)
        out[hi] = 0.5 * (L[hi]**2 + np.arccos(1 / v[hi])**2)
        out[~(lo | hi)] = 0.5 * np.log(0.5)**2
        return p * out
    return Potential(f, f_real, q_inf=0.0)


def mass_sheet(kappa):
    """psi = kappa x^2 / 2 (external convergence). Requires kappa < 1."""
    k = float(kappa)
    return Potential(lambda x: 0.5 * k * x * x, lambda x: 0.5 * k * x * x, q_inf=k)


def power_law(psi0=1.0, p=1.0):
    """psi = psi0 x^p / p  (0 < p < 2): deflection psi0 x^(p-1); p = 1 is the SIS."""
    a, p = float(psi0), float(p)
    return Potential(lambda x: a * x**p / p, lambda x: a * np.asarray(x, float)**p / p, q_inf=0.0)


# ================================================================== solver
_GL, _LAG = {}, {}


def _gl(n):
    if n not in _GL:
        _GL[n] = roots_legendre(n)
    return _GL[n]


def _lag(n):
    if n not in _LAG:
        _LAG[n] = roots_laguerre(n)
    return _LAG[n]


class Solver1D:
    """Everything that does not depend on w, for one (lens, y). Call it with w in [wmin, wmax].

    ngl, nlag   Gauss-Legendre nodes per head piece / Gauss-Laguerre nodes on the tail
                (ngl=20 gives ~1e-13; 16 ~1e-10; 12 ~1e-7 for strong lenses)
    xb_min      lower bound on the split radius (convergence test: F must not change)
    """

    def __init__(self, lens, y, wmin=1e-3, wmax=1e3, ngl=20, nlag=30, xb_min=0.0):
        if not isinstance(lens, Potential):
            lens = Potential(lens)
        if not lens._checked:
            lens.check_analytic()
        self.lens, self.y = lens, float(y)
        self.wmin, self.wmax = float(wmin), float(wmax)
        self.GX, self.GW = _gl(ngl)
        self.LX, self.LW = _lag(nlag)

        # ---- split: (y + psi'(x_b))/x_b <= thr, thr = 1/2 (or (1 + kappa)/2 with a mass sheet)
        q = lens.q_inf
        if q is None:
            q = max(float(lens.dpsi(R) / R) for R in (1e3, 1e4))
        if q >= 1:
            raise ValueError(f"lim psi'/x = {q:.3f} >= 1: the tail cannot be rotated upward.")
        self.thr = max(0.5, 0.5 * (1 + q))
        xb = max(3.0, 2 * self.y, float(xb_min))
        for _ in range(400):
            if (self.y + lens.dpsi(xb)) / xb <= self.thr:
                break
            xb *= 1.1
        else:
            raise RuntimeError("no split point found; does psi grow faster than x^2/2?")
        self.ub0 = xb * xb / 2
        self.ub_max = max(self.ub0, 20 / self.wmin)

        # ---- phase grid, valid for every w in [wmin, wmax]: V(u; w) = w * V1(u)
        self.u_floor = 1e-15 / max(self.wmax, 1.0)          # [0, u_floor] contributes < 1e-15
        n_lin = int(max(3000, 8 * self.wmax * self.ub0 / (2 * np.pi)))
        ug = np.unique(np.concatenate([np.linspace(self.u_floor, self.ub0, n_lin),
                                       np.geomspace(self.u_floor, self.ub_max, 20000)]))
        x = np.sqrt(2 * ug)
        with np.errstate(all="ignore"):
            ph1 = ug - lens.real(x)                          # phase per unit w (without J0)
        d = np.abs(np.diff(ph1))
        d[~np.isfinite(d)] = 0.0
        self.ug = ug
        self.V1 = np.concatenate([[0.0], np.cumsum(d)]) + self.y * (x - x[0])

    def edges(self, w, ub):
        V1_ub = np.interp(ub, self.ug, self.V1)
        e = np.interp(np.arange(0, w * V1_ub, 2 * np.pi) / w, self.V1, self.ug)
        # geometric grading towards the centre: consecutive edges differ by at most a factor e,
        # so cusps (SIS) and log singularities (point mass, NFW) are smooth in every piece
        n_geo = int(np.ceil(np.log(ub / self.u_floor))) + 1
        geo = np.geomspace(self.u_floor, ub, n_geo)
        return np.unique(np.concatenate([[0.0], geo, e[e < ub], np.linspace(0, ub, 5), [ub]]))

    def __call__(self, w):
        w = float(w)
        if not (self.wmin * (1 - 1e-12) <= w <= self.wmax * (1 + 1e-12)):
            raise ValueError(f"w = {w} outside [{self.wmin}, {self.wmax}] of this solver")
        y, lens = self.y, self.lens
        ub = max(self.ub0, 20 / w)                           # small-w rule
        xb = np.sqrt(2 * ub)

        # head: real arithmetic
        e = self.edges(w, ub)
        mid, half = (e[1:] + e[:-1]) / 2, (e[1:] - e[:-1]) / 2
        uu = (mid[:, None] + half[:, None] * self.GX[None, :]).ravel()
        ww = (half[:, None] * self.GW[None, :]).ravel()
        x = np.sqrt(2 * uu)
        head = np.sum(ww * j0(w * y * x) * np.exp(1j * w * (uu - lens.real(x))))

        # tail: complex, nlag points, scaled by the true decay rate
        a = w * (1 - (y + lens.dpsi(xb)) / xb)
        ut = ub + 1j * self.LX / a
        xt = np.sqrt(2 * ut)
        g = jv(0, w * y * xt) * np.exp(-1j * w * lens(xt)) * np.exp(-(w / a - 1) * self.LX)
        tail = 1j * np.exp(1j * w * ub) / a * np.sum(self.LW * g)

        return -1j * w * np.exp(0.5j * w * y * y) * (head + tail)


def amplification(w, y, lens, ngl=20, nlag=30, xb_min=0.0, estimate_error=False):
    """F(w, y) at one frequency. estimate_error=True also returns |F - F(split moved x1.5, ngl+8)|."""
    s = Solver1D(lens, y, wmin=w, wmax=w, ngl=ngl, nlag=nlag, xb_min=xb_min)
    F = s(w)
    if not estimate_error:
        return F
    F2 = Solver1D(lens, y, wmin=w, wmax=w, ngl=ngl + 8, nlag=nlag + 10,
                  xb_min=1.5 * np.sqrt(2 * max(s.ub0, 20 / w)))(w)
    return F, abs(F2 - F)


def amplification_many(ws, y, lens, ngl=20, nlag=30, xb_min=0.0):
    """F on an array of frequencies; the phase grid is built once and reused."""
    ws = np.atleast_1d(np.asarray(ws, dtype=float))
    s = Solver1D(lens, y, wmin=ws.min(), wmax=ws.max(), ngl=ngl, nlag=nlag, xb_min=xb_min)
    return np.array([s(w) for w in ws])


# ================================================================== example
if __name__ == "__main__":
    # a potential nobody coded for you: point mass inside a cored isothermal halo
    def my_psi(x):
        m, psi0, rc = 0.3, 0.8, 0.2
        r = np.sqrt(x * x + rc * rc)
        return m * np.log(x) + psi0 * (r + rc * np.log(2 * rc / (r + rc)))

    lens = Potential(my_psi)
    for w in (0.1, 1.0, 10.0, 100.0):
        F, err = amplification(w, 0.5, lens, estimate_error=True)
        print(f"w = {w:6.1f}   F = {F.real:+.12f} {F.imag:+.12f}i   |F| = {abs(F):.12f}   (self-check {err:.1e})")