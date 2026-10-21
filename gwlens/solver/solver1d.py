import os
import math
import time
import signal
import weakref
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager, nullcontext
import numpy as np
from scipy.special import j0, y0, jv, hankel1e, hankel2e, roots_legendre, roots_laguerre
try:
    import finufft
except ImportError:
    finufft = None
try:                                    # optional: keeps BLAS single-threaded inside the small matrix products
    from threadpoolctl import ThreadpoolController
except ImportError:
    ThreadpoolController = None


class Potential:
    """
    (We have to decide if we should keep this class along with solver or move to lens folder and keep the 1d and 2d lensing potentials in the lens folder. For now, we keep it here.)
    An axisymmetric lensing potential psi(x).

    psi       callable, analytic (accepts complex arrays)
    psi_real  optional callable for real x > 0 in real arithmetic (speed)
    q_inf     optional lim psi'(x)/x (external convergence); None = estimate


    Examples
    --------
    Basic usage with a Potential object:

    >>> import numpy as np
    >>> from gwlens import solver1d
    >>> def my_psi(x):
    >>>     return np.log(1 + x**2) 
    >>> # This potential object can be passed to the amplification factor function
    >>> pot = solver1d.Potential(my_psi)


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
        """
        Add two potentials (or a potential and a constant).
        Parameters
        ----------  
        other : Potential or float
            The potential or constant to add.
        Returns:
        --------
        Potential
            A new Potential instance representing the sum of the two potentials.    
        """
        new = Potential.__new__(Potential)
        new._psi = self._psi + other._psi
        new._psi_real = self._psi_real + other._psi_real
        new.q_inf = None if (self.q_inf is None or other.q_inf is None) else self.q_inf + other.q_inf
        new._checked = False
        return new

    def dpsi(self, x, h=1e-20):
        """
        psi'(x) for real x > 0 by the complex step (exact to rounding).
        This is better understood by Taylor expansion in the complex plane.
        Parameters:
        -----------
        x : float or array-like
            The point(s) at which to evaluate the derivative of psi.    
        h : float, optional
            The step size for the complex step (default is 1e-20).
        Returns:
        --------
        float or ndarray
        The derivative of psi at the specified point(s).    

        """
        x = np.asarray(x, dtype=float)
        return np.imag(self(x + 1j * h)) / h

    def check_analytic(self, verbose=False):
        """
        Compare complex-step and finite-difference derivatives; warn if psi is not analytic.
        The rotated tail part of the code requires the function to be analytic. 
        Parameters:
        -----------
        verbose : bool, optional
            If True, print detailed information about the derivative mismatch and other checks (default is False).
        Returns:
        --------
        float
            The maximum relative mismatch between the complex-step and finite-difference derivatives.   
        """
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

### These lenses can be moved to a separate part....
def point_mass(m=1.0):
    """
    Point mass lens potential psi = m ln(x).

    Parameters:
    -----------
    m : float, optional
        The mass of the point mass (default is 1.0).

    Returns:
    --------
    Potential
        A Potential object representing the point mass lens potential.
    """
    m = float(m)
    return Potential(lambda x: m * np.log(x), lambda x: m * np.log(x), q_inf=0.0)


def sis(psi0=1.0):
    """
    SIS (singular isothermal sphere) lens potential psi = psi0 x.

    Parameters:
    -----------
    psi0 : float, optional
        The characteristic surface density (default is 1.0).

    Returns:
    --------
    Potential
        A Potential object representing the SIS lens potential.
    """
    p = float(psi0)
    return Potential(lambda x: p * x, lambda x: p * x, q_inf=0.0)


def cis(psi0=1.0, rc=0.05):
    """
    Cored isothermal sphere (GLoW convention): psi0 [r + rc ln(2 rc/(r + rc))], r = sqrt(x^2 + rc^2).
    Parameters:
    ----------- 
    psi0 : float, optional
        The Amplitude (default is 1.0).
    rc : float, optional
        The core radius (default is 0.05).
    """
    p, rc = float(psi0), float(rc)

    def f(x):
        r = np.sqrt(x * x + rc * rc)
        return p * (r + rc * np.log(2 * rc / (r + rc)))
    return Potential(f, f, q_inf=0.0)


def nfw(k=1.0, psi0=1.0):
    """
    NFW: 0.5 psi0 [ln^2(v/2) + arccos^2(1/v)], v = k x  (k = xi0 / r_s).
    The convention is similar to GLoW refer Villarubia et. al.(put the proper reference here). 
    Parameters:
    -----------
    k : float, optional
        The scale factor (default is 1.0).
    psi0 : float, optional
        The characteristic surface density (default is 1.0).
    Returns:        
    --------
    Potential
        A Potential object representing the NFW lens potential.
    """
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
    """
    psi = kappa x^2 / 2 (external convergence). Requires kappa < 1.
    The solver fails for kappa > 1 with a warning. 
    The q_inf parameter exists in the potential to deal with such lenses.

    Parameters:
    -----------
    kappa : float
        The convergence parameter (must be less than 1).    
    Returns:
    --------
    Potential
        A Potential object representing the mass sheet lens potential.

    """
    k = float(kappa)
    return Potential(lambda x: 0.5 * k * x * x, lambda x: 0.5 * k * x * x, q_inf=k)


def power_law(psi0=1.0, p=1.0):
    """
    psi = psi0 x^p / p  (0 < p < 2): deflection psi0 x^(p-1); p = 1 is the SIS.
    Parameters:
    -----------
    psi0 : float, optional
        The amplitude (default is 1.0).
    p : float, optional
        The power-law index (default is 1.0, must be between 0 and 2).
    Returns:
    --------
    Potential
        A Potential object representing the power-law lens potential.
    """
    a, p = float(psi0), float(p)
    return Potential(lambda x: a * x**p / p, lambda x: a * np.asarray(x, float)**p / p, q_inf=0.0)


# Some helper functions for the solver (Gauss-Legendre, Gauss-Laguerre, Chebyshev points, barycentric interpolation)
# Much needed for the solver to work efficiently and accurately. These functions are cached for performance.
# Keep its as a private function to avoid polluting the namespace.
# TBD if its is to be documented and mentioned in the publication....
_GL, _LAG = {}, {}


def _gl(n):
    """
    Get Gauss-Legendre nodes and weights for n points.
    Parameters:
    -----------
    n : int
        The number of points.
    Returns:
    --------
    tuple
        A tuple containing the nodes and weights.
    """
    if n not in _GL:
        _GL[n] = roots_legendre(n)
    return _GL[n]


def _lag(n):
    """
    Get Gauss-Laguerre nodes and weights for n points.
    Parameters:
    -----------
    n : int
        The number of points.
    Returns:
    --------
    tuple
        A tuple containing the nodes and weights.
    """
    if n not in _LAG:
        _LAG[n] = roots_laguerre(n)
    return _LAG[n]


def _cheb_points(n):
    """
    Chebyshev points of the first kind in [-1, 1]. It is used for interpolation later.
    Faster than calling splines in some way.
    Parameters:
    -----------
    n : int
        The number of points.
    Returns:
    --------
    ndarray
        An array of Chebyshev points.
    """
    k = np.arange(n)
    return np.cos(np.pi * (k + 0.5) / n)


def _bary_matrix(xc, xe):
    """
    Barycentric interpolation matrix from Chebyshev points (first kind) xc to points xe.
    Parameters:
    -----------
    xc : ndarray
        Chebyshev points of the first kind.
    xe : ndarray
        Points to interpolate to.
    Returns:
    --------
    ndarray
        The barycentric interpolation matrix.
    """
    n = xc.size
    wts = (-1.0) ** np.arange(n) * np.sin(np.pi * (np.arange(n) + 0.5) / n)
    d = xe[:, None] - xc[None, :]
    exact = np.abs(d) <= 1e-15
    if exact.any():
        d[exact] = 1.0
    M = wts[None, :] / d
    M /= M.sum(axis=1, keepdims=True)
    if exact.any():
        rows = np.any(exact, axis=1)
        M[rows] = exact[rows].astype(float)
    return M


# Hankel expansion taken from here (https://dlmf.nist.gov/10.17). See 10.17.5 for more details.
# It coverges very fast so only few terms are needed.
# relative error < 1e-15 for z >= 25. Below that: J0 + i Y0 from scipy.
#The below factors are used in the functions _h_asym() and it is hardcoded because the scipy's default function call is 130ns per value
# _h_asym() function is called a million times so it is needed to speed things up....

_J0_TAYLOR = np.array([(-1.0) ** k / (4.0 ** k * math.factorial(k) ** 2) for k in range(24)])
_Z_ASYM = 25.0
_HA = [1.0]
for _k in range(1, 20):
    _HA.append(_HA[-1] * (-(2 * _k - 1) ** 2) / (8.0 * _k))
_HP = np.array([(-1) ** (k // 2) * _HA[k] for k in range(0, 20, 2)])          # P(t) = sum _HP[m] t^2m
_HQ = np.array([(-1) ** ((k - 1) // 2) * _HA[k] for k in range(1, 20, 2)])    # Q(t) = t sum _HQ[m] t^2m
_HC = np.sqrt(2 / np.pi) * np.exp(-0.25j * np.pi)


def _h_asym(z):
    """
    h(z) = H0^(1)(z) exp(-iz) for real z >= _Z_ASYM: asymptotic expansion, smooth, ~ sqrt(2/(pi z)) e^{-i pi/4}.
    A faster version...
    Parameters:
    -----------
    z : float or ndarray
        The input value(s) for which to compute h(z).
    Returns:
    --------
    complex or ndarray
        The computed value(s) of h(z).
    """
    t = 1.0 / z
    t2 = t * t
    P = _HP[-1] * t2 + _HP[-2]
    Q = _HQ[-1] * t2 + _HQ[-2]
    for cp, cq in zip(_HP[-3::-1], _HQ[-3::-1]):
        P *= t2
        P += cp
        Q *= t2
        Q += cq
    Q *= t
    s = np.sqrt(t)
    out = np.empty(z.shape, dtype=complex)
    out.real = s * (_HC.real * P - _HC.imag * Q)
    out.imag = s * (_HC.imag * P + _HC.real * Q)
    return out


def _h_env(z):
    """
    This function uses both the series expansion of the Bessel function used above and also the
    scipy functions, pickups the faster method which is accurate.
    h(z) = H0^(1)(z) exp(-iz) for real z > 0: smooth, ~ sqrt(2/(pi z)) e^{-i pi/4} for large z.
    Parameters:
    -----------
    z : float or ndarray
        The input value(s) for which to compute h(z).
    Returns:
    --------
    complex or ndarray
        The computed value(s) of h(z).
    """
    z = np.asarray(z, dtype=float)
    far = z >= _Z_ASYM
    if far.all():
        return _h_asym(z)
    out = np.empty(z.shape, dtype=complex)
    if far.any():
        out[far] = _h_asym(z[far])
    zn = z[~far]
    #Exact definition is used here and it is called from scipy
    out[~far] = (j0(zn) + 1j * y0(zn)) * np.exp(-1j * zn)
    return out


# ---- Bessel functions of complex argument (the rotated tail), without scipy's slow AMOS routines.
# Valid for Re z > 0 (the tail has 0 <= arg z < pi/4). |z| >= z_asy: Hankel asymptotic series
# (DLMF 10.17.5, error below the smallest term ~ e^{-2|z|}); |z| <= z_ser: ascending series of J0
# and Y0 (DLMF 10.8.2, ~(|z|/2)^{2k}/k!^2 of cancellation); in between: scipy.
# Default (8, 17): ~1e-14 relative (J0 scale); (11, 11) for looser settings: < 1e-9, no scipy calls.
_ZC_SER, _ZC_ASY = 9.0, 14.0
_AK = [1.0]
for _k in range(1, 40):
    _AK.append(_AK[-1] * (2 * _k - 1) ** 2 / (8.0 * _k))
_AK = np.array(_AK)                                                    # |a_k(0)|
_LOG_AK = np.log(_AK)
_EULER = 0.5772156649015329
_HARM = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, 40))])     # harmonic numbers H_k


def _hank_asym_c(z):
    """(H0^(1)(z) e^{-iz}, H0^(2)(z) e^{iz}) for complex |z| >= _ZC_ASY (asymptotic series)."""
    a = float(np.min(np.abs(z)))
    k = np.arange(1, _AK.size)
    small = np.nonzero(_LOG_AK[1:] - k * np.log(a) < np.log(1e-17))[0]
    K = int(min(_AK.size, (small[0] + 2) if small.size else _AK.size, max(4, int(2 * a))))
    c = _AK[:K] * (-1j) ** np.arange(K)                               # i^k a_k(0), a_k(0) = (-1)^k |a_k|
    t = 1.0 / z
    sp = np.full(z.shape, c[-1], dtype=complex)
    sm = np.full(z.shape, np.conj(c[-1]), dtype=complex)
    for ck in c[-2::-1]:
        sp *= t
        sp += ck
        sm *= t
        sm += np.conj(ck)
    pre = np.sqrt(2.0 / (np.pi * z))
    return pre * (np.exp(-0.25j * np.pi) * sp), pre * (np.exp(0.25j * np.pi) * sm)


def _j0y0_series_c(z, need_y=True):
    """J0(z), Y0(z) by their ascending series (complex z, |z| <= _ZC_SER); Y0 is None if not need_y."""
    q = -0.25 * z * z
    a2 = (0.5 * float(np.max(np.abs(z)))) ** 2
    term = np.ones(z.shape, dtype=complex)
    J = term.copy()
    S = np.zeros(z.shape, dtype=complex)
    mag = 1.0
    for k in range(1, 40):
        term *= q / (k * k)
        J += term
        if need_y:
            S += _HARM[k] * term
        mag *= a2 / (k * k)
        if mag < 1e-18:
            break
    if not need_y:
        return J, None
    return J, (2.0 / np.pi) * ((np.log(0.5 * z) + _EULER) * J - S)


def _hankel_e_c(z, z_ser=_ZC_SER, z_asy=_ZC_ASY):
    """scipy's (hankel1e(0, z), hankel2e(0, z)) for complex z with Re z > 0, mostly without scipy."""
    z = np.asarray(z, dtype=complex)
    a = np.abs(z)
    h1 = np.empty(z.shape, dtype=complex)
    h2 = np.empty(z.shape, dtype=complex)
    far, near = a >= z_asy, (a <= z_ser) & (a < z_asy)
    mid = ~(far | near)
    if far.any():
        h1[far], h2[far] = _hank_asym_c(z[far])
    if near.any():
        zn = z[near]
        J, Y = _j0y0_series_c(zn)
        e = np.exp(-1j * zn)
        h1[near] = (J + 1j * Y) * e
        h2[near] = (J - 1j * Y) / e
    if mid.any():
        h1[mid] = hankel1e(0, z[mid])
        h2[mid] = hankel2e(0, z[mid])
    return h1, h2


def _j0_c(z, z_ser=_ZC_SER, z_asy=_ZC_ASY):
    """J0(z) for complex z with Re z > 0 (scipy's jv(0, z) costs ~1 us per value)."""
    z = np.asarray(z, dtype=complex)
    a = np.abs(z)
    out = np.empty(z.shape, dtype=complex)
    far, near = a >= z_asy, (a <= z_ser) & (a < z_asy)
    mid = ~(far | near)
    if far.any():
        zf = z[far]
        h1, h2 = _hank_asym_c(zf)
        e = np.exp(1j * zf)
        out[far] = 0.5 * (h1 * e + h2 / e)
    if near.any():
        out[near] = _j0y0_series_c(z[near], need_y=False)[0]
    if mid.any():
        out[mid] = jv(0, z[mid])
    return out


# ---- error models that size the quadrature and the interpolation (Bernstein-ellipse bounds)
_RHO = 1.0 + np.geomspace(1e-3, 60.0, 64)


def _rho_sing(ts):
    """Bernstein-ellipse parameter of a singular point at t = -ts (ts >= 1) of the interval [-1, 1]."""
    ts = np.maximum(np.asarray(ts, dtype=float), 1.0)
    return ts + np.sqrt(ts * ts - 1.0)


def _gl_log_err(n, kappa, ts):
    """log of the Gauss-Legendre error bound for exp(i kappa t) g(t) on [-1, 1], g singular at t = -ts:
    min_rho (64/15) M rho^(-2n) / (rho^2 - 1), M = exp(kappa (rho - 1/rho) / 2), rho < rho_s^0.9."""
    kappa, ts, n = np.broadcast_arrays(np.asarray(kappa, float), np.asarray(ts, float), np.asarray(n, float))
    r = _RHO[None, :]
    v = (np.log(64.0 / 15.0) - np.log(r * r - 1.0) + kappa.reshape(-1, 1) * (r - 1.0 / r) / 2.0
         - 2.0 * n.reshape(-1, 1) * np.log(r))
    v = np.where(r < _rho_sing(ts).reshape(-1, 1) ** 0.9, v, np.inf)
    return v.min(axis=1).reshape(kappa.shape)


_GL_TAB, _GLC_TAB = {}, {}
_GLC_K = np.concatenate([[0.0], np.geomspace(1e-3, 1e3, 73)])          # kappa grid of the lookup table
_GLC_T = np.geomspace(1.0, 1e8, 81)                                     # ts grid


def _gl_table(n_max):
    """Gauss-Legendre points and weights for n = 1..n_max, padded into (n_max + 1, n_max) arrays."""
    tab = _GL_TAB.get(n_max)
    if tab is None:
        GX, GW = np.zeros((n_max + 1, n_max)), np.zeros((n_max + 1, n_max))
        for n in range(1, n_max + 1):
            GX[n, :n], GW[n, :n] = _gl(n)
        tab = _GL_TAB[n_max] = (GX, GW)
    return tab


def _gl_count_fast(kappa, ts, log_eps, n_min, n_max):
    """_gl_count from a lookup table (cell corner with the larger kappa and the smaller ts: never fewer
    points than the exact rule at the grid points)."""
    key = (round(log_eps, 9), n_min, n_max)
    tab = _GLC_TAB.get(key)
    if tab is None:
        tab = _GLC_TAB[key] = _gl_count(_GLC_K[:, None], _GLC_T[None, :], log_eps, n_min, n_max)
    i = np.minimum(np.searchsorted(_GLC_K, kappa, side="left"), _GLC_K.size - 1)
    j = np.clip(np.searchsorted(_GLC_T, ts, side="right") - 1, 0, _GLC_T.size - 1)
    n = tab[i, j]
    big = kappa > _GLC_K[-1]
    if np.any(big):
        n[big] = _gl_count(kappa[big], ts[big], log_eps, n_min, n_max)
    return n


def _gl_count(kappa, ts, log_eps, n_min, n_max):
    """Fewest Gauss-Legendre points whose error bound (see _gl_log_err) is below exp(log_eps)."""
    kappa, ts = np.broadcast_arrays(np.asarray(kappa, float), np.asarray(ts, float))
    r = _RHO[None, :]
    n = ((np.log(64.0 / 15.0) - np.log(r * r - 1.0) + kappa.reshape(-1, 1) * (r - 1.0 / r) / 2.0 - log_eps)
         / (2.0 * np.log(r)))
    n = np.where(r < _rho_sing(ts).reshape(-1, 1) ** 0.9, n, np.inf).min(axis=1)
    return np.clip(np.ceil(n), n_min, n_max).astype(int).reshape(kappa.shape)


def _cheb_count(Bh, ts, log_tol):
    """Chebyshev points to interpolate G on a panel to exp(log_tol): |G| <= e^{B |Im w|}, G singular at
    w = 0 (t = -ts). Coefficients decay as M rho^(-n) (Trefethen, ATAP Thm 8.1; the prefactor is dropped,
    the actual decay of F is faster), minimised over 1 < rho < rho_s."""
    Bh, ts = np.broadcast_arrays(np.asarray(Bh, float), np.asarray(ts, float))
    r = _RHO[None, :]
    n = (Bh.reshape(-1, 1) * (r - 1.0 / r) / 2.0 - log_tol) / np.log(r)
    n = np.where(r < _rho_sing(ts).reshape(-1, 1), n, np.inf).min(axis=1)
    return (np.ceil(n) + 1).reshape(Bh.shape)


# ---- threads: BLAS stays single-threaded inside the (small) matrix products of a spectrum
_TPC = {}


def _blas_single():
    if ThreadpoolController is None:
        return nullcontext()
    ctl = _TPC.get(os.getpid())
    if ctl is None:
        try:
            ctl = _TPC[os.getpid()] = ThreadpoolController()
        except Exception:
            return nullcontext()
    try:
        return ctl.limit(limits=1, user_api="blas")
    except Exception:
        return nullcontext()


_WARNED = set()


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(msg)


# ================================================================== parallel helpers
# Processes are started with 'fork', so workers inherit the lens (which may hold lambdas) and the
# finished solver without any pickling. A forked process must not start OpenMP threads if its parent
# already did (GNU libgomp then deadlocks), so FINUFFT runs with one thread in every forked process,
# and worker processes never start processes of their own.
_IMPORT_PID = os.getpid()
_WORKER = {}
_SPAWN_COST, _PAR_EFF, _MIN_GAIN = 0.02, 0.75, 0.8      # cost model (made to be pessimistic on purpose)


def available_cores():
    """
    Calculate the number of CPU cores this process may use.
    It is used for the optimization purpose. The default use of the all cores creates an overhead and sometimes slows done the 
    amplification factor caculation..
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _forked():
    return os.getpid() != _IMPORT_PID


def _can_fork():
    return "fork" in mp.get_all_start_methods() and not mp.current_process().daemon


@contextmanager
def _quiet_fork():
    with warnings.catch_warnings():         # Python >= 3.12 warns about fork() in threaded processes
        warnings.filterwarnings("ignore", message=r".*use of fork\(\) may lead to deadlocks.*")
        yield


def _init_worker(ref):
    signal.signal(signal.SIGINT, signal.SIG_IGN)          # Ctrl-C is handled by the parent
    try:                                                  # one BLAS / OpenMP thread per worker
        from threadpoolctl import threadpool_limits
        _WORKER["limits"] = threadpool_limits(1)
    except Exception:
        pass
    s = ref()
    s._pool = s._pool_fin = s._pool_pid = None            # the parent's pool is not ours
    s.n_workers, s.threads = 1, 1
    _WORKER["solver"] = s


def _eval_rows(task):
    """Worker: rows of Solver1D.evaluate. by_y: one spectrum per source position; else one profile per w."""
    s = _WORKER["solver"]
    by_y, ws, ys = task
    if by_y:
        return np.array([s.spectrum(ws, y) for y in ys]).reshape(len(ys), ws.size)
    return np.array([s.profile(w, ys) for w in ws]).reshape(ws.size, len(ys)).T


def _shutdown_pool(ex, pid):
    if os.getpid() == pid:
        ex.shutdown(wait=True, cancel_futures=True)


# ================================================================== solver
class _LazyBands:
    """The frequency bands of a solver, each built on first use: a sparse set of frequencies builds only
    the bands it touches. Indexing and iteration work as for a list."""

    def __init__(self, solver, lohi):
        self._s, self.lohi, self._b = solver, list(lohi), [None] * len(lohi)

    def __len__(self):
        return len(self._b)

    def __getitem__(self, i):
        b = self._b[i]
        if b is None:
            b = self._b[i] = self._s._make_band(*self.lohi[i])
        return b

    def __iter__(self):
        return (self[i] for i in range(len(self)))


class Solver1D:
    """
    Everything that does not depend on (w, y), for all wmin <= w <= wmax and all source positions
    

    Evaluation (all reuse the same nodes; psi is never evaluated again for the head):
      s(w, y=None)            F at one frequency, one source position (default: y0)
      s.spectrum(ws, y=None)  many frequencies, one source position   -> batched 1D non-uniform FFT
      s.profile(w, ys)        one frequency, many source positions    -> direct sums
      s.evaluate(ws, ys)      table (len(ys), len(ws))

    Accuracy/tuning  parameters (defaults give ~1e-12):
    cycles, ngl (phase cycles / maximum Gauss-Legendre nodes per head piece), nlag (Laguerre nodes on the tail), n_cheb (Chebyshev frequencies per band),
    band_ratio (width of the frequency bands), xb_min (move the split outward = convergence test),
    nufft_eps (FINUFFT tolerance), z_small (w y x below which J0 is used by its Taylor series),
    adaptive_gl (True: pieces with little phase get fewer nodes, sized by an error bound calibrated
    on the full pieces with ngl nodes; False: ngl nodes on every piece, as before).
    PE_SETTINGS: lighter settings for parameter estimation (accuracy ~1e-8).

    How a band is summed (speed): the head nodes near u = 0, where the phase w A_j changes by less than
    a few radians across the band and w y x_j is small, are summed once into a polynomial in w
    (Taylor series of exp(i (w - w_mid) A_j) and of J0; computed on the first call that needs the
    band); only the remaining nodes are summed per frequency (directly, or by NUFFT for many
    frequencies). Bands are built when first needed, so a few frequencies build only their bands.

    Interpolation (for many frequencies)
      interpolate  True (default): F is computed on Chebyshev panels sized by the lens time delays
                   (number of points per panel from an error bound) and interpolated to the requested
                   frequencies; used when it needs fewer exact evaluations than requested frequencies.
                   False: every frequency is computed. 'auto': interpolate only for >= ~4000 frequencies.
                   Can also be passed per call: s.spectrum(ws, y, interpolate=True).
      interp_tol   the interpolant is checked against exact values at four frequencies per panel;
                   a panel whose relative error exceeds interp_tol/3 is split (max 3 times); if
                   interp_tol is still not reached, the exact path is used. None (default):
                   max(1e-10, 100 nufft_eps), i.e. above the accuracy of the exact values (1e-7 with
                   PE_SETTINGS).

    Parallelism
      n_workers   maximum number of processes for evaluate() (1: none; None: all cores); used only
                  when timing of the first rows predicts a clear gain. close() stops the pool.
      threads     FINUFFT threads per call (None = 1: the transforms are small and one thread is
                  fastest; always 1 in forked or daemonic processes, e.g. bilby's npool workers).

    Parameters
    ----------
    lens : Potential or callable
        Potential object or a callable ``psi(x)`` representing the lensing potential.
        If callable, should accept a single array argument and return potential values.
    y : float, optional
        Source position in the lens plane (default: 0.0).
    wmin : float, optional
        Minimum frequency of the integration range (default: 1e-3).
    wmax : float, optional
        Maximum frequency of the integration range (default: 1e3).
    y_radius : float, optional
        Half-width of the source position range for parameter variation studies
        (default: 0.0).
    cycles : int, optional
        Maximum number of phase cycles per integration head piece. Affects the
        granularity of the integration scheme (default: 8).
    ngl : int, optional
        Number of Gauss-Legendre quadrature nodes per full head piece (``cycles`` phase
        cycles); with ``adaptive_gl`` lighter pieces get fewer. Higher values
        increase accuracy but raise computational cost (default: 32).
    nlag : int, optional
        Number of Gauss-Laguerre quadrature nodes on the tail portion of the
        integration domain (default: 30).
    n_cheb : int, optional
        Number of Chebyshev polynomial frequencies per frequency band (default: 18;
        14 gives the same accuracy for band_ratio=4).
    band_ratio : float, optional
        Width ratio defining the frequency bands. Controls the logarithmic spacing
        of frequency intervals (default: 8.0; fewer bands than 4.0 with n_cheb=14,
        same accuracy, ~20% faster).
    xb_min : float, optional
        Parameter to shift the split point outward for convergence testing
        and validation. Used in adaptive refinement (default: 0.0).
    nufft_eps : float, optional
        Tolerance parameter for the FINUFFT (Fast Implicit Nonuniform Fast
        Fourier Transform) algorithm. Lower values increase accuracy but raise
        computation time (default: 1e-14).
    z_small : float, optional
        Threshold below which the Bessel function J0 is approximated using
        its Taylor series expansion. Applies to ``w``, ``y``, and ``x`` values
        independently (default: 1.0).
    n_workers : int, optional
        Number of CPU cores for parallel processing. If ``None``, uses all
        available cores (default: None).
    interpolate : bool or 'auto', optional
        If ``True``, F is computed on Chebyshev panels in w and interpolated to the
        requested frequencies when that needs fewer exact evaluations (default: True).
    interp_tol : float, optional
        Relative accuracy required of the interpolation in w (default: None, i.e.
        max(1e-10, 100 nufft_eps)).
    adaptive_gl : bool, optional
        Fewer Gauss-Legendre nodes on head pieces with little phase (default: True).
 
    Returns
    -------
    result : ndarray
        Array containing the computed lensing values at the specified parameters.
        Shape and dtype depend on input dimensionality.
 
    Raises
    ------
    ValueError
        If ``wmin >= wmax`` or if ``y_radius < 0``.
    TypeError
        If ``lens`` is neither a Potential object nor callable.
    RuntimeError
        If FINUFFT initialization fails or convergence cannot be achieved.
 
    Notes
    -----
    This function uses a sophisticated adaptive quadrature scheme that combines:
 
    - **Gauss-Legendre integration** for the main oscillatory regime with
      automatic splitting based on phase cycles
    - **Gauss-Laguerre integration** for tail behavior capturing asymptotic
      contributions with exponential decay
    - **Chebyshev polynomial approximations** for frequency interpolation
      across logarithmic bands
 
    The FINUFFT tolerance (``nufft_eps``) directly affects both accuracy and
    computation time. Recommended values:
 
    - ``1e-14`` for high-precision calculations
    - ``1e-10`` for moderate precision with faster computation
    - ``1e-6`` for rapid prototyping
 
    Convergence with respect to ``ngl`` and ``nlag`` should be checked for
    new applications by varying these parameters.
 

    """

    def __init__(self, lens, y=0.0, wmin=1e-3, wmax=1e3, y_radius=0.0, cycles=8, ngl=32, nlag=30, n_cheb=18,
                 band_ratio=8.0, xb_min=0.0, nufft_eps=1e-14, z_small=1.0, n_workers=1, threads=None,
                 interpolate=True, interp_tol=None, adaptive_gl=True):
        """
        Initialize the 1D lensing solver.

        Parameters:
        -----------
        lens :        Potential object (or a callable psi(x) for the lensing potential)
        y           : source position (default: 0.0)
        wmin, wmax  : frequency range (default: 1e-3, 1e3)
        y_radius    : half-width of the source position range (default: 0.0)
        cycles      : max phase cycles per head piece (default: 8)
        ngl         : Gauss-Legendre nodes per full head piece (default: 32)
        nlag        : Gauss-Laguerre nodes on the tail (default: 30)
        n_cheb      : Chebyshev frequencies per band (default: 18)
        band_ratio  : width of the frequency bands (default: 8.0)
        xb_min      : move the split outward = convergence test (default: 0.0)
        nufft_eps   : FINUFFT tolerance (default: 1e-14)
        z_small     : w y x below which J0 is used by its Taylor series (default: 1.0)
        """

        if not isinstance(lens, Potential):
            lens = Potential(lens)
        if not lens._checked:
            lens.check_analytic()

        
        self.lens = lens

        self.y0, self.Yr = abs(float(y)), float(y_radius)
        self.y_lo, self.y_hi = max(0.0, self.y0 - self.Yr), self.y0 + self.Yr

        self.wmin, self.wmax = float(wmin), float(wmax)

        if not (0 < self.wmin <= self.wmax):
            raise ValueError("need 0 < wmin <= wmax")
        
        self.cycles = int(cycles)
        self.ngl = int(ngl)
        self.nlag = int(nlag)
        self.n_cheb = int(n_cheb)
        self.nufft_eps = float(nufft_eps)
        self.z_small =  float(z_small)
        self.n_workers = available_cores() if (n_workers is None or n_workers <= 0) else int(n_workers)

        self.threads = None if threads is None else max(1, int(threads))

        if interpolate not in (False, True, "auto"):
            raise ValueError("interpolate must be False, True or 'auto'")
        
        self.interpolate = interpolate
        self.interp_tol = max(1e-10, 100 * self.nufft_eps) if interp_tol is None else float(interp_tol)
        self.adaptive_gl = bool(adaptive_gl)
        self._tdcache = {}
        self._tdgrid = None
        self._pcache = {}
        self._pool = self._pool_fin = self._pool_pid = None

        self.GX, self.GW = _gl(self.ngl)
        self.LX, self.LW = _lag(self.nlag)
        # error level of a full head piece (pi*cycles radians half-phase, ngl nodes): lighter pieces get
        # the fewest nodes that reach the same bound (not below 1e-16: rounding)
        self._gl_log_eps = max(float(_gl_log_err(self.ngl, np.pi * self.cycles, 1e12)), np.log(1e-16))
        self._gl_nmin = min(self.ngl, 4)
        # truncation of the Taylor series (cluster, J0): far below the accuracy of the settings
        self._eps_t = float(np.clip(1e-3 * self.nufft_eps, 1e-17, 1e-11))
        self._zc = (11.0, 11.0) if self._eps_t >= 1e-12 else (_ZC_SER, _ZC_ASY)   # complex Bessel ranges

        # ---- split: (y_hi + psi'(x_b))/x_b <= thr, thr = 1/2 (or (1 + kappa)/2 with a mass sheet)
        q = lens.q_inf
        if q is None:
            q = max(float(lens.dpsi(R) / R) for R in (1e3, 1e4))
        if q >= 1:
            raise ValueError(f"lim psi'/x = {q:.3f} >= 1: the tail cannot be rotated upward.")

        #Threshold for splitting the integration into head and tail
        thr = max(0.5, 0.5 * (1 + q))
        #initial guess
        xb = max(3.0, 2 * self.y_hi, float(xb_min))
        #Loop to find the optimum point
        for _ in range(400):
            if (self.y_hi + lens.dpsi(xb)) / xb <= thr:
                break
            xb *= 1.1
        else:
            raise RuntimeError("no split point found; does psi grow faster than x^2/2?")
        
        self.q_inf = q 
        self.xb0 = xb
        self.ub0 = xb * xb / 2

        self.ub_max = max(self.ub0, 20 / self.wmin)

        self.u_floor = 1e-15 / max(self.wmax, 1.0)            # [0, u_floor] contributes < 1e-15

        # ---- phase per unit w, V1(u) (worst case over the y range), places the head pieces
        # Its a Phase meter here
        # linear grid on [0, u_b0] (resolves the oscillation), geometric grid out to ub_max (beyond u_b0
        # A(u) is monotonic, so ~60 points per e-fold suffice there and near u = 0)
        n_lin = int(max(3000, 3 * self.wmax * self.ub0 / (2 * np.pi)))
        n_geo = int(min(20000, max(2000, 60 * np.log(self.ub_max / self.u_floor))))
        ug = np.unique(np.concatenate([np.linspace(self.u_floor, self.ub0, n_lin),
                                       np.geomspace(self.u_floor, self.ub_max, n_geo)]))
        x = np.sqrt(2 * ug)
        with np.errstate(all="ignore"):
            d = np.abs(np.diff(ug - lens.real(x)))
        d[~np.isfinite(d)] = 0.0
        self.ug, self.V1 = ug, np.concatenate([[0.0], np.cumsum(d)]) + self.y_hi * (x - x[0])
        self.rate = np.diff(self.V1) / np.diff(self.ug)      # phase per unit w per unit u

        # ---- frequency bands [w_hi / band_ratio, w_hi], from wmax down to wmin
        # (each band is built when it is first needed, see _LazyBands)
        lohi, hi = [], self.wmax
        while True:
            lo = max(self.wmin, hi / band_ratio)
            lohi.append((lo, hi))
            if lo <= self.wmin * (1 + 1e-12):
                break
            hi = lo
        self.bands = _LazyBands(self, lohi)
        self._tcache = {}
        self._hcache = {}

    @property
    def info(self):
        """Split point, q_inf, y range and per band (lo, hi, u_b, nodes); builds all bands."""
        return dict(x_b=self.xb0, q_inf=self.q_inf, y_range=(self.y_lo, self.y_hi),
                    bands=[(b["lo"], b["hi"], b["ub"], b["xa"].size) for b in self.bands],
                    n_nodes=int(sum(b["xa"].size for b in self.bands)),
                    n_nodes_per_w=int(sum(b["x"].size for b in self.bands)))


    def _make_band(self, lo, hi):
        """
        All weights for the head and tail integration are chosen here in this function.
        Make the nodes and weights for one frequency band [lo, hi].

        Parameters:
        -----------
        lo : lower frequency bound
        hi : upper frequency bound
        Returns:
        --------
        dict containing the following keys:
        - lo, hi : frequency bounds
        - ub : upper bound for the head (u_b = max(u_b0, 20 / lo))
        - xb : corresponding x_b = sqrt(2 * u_b)
        - al : decay rate for the tail
        - psib : psi(x_b)
        - x : array of x_j nodes for the head
        - W : array of weights W_j for the head
        - A : array of phases A_j = u_j - psi(x_j) for the head
        - cx : Chebyshev points for interpolation in ln w
        - wc : Chebyshev frequencies for interpolation in ln w
        - xt : array of x_j nodes for the tail
        - dx : array of differences xt - xb for the tail
        - E : array of weights for the tail
        - cl : the near-zero cluster (nodes summed once into a polynomial in w), or None
        - x, W, A then hold only the nodes outside the cluster (summed per frequency)
        """
        lens = self.lens
        ub = max(self.ub0, 20 / lo)                          # small-w rule
        xb = np.sqrt(2 * ub)
        # head nodes: a piece every `cycles` phase cycles at w = hi, geometric grading towards 0
        V_ub = np.interp(ub, self.ug, self.V1)
        cuts = np.interp(np.arange(0, V_ub, 2 * np.pi * self.cycles / hi), self.V1, self.ug)
        lg = np.log(self.u_floor)
        geo = np.exp(np.linspace(lg, np.log(ub), int(np.ceil(np.log(ub) - lg)) + 1))
        e = np.sort(np.concatenate([[0.0], geo, cuts[cuts < ub], np.linspace(0, ub, 5), [ub]]))
        e = e[np.concatenate([[True], np.diff(e) > 0])]
        # the cuts bound the phase change per piece; also bound (largest phase rate) x (width), which
        # matters where the rate varies strongly inside a piece (stationary points: chirps)
        m = max(1, int(np.searchsorted(self.ug, ub, side="right")))
        rate = self.rate[:m]                                 # phase per unit w per unit u, up to ub
        i0 = np.clip(np.searchsorted(self.ug, e[:-1], side="right") - 1, 0, rate.size - 1)
        peak = np.maximum.reduceat(rate, i0)                 # largest rate in each piece

        n_sub = np.maximum(1, np.ceil(peak * np.diff(e) * hi / (1.25 * 2 * np.pi * self.cycles))).astype(int)
        if np.any(n_sub > 1):
            e = np.unique(np.concatenate([e] + [np.linspace(e[k], e[k + 1], n + 1) for k, n in enumerate(n_sub) if n > 1]))

        #Mid and the half points in each head piece
        mid = (e[1:] + e[:-1]) / 2
        half = (e[1:] - e[:-1]) / 2

        # Gauss-Legendre points per piece: ngl on the full pieces; with adaptive_gl, pieces with less
        # phase (half-phase kappa at w = hi) or far from the singular point u = 0 get the fewest points
        # that reach the error bound of a full piece (see _gl_count)
        if self.adaptive_gl:
            i0 = np.clip(np.searchsorted(self.ug, e[:-1], side="right") - 1, 0, rate.size - 1)
            peak = np.maximum.reduceat(rate, i0)
            dV = np.diff(np.interp(e, self.ug, self.V1))
            kappa = 0.5 * hi * np.maximum(peak * np.diff(e), dV)
            with np.errstate(divide="ignore", invalid="ignore"):
                ts = np.where(e[:-1] > 0, (e[1:] + e[:-1]) / (e[1:] - e[:-1]), 1.0)
            n_pc = _gl_count_fast(kappa, ts, self._gl_log_eps, self._gl_nmin, self.ngl)
        else:
            n_pc = np.full(mid.size, self.ngl)
        # all pieces at once: piece p gets the n_pc[p]-point rule (padded table of rules)
        GX, GW = _gl_table(self.ngl)
        p = np.repeat(np.arange(mid.size), n_pc)
        j = np.arange(p.size) - np.repeat(np.cumsum(n_pc) - n_pc, n_pc)
        u = mid[p] + half[p] * GX[n_pc[p], j]
        W = half[p] * GW[n_pc[p], j]

        x = np.sqrt(2 * u)
        with np.errstate(all="ignore"):
            A = u - lens.real(x)
        nc = self._cluster_size(lo, hi, x, A)

        # tail: decay rate for the largest y, psi at the complex Laguerre points of every
        # Chebyshev frequency of the band (y-independent)

        #The decay term alpha
        al = 1 - (self.y_hi + float(lens.dpsi(xb))) / xb
        if al <= 0:
            raise RuntimeError("tail does not decay; increase xb_min")
        
        #The value of the potential at the split point
        psib = float(np.real(np.asarray(lens(np.array([xb + 0j])))).ravel()[0])
        if hi / lo < 1 + 1e-12:
            cx, zc = np.zeros(1), np.array([np.log(hi)])
        else:
            cx = _cheb_points(self.n_cheb) #Cheb points for interpolation in ln w
            zc = 0.5 * (np.log(lo) + np.log(hi)) + 0.5 * (np.log(hi) - np.log(lo)) * cx
        wc = np.exp(zc)
        
        t = self.LX[None, :] / (al * wc[:, None])
        xt = np.sqrt(2 * (ub + 1j * t))
        # xa, Wa, Aa: all head nodes; x, W, A: the per-frequency ones (after the cluster of nc nodes)
        return dict(lo=lo, hi=hi, ub=ub, xb=xb, al=al, psib=psib, xa=x, Wa=W, Aa=A, nc=nc,
                    x=x[nc:], W=W[nc:], A=A[nc:], cx=cx, wc=wc, xt=xt, dx=xt - xb, E=None,
                    zlo=np.log(lo), zhi=np.log(hi), cl=None)

    def _tail_E(self, b):
        """Tail weights at the Laguerre points of every Chebyshev frequency of band b (y-independent;
        computed on first use)."""
        if b["E"] is None:
            al, wc = b["al"], b["wc"][:, None]
            b["E"] = self.LW[None, :] * np.exp(self.LX[None, :] * (1 - 1 / al) - 1j * wc * (self.lens(b["xt"]) - b["psib"]))
        return b["E"]

    # ------------------------------------------------------------------ near-zero cluster
    _PHI_CL = 3.0        # largest |(w - w_mid) (A_j - A_0)| in the cluster (Taylor terms ~ e^PHI, no cancellation)
    _Z_CL = 2.0          # largest w y x_j in the cluster (J0 Taylor series)
    _MIN_CL = 16         # fewer nodes than this: no cluster

    _M_CL = 8            # fewer frequencies than this in a band (and no cluster yet): plain direct sums

    def _cluster_size(self, lo, hi, x, A):
        """Number of head nodes (sorted in u, from u = 0) in the near-zero cluster of band [lo, hi]: up to
        the first node where the phase spread (hi - lo)/2 (max A - min A) exceeds 2 _PHI_CL or
        hi y_hi x_j exceeds _Z_CL (0 if fewer than _MIN_CL)."""
        n = x.size
        if n < self._MIN_CL:
            return 0
        hw = 0.5 * (hi - lo)
        with np.errstate(invalid="ignore"):
            Amin = np.minimum.accumulate(A)
            Amax = np.maximum.accumulate(A)
            ok = np.isfinite(A) & (hw * (Amax - Amin) <= 2 * self._PHI_CL) & (hi * self.y_hi * x <= self._Z_CL)
        nc = int(np.argmin(ok)) if not ok.all() else n
        return nc if nc >= self._MIN_CL else 0

    def _cluster(self, b):
        """
        The cluster of band b (computed on first use; None if the band has none). For its nc nodes,
        over the band,
            sum_cl W_j J0(w y x_j) e^{i w A_j}
              = e^{i w A_0} sum_{k,l} Q_kl tau^k (w/hi)^{2l} a_l (y/y_hi)^{2l},   tau = (w - w_mid)/h_w,
            Q_kl = sum_cl W_j e^{i w_mid d_j} (i h_w d_j)^k / k! (hi y_hi x_j)^{2l},   d_j = A_j - A_0,
        (Taylor series of exp(i (w - w_mid) d_j) and of J0, a_l = (-1)^l / (4^l l!^2)). Q does not depend
        on w or y: these nodes cost nothing per frequency.
        """
        nc = b["nc"]
        if nc == 0:
            return None
        if b["cl"] is not None:
            return b["cl"]
        lo, hi = b["lo"], b["hi"]
        hw, wm = 0.5 * (hi - lo), 0.5 * (hi + lo)
        yr = self.y_hi
        xc, Wc, Ac = b["xa"][:nc], b["Wa"][:nc], b["Aa"][:nc]
        A0 = 0.5 * (Ac.max() + Ac.min())
        d = Ac - A0
        phi = hw * float(np.max(np.abs(d)))
        K, mag = 1, 1.0                                      # Taylor terms of exp(i phi tau): phi^K/K! < 1e-17
        while K < 60 and mag * phi ** K / math.factorial(K) > self._eps_t:
            K += 1
        if hw == 0:
            K = 1
        # Q_kl = i^k sum_j [(h_w d_j)^k / k!] [W_j e^{i w_mid d_j} (hi y_hi x_j)^{2l}]
        Pc = Wc * np.exp(1j * wm * d)
        if K > 1:
            R = np.empty((K, nc))
            R[0] = 1.0
            R[1:] = (hw * d)[None, :] / np.arange(1, K)[:, None]
            np.cumprod(R, axis=0, out=R)
        else:
            R = np.ones((1, nc))
        if yr > 0:
            X = (hi * yr * xc) ** 2
            zmax2 = float(X.max())
            L = 1
            while L < _J0_TAYLOR.size and abs(_J0_TAYLOR[L]) * zmax2 ** L > self._eps_t:
                L += 1
            Z = np.empty((nc, L))
            Z[:, 0] = 1.0
            if L > 1:
                Z[:, 1:] = X[:, None]
                np.cumprod(Z[:, 1:], axis=1, out=Z[:, 1:])
            Q = (R @ (Pc.real[:, None] * Z) + 1j * (R @ (Pc.imag[:, None] * Z))) * ((1j) ** np.arange(K))[:, None]
        else:
            L = 1
            Q = (R @ Pc)[:, None] * ((1j) ** np.arange(K))[:, None]
        b["cl"] = dict(A0=A0, wm=wm, hw=hw, Q=Q, K=K, L=L, y_ref=yr, n=nc, hi=hi)
        return b["cl"]

    def _cluster_M(self, cl, y):
        """Q scaled for the source position y: M_kl = Q_kl a_l (y / y_hi)^{2l}."""
        L = cl["L"]
        if L == 1:
            return cl["Q"]
        r2 = (y / cl["y_ref"]) ** 2
        return cl["Q"] * (_J0_TAYLOR[:L] * r2 ** np.arange(L))[None, :]

    def _cluster_sum(self, cl, y, wb):
        """Cluster part of the head sum at the frequencies wb (one y)."""
        if cl is None:
            return 0.0
        M = self._cluster_M(cl, y)
        K, L = cl["K"], cl["L"]
        tau = (wb - cl["wm"]) / cl["hw"] if cl["hw"] > 0 else np.zeros_like(wb)
        Vt = np.empty((wb.size, K))
        Vt[:, 0] = 1.0
        if K > 1:
            Vt[:, 1:] = tau[:, None]
            np.cumprod(Vt[:, 1:], axis=1, out=Vt[:, 1:])
        R = Vt @ M                                           # (m, L)
        if L > 1:
            s2 = (wb / cl["hi"]) ** 2
            acc = R[:, L - 1]
            for l in range(L - 2, -1, -1):
                acc = acc * s2 + R[:, l]
        else:
            acc = R[:, 0]
        return np.exp(1j * wb * cl["A0"]) * acc

    def _cluster_profile(self, cl, w, ys):
        """Cluster part of the head sum at one frequency w for many source positions ys."""
        if cl is None:
            return 0.0
        K, L = cl["K"], cl["L"]
        tau = (w - cl["wm"]) / cl["hw"] if cl["hw"] > 0 else 0.0
        c = (tau ** np.arange(K)) @ cl["Q"]                  # (L,)
        if L == 1:
            return np.exp(1j * w * cl["A0"]) * c[0] * np.ones(ys.shape)
        c = c * _J0_TAYLOR[:L] * ((w / cl["hi"]) ** 2) ** np.arange(L)
        r2 = (ys / cl["y_ref"]) ** 2
        acc = np.full(ys.shape, c[L - 1], dtype=complex)
        for l in range(L - 2, -1, -1):
            acc = acc * r2 + c[l]
        return np.exp(1j * w * cl["A0"]) * acc

    def _bary(self, b, wb):
        """
        Barycentric interpolation matrix from Chebyshev points b["cx"] to frequencies wb.

        Parameters:
        -----------
        b : dict containing the band information (from _make_band)
        wb : array of frequencies to interpolate to
        Returns:
        --------
        Interpolation matrix of shape (len(wb), len(b["cx"])) that maps values at the Chebyshev points to values at the frequencies wb.     
        """
        if b["cx"].size == 1:
            return np.ones((wb.size, 1))
        return _bary_matrix(b["cx"], (2 * np.log(wb) - (b["zlo"] + b["zhi"])) / (b["zhi"] - b["zlo"]))

    # threads and processes
    def _nthreads(self):
        """
        Determine the number of threads to use for FINUFFT calls.
        """
        # The transforms here are small (thousands of nodes): one thread is faster and steadier than
        # several, so threads=None means 1. A forked process must use 1 (OpenMP deadlock otherwise).
        # (Python threads over the frequency bands were tried: slower, the sums are many small numpy
        # operations and the threads mostly wait for the GIL.)
        if self.threads is None:
            return 1
        if self.threads > 1 and (_forked() or mp.current_process().daemon):
            _warn_once("forkthreads", "threads > 1 is ignored in forked/pool processes (OpenMP would deadlock).")
            return 1
        return self.threads

    def _fopts(self):
        return {"nthreads": self._nthreads()}

    def _pmax(self, n_tasks):
        n = min(self.n_workers, available_cores(), int(n_tasks))
        if n <= 1:
            return 1
        if not _can_fork():
            if not mp.current_process().daemon:
                _warn_once("nofork", "n_workers > 1 needs the 'fork' start method (Linux/macOS); running serially.")
            return 1
        return n

    def _get_pool(self):
        if self._pool is not None and self._pool_pid != os.getpid():     # inherited from a parent process
            self._pool = self._pool_fin = None
        if self._pool is None:
            n = max(1, min(self.n_workers, available_cores()))
            self._pool = ProcessPoolExecutor(n, mp_context=mp.get_context("fork"), initializer=_init_worker,
                                             initargs=(weakref.ref(self),))
            self._pool_pid = os.getpid()
            self._pool_fin = weakref.finalize(self, _shutdown_pool, self._pool, self._pool_pid)
        return self._pool

    def close(self):
        """Stop the worker processes of evaluate() (they are restarted automatically if needed)."""
        if self._pool_fin is not None:
            self._pool_fin()
        self._pool = self._pool_fin = self._pool_pid = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getstate__(self):                     
        d = self.__dict__.copy()
        d["_pool"] = d["_pool_fin"] = d["_pool_pid"] = None
        return d

    # checks for w and y 
    def _check_y(self, y):
        y = np.abs(np.asarray(y, dtype=float))
        tol = 1e-9 * max(self.y_hi, 1.0) + 1e-12
        if np.any(y < self.y_lo - tol) or np.any(y > self.y_hi + tol):
            raise ValueError(f"|y| must lie in [{self.y_lo:g}, {self.y_hi:g}] for this solver; "
                             "build it with a larger y_radius")

    def _check_w(self, wp):
        if np.any(wp < self.wmin * (1 - 1e-12)) or np.any(wp > self.wmax * (1 + 1e-12)):
            raise ValueError(f"frequencies must lie in [{self.wmin:g}, {self.wmax:g}] for this solver")

    def _band_index(self, wp):
        idx = np.full(wp.size, -1)
        for i, (lo, hi) in enumerate(self.bands.lohi):
            sel = (idx < 0) & (wp >= lo * (1 - 1e-12)) & (wp <= hi * (1 + 1e-12))
            idx[sel] = i
        return idx

    # ------------------------------------------------------------------ sums
    def _nufft(self, src, C, wb):

        """
        $sum_j C[k, j] exp(i w src_j)$ for every row k of C and every w in wb -> (rows, len(wb)).
        
        Parameters:
        -----------
        src : array of source positions (x_j)
        C : 2D array of coefficients (C[k, j])
        wb : array of frequencies (w)

        Returns:
        --------
        2D array of shape (C.shape[0], wb.size) containing the sums for each row of C and each frequency in wb.
        """
        C = np.atleast_2d(C)
        if finufft is not None and wb.size >= 4 and src.size * wb.size >= 40_000:
            R = finufft.nufft1d3(src, np.ascontiguousarray(C, dtype=complex), wb, isign=1, eps=self.nufft_eps,
                                 **self._fopts())
            return np.reshape(R, (C.shape[0], wb.size))
        if finufft is None and wb.size * src.size > 2_000_000:
            _warn_once("nufft", "finufft is not installed: frequency sums are done directly (slower).")
        out = np.empty((C.shape[0], wb.size), dtype=complex)
        step = max(1, 2_000_000 // max(src.size, 1))
        for s in range(0, wb.size, step):
            out[:, s:s + step] = C @ np.exp(1j * np.outer(src, wb[s:s + step]))
        return out

    def _h_cached(self, bi, y, big):
        """W_j h(wc y x_j) for the band's non-small per-frequency nodes at its Chebyshev frequencies wc
        (cached per band for the last source position)."""
        hit = self._hcache.get(bi)
        if hit is not None and hit[0] == y:
            return hit[1]
        b = self.bands[bi]
        H = _h_env(b["wc"][:, None] * (y * b["x"][big])[None, :]) * b["W"][big][None, :]
        self._hcache[bi] = (y, H)
        return H

    def _head_fast(self, bi, y, wb):
        """sum_j W_j J0(w y x_j) exp(i w A_j) over the band's per-frequency nodes, many w
        (Taylor + Hankel/Chebyshev + NUFFT)."""
        b = self.bands[bi]
        x, W, A, hi = b["x"], b["W"], b["A"], b["hi"]
        out = np.zeros(wb.size, dtype=complex)
        if x.size == 0:
            return out
        s = hi * y * x                                       # largest w y x of each node in the band
        small = s <= self.z_small
        if small.any():                                      # J0 = sum_k a_k (w/hi)^2k s^2k
            ss2 = s[small] ** 2
            smax2 = ss2.max()
            K = 1
            while K < _J0_TAYLOR.size and abs(_J0_TAYLOR[K]) * smax2 ** K > self._eps_t:
                K += 1
            C = _J0_TAYLOR[:K, None] * ss2[None, :] ** np.arange(K)[:, None] * W[small][None, :]
            R = self._nufft(A[small], C, wb)
            q = (wb / hi) ** 2
            acc = R[K - 1]
            for k in range(K - 2, -1, -1):
                acc = acc * q + R[k]
            out += acc
        if not small.all():                                  # J0 = [h e^{iz} + conj(h) e^{-iz}] / 2
            big = ~small
            xl, Al = x[big], A[big]
            H = self._h_cached(bi, y, big)
            R = self._nufft(np.concatenate([Al + y * xl, Al - y * xl]), np.concatenate([H, H.conj()], axis=1), wb)
            out += 0.5 * np.einsum("mc,cm->m", self._bary(b, wb), R)
        return out

    def _head_dsum(self, b, y, wb, all_nodes=False):
        """The same sum by direct summation (few w): exp(i w A_j) by cos/sin; J0 by its Taylor series in
        (w/hi)^2 for the nodes with hi y x_j <= z_small (no Bessel calls), by scipy's j0 for the others.
        all_nodes: over all head nodes of the band (no cluster) instead of the per-frequency ones."""
        x, W, A, hi = (b["xa"], b["Wa"], b["Aa"], b["hi"]) if all_nodes else (b["x"], b["W"], b["A"], b["hi"])
        out = np.zeros(wb.size, dtype=complex)
        if x.size == 0 or wb.size == 0:
            return out
        s = hi * y * x
        small = s <= self.z_small
        big = ~small
        if small.any():
            ss2 = s[small] ** 2
            smax2 = ss2.max()
            L = 1
            while L < _J0_TAYLOR.size and abs(_J0_TAYLOR[L]) * smax2 ** L > self._eps_t:
                L += 1
            Cs = (W[small][:, None] * ss2[:, None] ** np.arange(L)[None, :]) * _J0_TAYLOR[None, :L]   # (ns, L)
            As = A[small]
        if big.any():
            Ab, yxb, Wb = A[big], y * x[big], W[big]
        step = max(1, 1_000_000 // x.size)
        for s0 in range(0, wb.size, step):
            wk = wb[s0:s0 + step]
            if small.any():
                ph = np.multiply.outer(wk, As)
                E = np.empty(ph.shape, dtype=complex)
                np.cos(ph, out=E.real)
                np.sin(ph, out=E.imag)
                R = E @ Cs                                   # (m, L)
                q = (wk / hi) ** 2
                acc = R[:, L - 1]
                for l in range(L - 2, -1, -1):
                    acc = acc * q + R[:, l]
                out[s0:s0 + step] += acc
            if big.any():
                ph = np.multiply.outer(wk, Ab)
                J = j0(np.multiply.outer(wk, yxb))
                J *= Wb
                out[s0:s0 + step] += np.einsum("mn,mn->m", J, np.cos(ph)) + 1j * np.einsum("mn,mn->m", J, np.sin(ph))
        return out

    def _tail_cached(self, bi, y):
        hit = self._tcache.get(bi)
        return hit is not None and hit[0] == y

    def _tail_prepare(self, bis, y):
        """Tail sums at the Chebyshev frequencies of the bands bis for source position y (cached per band).
        The complex Bessel functions of all bands are evaluated in one call."""
        todo = [bi for bi in bis if not self._tail_cached(bi, y)]
        if not todo:
            return
        zj, zh = [], []
        for bi in todo:
            b = self.bands[bi]
            if y == 0:
                self._tcache[bi] = (y, (0, self._tail_E(b).sum(axis=1)))
            elif b["hi"] * y * b["xb"] <= self.z_small:
                zj.append(bi)
            else:
                zh.append(bi)
        if zj:
            zs = [self.bands[bi]["wc"][:, None] * y * self.bands[bi]["xt"] for bi in zj]
            J = np.split(_j0_c(np.concatenate([z.ravel() for z in zs]), *self._zc), np.cumsum([z.size for z in zs])[:-1])
            for bi, z, Jb in zip(zj, zs, J):
                self._tcache[bi] = (y, (0, np.sum(self._tail_E(self.bands[bi]) * Jb.reshape(z.shape), axis=1)))
        if zh:
            zs = [self.bands[bi]["wc"][:, None] * y * self.bands[bi]["xt"] for bi in zh]
            cut = np.cumsum([z.size for z in zs])[:-1]
            h1, h2 = _hankel_e_c(np.concatenate([z.ravel() for z in zs]), *self._zc)
            for bi, z, a1, a2 in zip(zh, zs, np.split(h1, cut), np.split(h2, cut)):
                b = self.bands[bi]
                ph = np.exp(1j * b["wc"][:, None] * y * b["dx"])
                E = self._tail_E(b)
                self._tcache[bi] = (y, (1, np.sum(E * a1.reshape(z.shape) * ph, axis=1),
                                        np.sum(E * a2.reshape(z.shape) / ph, axis=1)))

    def _tail_values(self, bi, y):
        """Tail sums at the Chebyshev frequencies of band bi for source position y (cached per band)."""
        self._tail_prepare([bi], y)
        return self._tcache[bi][1]

    def _tail_fast(self, bi, y, wb):
        b = self.bands[bi]
        val = self._tail_values(bi, y)
        B = self._bary(b, wb)
        pre = 1j / (b["al"] * wb) * np.exp(1j * wb * (b["ub"] - b["psib"]))
        if val[0] == 0:
            return pre * (B @ val[1])
        ph = np.exp(1j * wb * y * b["xb"])
        return pre * 0.5 * (ph * (B @ val[1]) + (B @ val[2]) / ph)

    def _tail_exact(self, b, y, wb):
        """Tail at each frequency of wb (one y), with the Laguerre points of that frequency (few w)."""
        t = self.LX[None, :] / (b["al"] * wb[:, None])
        xt = np.sqrt(2 * (b["ub"] + 1j * t))
        E = self.LW[None, :] * np.exp(self.LX[None, :] * (1 - 1 / b["al"]) - 1j * wb[:, None] * (self.lens(xt) - b["psib"]))
        S = np.sum(E * _j0_c(wb[:, None] * y * xt, *self._zc), axis=1) if y > 0 else E.sum(axis=1)
        return 1j / (b["al"] * wb) * np.exp(1j * wb * (b["ub"] - b["psib"])) * S

    def _tail_direct(self, b, y, w):
        """Tail at one w for an array of source positions y (direct; complex Bessel J0)."""
        t = self.LX / (b["al"] * w)
        xt = np.sqrt(2 * (b["ub"] + 1j * t))
        E = self.LW * np.exp(self.LX * (1 - 1 / b["al"]) - 1j * w * (self.lens(xt) - b["psib"]))
        S = _j0_c(w * np.multiply.outer(np.atleast_1d(y), xt), *self._zc) @ E
        return 1j / (b["al"] * w) * np.exp(1j * w * (b["ub"] - b["psib"])) * S

    def _head_direct(self, b, y, w):
        """Head at one w for an array of source positions y: cluster polynomial + direct sums."""
        y = np.atleast_1d(y)
        out = np.zeros(y.size, dtype=complex) + self._cluster_profile(self._cluster(b), w, y)
        if b["x"].size == 0:
            return out
        c = b["W"] * np.exp(1j * w * b["A"])
        step = max(1, 2_000_000 // b["x"].size)
        for s in range(0, y.size, step):
            out[s:s + step] += j0(w * np.multiply.outer(y[s:s + step], b["x"])) @ c
        return out

    # cost model (seconds) for the per-frequency nodes of a band: direct sums vs NUFFT. Constants measured
    # on a Skylake-X core (numpy 2.3, FINUFFT 2.5); only their ratios matter for the choice.
    _C_TRIG, _C_J0 = 4.0e-8, 3.5e-8           # cos+sin per (w, node); scipy j0 per (w, node)
    _C_NU, _C_NU0 = 5.0e-8, 3.0e-4            # FINUFFT type 3 per point per row (eps 1e-14); per call
    _C_HENV = 8.0e-8                          # Hankel envelope per (Chebyshev frequency, node)

    def _fast_pays(self, bi, y, m):
        """True if the NUFFT path is predicted to be faster than direct sums for m frequencies."""
        b = self.bands[bi]
        if finufft is None or m < 4:
            return False
        x = b["x"]
        if x.size == 0:
            return False
        small = b["hi"] * y * x <= self.z_small
        ns = int(np.count_nonzero(small))
        nb = x.size - ns
        nc = b["cx"].size
        direct = m * (self._C_TRIG * x.size + self._C_J0 * nb)
        c_nu = self._C_NU * (0.6 + 0.4 * min(1.0, np.log10(1.0 / self.nufft_eps) / 14))
        fast = (self._C_NU0 * ((ns > 0) + (nb > 0)) + c_nu * (10 * (ns + m) * (ns > 0) + nc * (2 * nb + m) * (nb > 0))
                + (0.0 if (self._hcache.get(bi, (None,))[0] == y) else self._C_HENV * nc * nb))
        return fast < direct

    #  To be used by the user
    def spectrum(self, ws, y=None, interpolate=None):
        """F at an array of frequencies for one source position y (default: y0).
        interpolate: None = the solver's setting; True / False / 'auto' (see the class docstring)."""
        ws = np.atleast_1d(np.asarray(ws, dtype=float))
        y = self.y0 if y is None else abs(float(y))
        self._check_y(y)
        F = np.ones(ws.size, dtype=complex)
        pos = ws > 0
        wp = ws[pos]
        if wp.size == 0:
            return F
        self._check_w(wp)
        mode = self.interpolate if interpolate is None else interpolate
        with _blas_single():
            if mode and wp.size >= 64:
                Fi = self._spectrum_interp(wp, y, auto=(mode == "auto"))
                if Fi is not None:
                    F[pos] = Fi
                    return F
            F[pos] = self._spectrum_exact(wp, y)
        return F

    def _spectrum_exact(self, wp, y):
        total = np.empty(wp.size, dtype=complex)
        idx = self._band_index(wp)
        used, counts = np.unique(idx, return_counts=True)
        # tails by Chebyshev interpolation in ln w (n_cheb tail sums per band, cached per y), unless only a
        # few frequencies of a band without cached values are needed
        cheb = {int(bi) for bi, c in zip(used, counts)
                if 2 * c >= self.n_cheb or self._tail_cached(bi, y)}
        self._tail_prepare(sorted(cheb), y)

        def band(bi):
            sel = np.nonzero(idx == bi)[0]
            wb, b = wp[sel], self.bands[bi]
            if b["nc"] and b["cl"] is None and wb.size < self._M_CL:
                head = self._head_dsum(b, y, wb, all_nodes=True)     # few w: the cluster does not pay
            else:
                head = self._cluster_sum(self._cluster(b), y, wb)
                if self._fast_pays(bi, y, wb.size):
                    head = head + self._head_fast(bi, y, wb)
                else:
                    head = head + self._head_dsum(b, y, wb)
            tail = self._tail_fast(bi, y, wb) if bi in cheb else self._tail_exact(b, y, wb)
            return sel, head + tail

        for bi in used:
            sel, v = band(bi)
            total[sel] = v
        return -1j * wp * np.exp(0.5j * wp * y * y) * total

   #optional interpolation in w
    def _time_delay_range(self, y):
        """(T_min, T_max) of the Fermat potential T(s) = (s - y)^2/2 - psi(|s|) on the lens axis at the
        real images, at the points where images merge (caustics) and at s = 0 if psi(0) is finite.
        F(w) oscillates in w only through exp(i w T) at these times (plus smooth terms)."""
        if y in self._tdcache:
            return self._tdcache[y]
        if self._tdgrid is None:                             # y-independent part, once per solver
            lens = self.lens
            xmax = 1.5 * max(self.xb0, 2 * self.y_hi + 2)
            x = np.unique(np.concatenate([np.geomspace(1e-6, 1.0, 600), np.linspace(1.0, xmax, 1200)]))
            with np.errstate(all="ignore"):
                ps = lens.real(x)
                g = x - np.gradient(ps, x)                   # lens equation: g(|s|) = +-y
                p0 = lens.real(np.array([1e-12, 1e-9]))
            dg = np.diff(g)                                  # merging points (caustics): extrema of g
            ext = np.nonzero(np.isfinite(dg[:-1]) & np.isfinite(dg[1:]) & (np.sign(dg[:-1]) != np.sign(dg[1:])))[0] + 1
            core = p0[0] if (np.all(np.isfinite(p0)) and abs(p0[0] - p0[1]) < 1e-6 * (1 + abs(p0[0]))) else None
            self._tdgrid = (x, ps, g, ext, core)
        x, ps, g, ext, core = self._tdgrid
        Ts = []
        with np.errstate(all="ignore"):
            for sgn in (1.0, -1.0):                          # image side (s > 0) and opposite side
                T = 0.5 * (x - sgn * y) ** 2 - ps
                d = g - sgn * y
                ok = np.isfinite(d) & np.isfinite(T)
                roots = np.nonzero(ok[:-1] & ok[1:] & (np.sign(d[:-1]) != np.sign(d[1:])))[0]
                Ts += list(T[roots]) + list(T[roots + 1])
                Ts += list(T[ext][np.isfinite(T[ext])])
        if core is not None:
            Ts.append(0.5 * y * y - core)                    # cusp / core at the centre
        if not Ts:
            Ts = [0.5 * y * y - ps[0]]
        out = (float(min(Ts)), float(max(Ts)))
        if len(self._tdcache) > 64:
            self._tdcache.clear()
        self._tdcache[y] = out
        return out

    _INTERP_NMAX = 64     # largest number of Chebyshev points per panel

    def _interp_panels(self, w_lo, w_hi, B, tol):
        """
        Panels [a, b <= 4a] covering [w_lo, w_hi] and the number of Chebyshev points of each, chosen from
        the left so that each panel covers the most frequency per point. Points per panel from the bound for
        G(w) = F(w) e^{-i w Tc}: |G| <= e^{B |Im w|} (time delays within Tc +- B), singular at w = 0
        (_cheb_count), +2 for safety. Near w = 0 this gives geometric panels, further out panels of
        ~B*h ~ 25 radians.
        """
        key = (w_lo, w_hi, round(B, 12), tol)
        hit = self._pcache.get(key)
        if hit is not None:
            return hit
        edges, npts, a = [w_lo], [], w_lo
        lt = np.log(tol)
        nmax = self._INTERP_NMAX
        while a < w_hi * (1 - 1e-14):
            # b <= 4a: the bound uses e^{B |Im w|}, which is not a bound for the continuation of F far
            # below the real axis (e.g. poles of log-type potentials); wider panels can need more points
            hmax = min(0.5 * (w_hi - a), 1.5 * a)
            h = hmax * 2.0 ** (-np.arange(0, 48) / 2.0)
            n = _cheb_count(B * h, (a + h) / h, lt) + 1
            score = np.where(n <= nmax, 2 * h / n, -1.0)
            k = int(np.argmax(score))
            if score[k] <= 0:
                k = h.size - 1
            b = min(w_hi, a + 2 * h[k])
            if w_hi - b < 1e-3 * (b - a):
                b = w_hi
            edges.append(b)
            npts.append(int(min(n[k], nmax)) if np.isfinite(n[k]) else nmax)
            a = b
        out = (np.array(edges), np.array(npts, dtype=int))
        if len(self._pcache) > 64:
            self._pcache.clear()
        self._pcache[key] = out
        return out

    _CHECK_T = np.array([0.985, 0.45, -0.6, -0.97])         # check points per panel (panel coordinate)

    def _spectrum_interp(self, wp, y, auto=False):
        """
        F at the frequencies wp by Chebyshev interpolation on panels, or None if not worth it/failed.
        Panels and points per panel from _interp_panels (error bound). Each panel's interpolant is
        checked against exact values at four frequencies of the panel (computed in the same pass as the
        nodes); a panel whose error exceeds interp_tol/3 (the largest error between the check points can
        be ~3x larger) is split in two halves with the same number of points each, up to 3 times.

        Tested and keepig it here,becoz its faster than splines from scipy. And also accurate
        """
        T_lo, T_hi = self._time_delay_range(y)
        Tc = 0.5 * (T_lo + T_hi)
        B = 0.5 * (T_hi - T_lo) * 1.05 + 0.25               # half bandwidth of F e^{-i w Tc}, with margin
        w_lo, w_hi = wp.min(), wp.max()
        edges, npts = self._interp_panels(w_lo, w_hi, B, 0.1 * self.interp_tol)
        if (auto and wp.size < max(4000, 3 * int(npts.sum()))) or int(npts.sum()) >= wp.size:
            return None                                      # exact is cheaper
        todo = [(edges[p], edges[p + 1], int(npts[p])) for p in range(npts.size)]
        done = []                                            # (a, b, values at the n Chebyshev points)
        tc = self._CHECK_T
        for rnd in range(4):
            if sum(n for _, _, n in todo) + sum(g.size for _, _, g in done) >= wp.size:
                return None                                  # refinement made exact cheaper
            wn = [0.5 * (a + b) + 0.5 * (b - a) * _cheb_points(n) for a, b, n in todo]
            wt = [0.5 * (a + b) + 0.5 * (b - a) * tc for a, b, _ in todo]
            wall = np.concatenate(wn + wt)
            vals = self._spectrum_exact(wall, y)
            ns = np.cumsum([w.size for w in wn])
            Gs = np.split(vals[:ns[-1]] * np.exp(-1j * wall[:ns[-1]] * Tc), ns[:-1])
            Ft = vals[ns[-1]:].reshape(len(todo), tc.size)
            nxt = []
            for (a, b, n), G, w_t, F_t in zip(todo, Gs, wt, Ft):
                Fi = self._interp_eval(w_t, np.array([a, b]), [G]) * np.exp(1j * w_t * Tc)
                err = np.max(np.abs(Fi - F_t) / np.maximum(1, np.abs(F_t)))
                if err <= self.interp_tol / 3 or rnd == 3:
                    done.append((a, b, G))
                    if err > self.interp_tol:
                        _warn_once("interp", f"interpolation in w did not reach interp_tol={self.interp_tol:g}; computing all frequencies")
                        return None
                else:
                    m = 0.5 * (a + b)
                    nxt += [(a, m, n), (m, b, n)]
            todo = nxt
            if not todo:
                break
        done.sort(key=lambda t: t[0])
        edges = np.array([a for a, _, _ in done] + [done[-1][1]])
        return self._interp_eval(wp, edges, [g for _, _, g in done]) * np.exp(1j * wp * Tc)

    _BARY_W = {}

    @classmethod
    def _interp_eval(cls, w, edges, Gp):
        """Evaluate the panel-wise Chebyshev interpolant (values Gp[panel] at the Chebyshev points of the
        first kind) at w by the barycentric formula (Berrut & Trefethen 2004), panel by panel."""
        out = np.empty(w.size, dtype=complex)
        p = np.clip(np.searchsorted(edges, w, side="right") - 1, 0, len(Gp) - 1)
        order = np.argsort(p, kind="stable")
        bounds = np.searchsorted(p[order], np.arange(len(Gp) + 1))
        for q, G in enumerate(Gp):
            sel = order[bounds[q]:bounds[q + 1]]
            if sel.size == 0:
                continue
            n = G.size
            bw = cls._BARY_W.get(n)
            if bw is None:
                k = np.arange(n)
                bw = cls._BARY_W[n] = (_cheb_points(n), (-1.0) ** k * np.sin(np.pi * (k + 0.5) / n))
            xc, wts = bw
            t = (2 * w[sel] - (edges[q] + edges[q + 1])) / (edges[q + 1] - edges[q])
            d = np.subtract.outer(t, xc)
            hit = d == 0
            if hit.any():
                d[hit] = 1.0
            np.divide(wts, d, out=d)                         # barycentric terms (real)
            den = d.sum(axis=1)
            v = np.empty(t.size, dtype=complex)
            v.real = (d @ G.real) / den
            v.imag = (d @ G.imag) / den
            if hit.any():
                r, c = np.nonzero(hit)
                v[r] = G[c]
            out[sel] = v
        return out

    many = spectrum

    def profile(self, w, ys):
        """
        F at one frequency for an array of source positions ys (any shape).
        Created with the intention of continuous wave lensing analysis so that data generation for 
        long duration signals can be done in a fast and efficient manner for TDFstat analysis or other
        CW search pipelines being used in the collaboration. The evaluation happens as a rapid pace..

        Parameters:
        ----------
        w: float
           You frquency choice for a CW signal
        ys: array_like
           Pass your y array (may be from Einstein Crossing time for microlensing)

        Returns:
        --------
         Amplification factor
        
        """
        w = float(w)
        ys = np.abs(np.asarray(ys, dtype=float))
        shape, yr = ys.shape, ys.ravel()
        self._check_y(yr)
        if w == 0:
            return np.ones(shape, dtype=complex)
        self._check_w(np.array([w]))
        b = self.bands[self._band_index(np.array([w]))[0]]
        with _blas_single():
            F = -1j * w * np.exp(0.5j * w * yr * yr) * (self._head_direct(b, yr, w) + self._tail_direct(b, yr, w))
        return F.reshape(shape)

    def evaluate(self, ws, ys):
        """Table F[k, m] = F(ws[m], ys[k]): one spectrum per source position (or one profile per
        frequency if there are more positions than frequencies). With n_workers > 1, rows beyond the
        first ~50 ms go to worker processes when that is predicted to be clearly faster.

        Having the multiprocessing turned on by default is slower for certain regions of the parameter space.
        Hence it is conditionally turned on automatically when needed.
        
         I
        """
        ws = np.atleast_1d(np.asarray(ws, dtype=float))
        ys = np.abs(np.atleast_1d(np.asarray(ys, dtype=float))).ravel()
        #Doing the usual checks
        self._check_y(ys)
        self._check_w(ws[ws > 0])

        by_y = ys.size <= ws.size
        n_rows = ys.size if by_y else ws.size

        out = np.empty((ys.size, ws.size), dtype=complex)

        def put(rows, val):
            if by_y:
                out[rows] = val
            else:
                out[:, rows] = val
        
        pmax = self._pmax(n_rows // 2)
        t0, k = time.perf_counter(), 0
        while k < n_rows:                                    # rows here until ~50 ms have passed
            put(k, self.spectrum(ws, ys[k]) if by_y else self.profile(ws[k], ys))
            k += 1
            if pmax > 1 and time.perf_counter() - t0 >= 0.05:
                break
        if k < n_rows and pmax > 1:
            t_rest = (time.perf_counter() - t0) / k * (n_rows - k)
            n = max(1, min(self.n_workers, available_cores()))
            start = 0.0 if (self._pool is not None and self._pool_pid == os.getpid()) else _SPAWN_COST * (n + 1)
            if t_rest / (1 + _PAR_EFF * (n - 1)) + start < _MIN_GAIN * t_rest:
                rest = np.arange(k, n_rows)
                n_tasks = min(rest.size, 4 * n)
                rows = [rest[i::n_tasks] for i in range(n_tasks)]          # round robin: balanced costs
                tasks = [(True, ws, ys[r]) if by_y else (False, ws[r], ys) for r in rows]
                try:
                    with _quiet_fork():
                        res = list(self._get_pool().map(_eval_rows, tasks))
                except BrokenProcessPool:
                    self.close()
                    raise
                for r, v in zip(rows, res):
                    put(r, v)
                return out
        for kk in range(k, n_rows):
            put(kk, self.spectrum(ws, ys[kk]) if by_y else self.profile(ws[kk], ys))
        return out

    def __call__(self, w, y=None):
        return self.spectrum([w], y)[0]


# Lighter settings for parameter estimation: fewer nodes and a looser FINUFFT tolerance, accuracy ~1e-8
# (n_cheb=13 with band_ratio=8 is the former n_cheb=10 with band_ratio=4; interp_tol becomes 1e-7)
# in F (far below detector noise). Usage: Solver1D(lens, y, wmin, wmax, y_radius=R, **PE_SETTINGS)
PE_SETTINGS = dict(ngl=24, nlag=24, n_cheb=13, nufft_eps=1e-9)


# ================================================================== convenience functions
def _err_kw(s, kw):
    return dict(kw, ngl=kw.get("ngl", 32) + 8, nlag=kw.get("nlag", 30) + 10, xb_min=1.5 * s.xb0)


def amplification(w, y, lens, estimate_error=False, **kw):
    """
    F(w, y) for scalars or arrays, with numpy broadcasting: the result has the broadcast shape of w and
    y, and one solver serves all points. The fastest path is chosen from the inputs:

        amplification(0.5, 1.2, lens)                       one value
        amplification(ws, 1.2, lens)                        many frequencies, one y  -> s.spectrum
        amplification(0.5, ys, lens)                        one frequency, many y    -> s.profile
        amplification(ws[None, :], ys[:, None], lens)       table (len(ys), len(ws)) -> s.evaluate
        amplification(ws, ys, lens)                         pairs (ws[i], ys[i]), same shape

    Pairs are grouped by the coordinate with fewer distinct values (one spectrum per distinct y, or
    one profile per distinct w); if they fill most of a grid, the table is computed instead.
    It supports all keywords of Solver1D, e.g. nlag, ngl, n_cheb, y_radius, threads, n_workers,
    interpolate, and **PE_SETTINGS.

    Parameters:
    -----------
    w : float or array_like
        Frequencies (w = 0 gives F = 1).
    y : float or array_like
        Source positions (|y| is used); broadcast against w.
    lens : Potential or callable
        Lens potential psi(x).
    estimate_error : bool, optional
        If True, also returns |F - F'| with F' from a solver with more nodes and a farther split
        (an estimate of the error, same shape as F).
    **kw : dict
        Keyword arguments of Solver1D (see the Solver1D class for available options).

    Returns:
    --------
    F with the broadcast shape of w and y (a complex scalar if both are scalars); with
    estimate_error, the tuple (F, error).
    """
    wb, yb = np.broadcast_arrays(np.asarray(w, dtype=float), np.abs(np.asarray(y, dtype=float)))
    shape, wf, yf = wb.shape, wb.ravel(), yb.ravel()
    pos = wf > 0
    if not pos.any():
        F = np.ones(shape, dtype=complex)
        F, E = (F[()], np.zeros(shape)[()]) if shape == () else (F, np.zeros(shape))
        return (F, E) if estimate_error else F
    wlo, whi, yu = wf[pos].min(), wf[pos].max(), np.unique(yf[pos])
    with _solver_for(lens, wlo, whi, yu, kw) as s:
        F = _evaluate_pairs(s, wf, yf)
    if estimate_error:
        with _solver_for(lens, wlo, whi, yu, _err_kw(s, kw)) as s2:
            E = np.abs(_evaluate_pairs(s2, wf, yf) - F)
    F = F.reshape(shape)
    if shape == ():
        F = F[()]
    if not estimate_error:
        return F
    E = E.reshape(shape)
    return F, (E[()] if shape == () else E)


def _solver_for(lens, wmin, wmax, ys, kw):
    """One solver for the frequencies [wmin, wmax] and the (distinct, non-negative) source positions ys.
    A y_radius in kw is kept if it is larger than needed."""
    kw = dict(kw)
    r_user = float(kw.pop("y_radius", 0.0))
    if ys.size == 1 and r_user == 0:
        return Solver1D(lens, float(ys[0]), wmin, wmax, **kw)
    yc, R = _y_range(ys)
    if ys.size == 1:
        yc = float(ys[0])
    return Solver1D(lens, yc, wmin, wmax, y_radius=max(R, r_user), **kw)


def _evaluate_pairs(s, wf, yf):
    """F at the pairs (wf[i], yf[i]) (flat arrays) with the solver s, by the cheapest path."""
    out = np.ones(wf.size, dtype=complex)
    pos = wf > 0
    wq, yq = wf[pos], yf[pos]
    uw, iw = np.unique(wq, return_inverse=True)
    uy, iy = np.unique(yq, return_inverse=True)
    if uy.size == 1:                                         # one source position: one spectrum
        vals = s.spectrum(uw, uy[0])[iw]
    elif uw.size == 1:                                       # one frequency: one profile
        vals = s.profile(uw[0], uy)[iy]
    elif uw.size * uy.size <= 4 * wq.size:                   # (nearly) a grid: the table
        vals = s.evaluate(uw, uy)[iy, iw]
    else:                                                    # scattered pairs
        vals = np.empty(wq.size, dtype=complex)
        if uy.size <= uw.size:
            for k in range(uy.size):
                sel = iy == k
                vals[sel] = s.spectrum(wq[sel], uy[k])
        else:
            for k in range(uw.size):
                sel = iw == k
                vals[sel] = s.profile(uw[k], yq[sel])
    out[pos] = vals
    return out


def amplification_spectrum(ws, y, lens, estimate_error=False, **kw):
    """

    F on an array of frequencies for one source position: the frequency spectrum of the amplification
    (one solver, s.spectrum). Formerly amplification_many.
    It supports all keywords of Solver1D, e.g. nlag, ngl, n_cheb, y_radius, threads, n_workers,interpolate etc.

    Parameters:
    -----------
    ws : array_like
        Frequencies at which to compute the amplification factor.
    y : float
        Source position.
    lens : callable
        Lens model function.
    estimate_error : bool, optional
        If True, also returns an estimate of the error in the computed amplification factor.
    **kw : dict
        Additional keyword arguments to pass to the Solver1D class.(See the Solver1D class for available options.)

    Returns :   
    --------    
    Amplification factor F at the specified frequencies. If estimate_error is True, also returns an array of estimated errors.
    """
    ws = np.atleast_1d(np.asarray(ws, dtype=float))
    wp = ws[ws > 0]
    if wp.size == 0:
        F = np.ones(ws.size, dtype=complex)
        return (F, np.zeros(ws.size)) if estimate_error else F
    with Solver1D(lens, y, wp.min(), wp.max(), **kw) as s:
        F = s.spectrum(ws)
    if not estimate_error:
        return F
    with Solver1D(lens, y, wp.min(), wp.max(), **_err_kw(s, kw)) as s2:
        return F, np.abs(s2.spectrum(ws) - F)


def _y_range(ys):
    """Compute the center and radius of the range of source positions.
    Parameters:
    -----------
    ys : array_like
        Array of source positions.
    Returns:
    --------
    yc : float
        Center of the range of source positions.
    R : float
        Radius of the range of source positions.
    """
    ya = np.abs(np.asarray(ys, dtype=float)).ravel()
    lo, hi = ya.min(), ya.max()
    return 0.5 * (lo + hi), 0.5 * (hi - lo) * (1 + 1e-9) + 1e-12


def amplification_profile(w, ys, lens, **kw):

    """
    F at one frequency for many source positions ys (any shape): the profile of the amplification in y
    (one solver, s.profile).
    It supports all keywords of Solver1D, e.g. nlag, ngl, n_cheb, y_radius, threads, n_workers,interpolate etc.

    Parameters:
    -----------
    w : float
        Frequency at which to compute the amplification factor.
    ys : array_like
        Array of source positions.
    lens : callable
        Lens model function.
    **kw : dict
        Additional keyword arguments to pass to the Solver1D class.(See the Solver1D class for available options.)
    
    Returns:
    --------
    Amplification factor F at the specified frequency for each source position in ys.   
    """
    yc, R = _y_range(ys)
    with Solver1D(lens, yc, w, w, y_radius=R, **kw) as s:
        return s.profile(w, ys)


def amplification_table(ws, ys, lens, **kw):
    """
    Table F[k, m] = F(ws[m], ys[k]) for many frequencies and source positions (one solver,
    s.evaluate). Formerly amplification_grid.
    Pass n_workers=None (all cores) or n_workers=k to compute the rows in parallel.
    It supports all keywords of Solver1D, e.g. nlag, ngl, n_cheb, y_radius, threads, n_workers,interpolate etc.

    Parameters:
    -----------
    ws : array_like
        Frequencies at which to compute the amplification factor.
    ys : array_like
        Array of source positions.      
    lens : callable
        Lens model function.
    **kw : dict
        Additional keyword arguments to pass to the Solver1D class.(See the Solver1D class for available options.)  
    
    Returns:
    --------
    2D array of shape (len(ys), len(ws)) containing the amplification factors for each combination of frequency and source position.        
    """
    ws = np.atleast_1d(np.asarray(ws, dtype=float))
    ys = np.atleast_1d(np.asarray(ys, dtype=float)).ravel()
    wp = ws[ws > 0]
    if wp.size == 0:
        return np.ones((ys.size, ws.size), dtype=complex)
    yc, R = _y_range(ys)
    with Solver1D(lens, yc, wp.min(), wp.max(), y_radius=R, **kw) as s:
        return s.evaluate(ws, ys)


# Old names with Deprecation warning
def amplification_many(ws, y, lens, estimate_error=False, **kw):
    """Former name of amplification_spectrum (deprecated)."""
    warnings.warn("amplification_many is now amplification_spectrum", DeprecationWarning, stacklevel=2)
    return amplification_spectrum(ws, y, lens, estimate_error=estimate_error, **kw)


def amplification_grid(ws, ys, lens, **kw):
    """Former name of amplification_table (deprecated)."""
    warnings.warn("amplification_grid is now amplification_table", DeprecationWarning, stacklevel=2)
    return amplification_table(ws, ys, lens, **kw)