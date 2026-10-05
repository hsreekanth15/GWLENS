
import os
import math
import time
import signal
import weakref
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager
import numpy as np
from scipy.special import j0, y0, jv, hankel1e, hankel2e, roots_legendre, roots_laguerre
try:
    import finufft                     
except ImportError:
    finufft = None


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
    exact = np.isclose(d, 0.0, atol=1e-15)
    d[exact] = 1.0
    M = wts[None, :] / d
    M /= M.sum(axis=1, keepdims=True)
    rows = np.any(exact, axis=1)
    M[rows] = exact[rows].astype(float)
    return M


_J0_TAYLOR = np.array([(-1.0) ** k / (4.0 ** k * math.factorial(k) ** 2) for k in range(24)])

# Hankel expansion taken from here (https://dlmf.nist.gov/10.17). See 10.17.5 for more details.
# It coverges very fast so only few terms are needed.
# relative error < 1e-15 for z >= 25. Below that: J0 + i Y0 from scipy.
#The below factors are used in the functions _h_asym() and it is hardcoded because the scipy's default function call is 130ns per value
# _h_asym() function is called a million times so it is needed to speed things up....
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
class Solver1D:
    """
    Everything that does not depend on (w, y), for all wmin <= w <= wmax and all source positions
    

    Evaluation (all reuse the same nodes; psi is never evaluated again for the head):
      s(w, y=None)            F at one frequency, one source position (default: y0)
      s.spectrum(ws, y=None)  many frequencies, one source position   -> batched 1D non-uniform FFT
      s.profile(w, ys)        one frequency, many source positions    -> direct sums
      s.evaluate(ws, ys)      table (len(ys), len(ws))

    Accuracy/tuning  parameters (defaults give ~1e-12): 
    cycles, ngl (phase cycles / Gauss-Legendre nodes per head piece), nlag (Laguerre nodes on the tail), n_cheb (Chebyshev frequencies per band),
    band_ratio (width of the frequency bands), xb_min (move the split outward = convergence test),
    nufft_eps (FINUFFT tolerance), z_small (w y x below which J0 is used by its Taylor series).
    PE_SETTINGS: lighter settings for parameter estimation (accuracy ~1e-8).

    Interpolation (for very many frequencies)
      interpolate  False (default): every frequency is computed. True: F is computed on Chebyshev
                   panels sized by the lens time delays and interpolated to the requested
                   frequencies (worth it from ~10^4 frequencies on). 'auto': whichever is cheaper.
                   Can also be passed per call: s.spectrum(ws, y, interpolate=True).
      interp_tol   the interpolant is checked against exact values at a few frequencies per call;
                   if the relative error exceeds interp_tol the panels are refined (max 3 times),
                   then the exact path is used.

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
        Number of Gauss-Legendre quadrature nodes per head piece. Higher values
        increase accuracy but raise computational cost (default: 32).
    nlag : int, optional
        Number of Gauss-Laguerre quadrature nodes on the tail portion of the
        integration domain (default: 30).
    n_cheb : int, optional
        Number of Chebyshev polynomial frequencies per frequency band (default: 14).
    band_ratio : float, optional
        Width ratio defining the frequency bands. Controls the logarithmic spacing
        of frequency intervals (default: 4.0).
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
    interpolate : bool, optional
        If ``True``, applies interpolation to the calculated frequencies.
        Useful for parameter estimation and smoothing results. Can reduce
        artifacts at band boundaries (default: False).
 
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

    def __init__(self, lens, y=0.0, wmin=1e-3, wmax=1e3, y_radius=0.0, cycles=8, ngl=32, nlag=30, n_cheb=14,
                 band_ratio=4.0, xb_min=0.0, nufft_eps=1e-14, z_small=1.0, n_workers=1, threads=None,
                 interpolate=False, interp_tol=1e-10):
        """
        Initialize the 1D lensing solver.

        Parameters:
        -----------
        lens :        Potential object (or a callable psi(x) for the lensing potential)
        y           : source position (default: 0.0)
        wmin, wmax  : frequency range (default: 1e-3, 1e3)
        y_radius    : half-width of the source position range (default: 0.0)
        cycles      : max phase cycles per head piece (default: 8)
        ngl         : Gauss-Legendre nodes per head piece (default: 32)
        nlag        : Gauss-Laguerre nodes on the tail (default: 30)
        n_cheb      : Chebyshev frequencies per band (default: 14)
        band_ratio  : width of the frequency bands (default: 4.0)
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
        self.interp_tol = float(interp_tol)
        self._tdcache = {}
        self._pool = self._pool_fin = self._pool_pid = None

        self.GX, self.GW = _gl(self.ngl)
        self.LX, self.LW = _lag(self.nlag)

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
        n_lin = int(max(3000, 3 * self.wmax * self.ub0 / (2 * np.pi)))
        ug = np.unique(np.concatenate([np.linspace(self.u_floor, self.ub0, n_lin),
                                       np.geomspace(self.u_floor, self.ub_max, 20000)]))
        x = np.sqrt(2 * ug)
        with np.errstate(all="ignore"):
            d = np.abs(np.diff(ug - lens.real(x)))
        d[~np.isfinite(d)] = 0.0
        self.ug, self.V1 = ug, np.concatenate([[0.0], np.cumsum(d)]) + self.y_hi * (x - x[0])

        # ---- frequency bands [w_hi / band_ratio, w_hi], from wmax down to wmin
        self.bands, hi = [], self.wmax
        while True:
            lo = max(self.wmin, hi / band_ratio)
            self.bands.append(self._make_band(lo, hi))
            if lo <= self.wmin * (1 + 1e-12):
                break
            hi = lo
        self._tcache = {}
        self.info = dict(x_b=self.xb0, q_inf=q, y_range=(self.y_lo, self.y_hi),
                         bands=[(b["lo"], b["hi"], b["ub"], b["x"].size) for b in self.bands],
                         n_nodes=int(sum(b["x"].size for b in self.bands)))

    # ------------------------------------------------------------------ one frequency band
    def _make_band(self, lo, hi):
        """
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
        """
        lens = self.lens
        ub = max(self.ub0, 20 / lo)                          # small-w rule
        xb = np.sqrt(2 * ub)
        # head nodes: a piece every `cycles` phase cycles at w = hi, geometric grading towards 0
        V_ub = np.interp(ub, self.ug, self.V1)
        cuts = np.interp(np.arange(0, V_ub, 2 * np.pi * self.cycles / hi), self.V1, self.ug)
        geo = np.geomspace(self.u_floor, ub, int(np.ceil(np.log(ub / self.u_floor))) + 1)
        e = np.unique(np.concatenate([[0.0], geo, cuts[cuts < ub], np.linspace(0, ub, 5), [ub]]))
        # the cuts bound the phase change per piece; also bound (largest phase rate) x (width), which
        # matters where the rate varies strongly inside a piece (stationary points: chirps)
        m = max(1, int(np.searchsorted(self.ug, ub, side="right")))
        rate = (np.diff(self.V1) / np.diff(self.ug))[:m]   # phase per unit w per unit u, up to ub
        i0 = np.clip(np.searchsorted(self.ug, e[:-1], side="right") - 1, 0, rate.size - 1)
        peak = np.maximum.reduceat(rate, i0)                 # largest rate in each piece
        n_sub = np.maximum(1, np.ceil(peak * np.diff(e) * hi / (1.25 * 2 * np.pi * self.cycles))).astype(int)
        if np.any(n_sub > 1):
            e = np.unique(np.concatenate([e] + [np.linspace(e[k], e[k + 1], n + 1) for k, n in enumerate(n_sub) if n > 1]))
        mid, half = (e[1:] + e[:-1]) / 2, (e[1:] - e[:-1]) / 2
        u = (mid[:, None] + half[:, None] * self.GX[None, :]).ravel()
        W = (half[:, None] * self.GW[None, :]).ravel()
        x = np.sqrt(2 * u)
        with np.errstate(all="ignore"):
            A = u - lens.real(x)
        # tail: decay rate for the largest y, psi at the complex Laguerre points of every
        # Chebyshev frequency of the band (y-independent)
        al = 1 - (self.y_hi + float(lens.dpsi(xb))) / xb
        if al <= 0:
            raise RuntimeError("tail does not decay; increase xb_min")
        psib = float(np.real(np.asarray(lens(np.array([xb + 0j])))).ravel()[0])
        if hi / lo < 1 + 1e-12:
            cx, zc = np.zeros(1), np.array([np.log(hi)])
        else:
            cx = _cheb_points(self.n_cheb)
            zc = 0.5 * (np.log(lo) + np.log(hi)) + 0.5 * (np.log(hi) - np.log(lo)) * cx
        wc = np.exp(zc)
        t = self.LX[None, :] / (al * wc[:, None])
        xt = np.sqrt(2 * (ub + 1j * t))
        E = self.LW[None, :] * np.exp(self.LX[None, :] * (1 - 1 / al) - 1j * wc[:, None] * (lens(xt) - psib))
        return dict(lo=lo, hi=hi, ub=ub, xb=xb, al=al, psib=psib, x=x, W=W, A=A, cx=cx, wc=wc, xt=xt,
                    dx=xt - xb, E=E, zlo=np.log(lo), zhi=np.log(hi))

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
        for i, b in enumerate(self.bands):
            sel = (idx < 0) & (wp >= b["lo"] * (1 - 1e-12)) & (wp <= b["hi"] * (1 + 1e-12))
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
            _warn_once("nufft", "finufft is not installed: frequency sums are done directly (slower). pip install finufft")
        out = np.empty((C.shape[0], wb.size), dtype=complex)
        step = max(1, 2_000_000 // max(src.size, 1))
        for s in range(0, wb.size, step):
            out[:, s:s + step] = C @ np.exp(1j * np.outer(src, wb[s:s + step]))
        return out

    def _head_fast(self, b, y, wb):
        """sum_j W_j J0(w y x_j) exp(i w A_j) for many w in one band (Taylor + Hankel/Chebyshev + NUFFT)."""
        x, W, A, hi = b["x"], b["W"], b["A"], b["hi"]
        out = np.zeros(wb.size, dtype=complex)
        s = hi * y * x                                       # largest w y x of each node in the band
        small = s <= self.z_small
        if small.any():                                      # J0 = sum_k a_k (w/hi)^2k s^2k
            ss2 = s[small] ** 2
            smax2 = ss2.max()
            K = 1
            while K < _J0_TAYLOR.size and abs(_J0_TAYLOR[K]) * smax2 ** K > 1e-17:
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
            xl, Wl, Al = x[big], W[big], A[big]
            H = _h_env(b["wc"][:, None] * (y * xl)[None, :]) * Wl[None, :]
            R = self._nufft(np.concatenate([Al + y * xl, Al - y * xl]), np.concatenate([H, H.conj()], axis=1), wb)
            out += 0.5 * np.einsum("mc,cm->m", self._bary(b, wb), R)
        return out

    def _tail_values(self, bi, y):
        """Tail sums at the Chebyshev frequencies of band bi for source position y (cached per band)."""
        hit = self._tcache.get(bi)
        if hit is not None and hit[0] == y:
            return hit[1]
        b = self.bands[bi]
        wc, E = b["wc"][:, None], b["E"]
        if y == 0:
            val = (0, E.sum(axis=1))
        elif b["hi"] * y * b["xb"] <= self.z_small:
            val = (0, np.sum(E * jv(0, wc * y * b["xt"]), axis=1))
        else:
            z = wc * y * b["xt"]
            val = (1, np.sum(E * hankel1e(0, z) * np.exp(1j * wc * y * b["dx"]), axis=1),
                   np.sum(E * hankel2e(0, z) * np.exp(-1j * wc * y * b["dx"]), axis=1))
        self._tcache[bi] = (y, val)
        return val

    def _tail_fast(self, bi, y, wb):
        b = self.bands[bi]
        val = self._tail_values(bi, y)
        B = self._bary(b, wb)
        pre = 1j / (b["al"] * wb) * np.exp(1j * wb * (b["ub"] - b["psib"]))
        if val[0] == 0:
            return pre * (B @ val[1])
        ph = np.exp(1j * wb * y * b["xb"])
        return pre * 0.5 * (ph * (B @ val[1]) + (B @ val[2]) / ph)

    def _tail_direct(self, b, y, w):
        """Tail at one w for an array of source positions y (direct; complex Bessel J0).

        To be tested!!!!!
        Use with Caution!!!
        
        """
        t = self.LX / (b["al"] * w)
        xt = np.sqrt(2 * (b["ub"] + 1j * t))
        E = self.LW * np.exp(self.LX * (1 - 1 / b["al"]) - 1j * w * (self.lens(xt) - b["psib"]))
        S = jv(0, w * np.multiply.outer(np.atleast_1d(y), xt)) @ E
        return 1j / (b["al"] * w) * np.exp(1j * w * (b["ub"] - b["psib"])) * S

    def _head_direct(self, b, y, w):
        """Head at one w for an array of source positions y (direct).

        To be tested!!!!!
        Use with Caution!!!
        
        """
        y = np.atleast_1d(y)
        c = b["W"] * np.exp(1j * w * b["A"])
        out = np.empty(y.size, dtype=complex)
        step = max(1, 2_000_000 // b["x"].size)
        for s in range(0, y.size, step):
            out[s:s + step] = j0(w * np.multiply.outer(y[s:s + step], b["x"])) @ c
        return out

    def _spectrum_direct(self, b, y, wb):
        """Head + tail at the frequencies wb of band b for one y, by direct sums (few frequencies)."""
        x, W, A = b["x"], b["W"], b["A"]
        head = np.empty(wb.size, dtype=complex)
        step = max(1, 1_000_000 // x.size)
        for s in range(0, wb.size, step):
            wk = wb[s:s + step, None]
            head[s:s + step] = (j0(wk * (y * x)[None, :]) * np.exp(1j * wk * A[None, :])) @ W
        t = self.LX[None, :] / (b["al"] * wb[:, None])
        xt = np.sqrt(2 * (b["ub"] + 1j * t))
        E = self.LW[None, :] * np.exp(self.LX[None, :] * (1 - 1 / b["al"]) - 1j * wb[:, None] * (self.lens(xt) - b["psib"]))
        S = np.sum(E * jv(0, wb[:, None] * y * xt), axis=1)
        return head + 1j / (b["al"] * wb) * np.exp(1j * wb * (b["ub"] - b["psib"])) * S

    def _fast_pays(self, b, y, m):
        """Cost model (seconds, this design's measured constants): direct sums vs Hankel/Chebyshev + NUFFT."""
        if finufft is None or m < 4:
            return False
        x = b["x"]
        z_hi, z_lo = b["hi"] * y * x, b["lo"] * y * x
        n_small = int(np.count_nonzero(z_hi <= self.z_small))
        n_far = int(np.count_nonzero(z_lo >= _Z_ASYM))
        n_mid = x.size - n_small - n_far
        nc = b["cx"].size
        direct = m * (6e-8 * x.size + 2e-5)
        fast = (4e-3 + nc * (1.3e-7 * n_mid + 5e-8 * n_far)
                + 3.5e-8 * (10 * n_small + 2 * nc * (n_mid + n_far)) + 3e-7 * m)
        return fast < direct

    # ------------------------------------------------------------------ public evaluation
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
        for bi in np.unique(idx):
            sel = idx == bi
            wb, b = wp[sel], self.bands[bi]
            if self._fast_pays(b, y, wb.size):
                total[sel] = self._head_fast(b, y, wb) + self._tail_fast(bi, y, wb)
            else:
                total[sel] = self._spectrum_direct(b, y, wb)
        return -1j * wp * np.exp(0.5j * wp * y * y) * total

    # ------------------------------------------------------------------ optional interpolation in w
    def _time_delay_range(self, y):
        """(T_min, T_max) of the Fermat potential T(s) = (s - y)^2/2 - psi(|s|) on the lens axis at the
        real images, at the points where images merge (caustics) and at s = 0 if psi(0) is finite.
        F(w) oscillates in w only through exp(i w T) at these times (plus smooth terms)."""
        if y in self._tdcache:
            return self._tdcache[y]
        lens = self.lens
        xmax = 1.5 * max(self.xb0, 2 * y + 2)
        x = np.unique(np.concatenate([np.geomspace(1e-6, 1.0, 3000), np.linspace(1.0, xmax, 6000)]))
        with np.errstate(all="ignore"):
            g = x - lens.dpsi(x)                             # lens equation: g(|s|) = +-y
            ps = lens.real(x)
        Ts = []
        for sgn in (1.0, -1.0):                              # image side (s > 0) and opposite side
            T = 0.5 * (x - sgn * y) ** 2 - ps
            d = g - sgn * y
            ok = np.isfinite(d) & np.isfinite(T)
            roots = np.nonzero(ok[:-1] & ok[1:] & (np.sign(d[:-1]) != np.sign(d[1:])))[0]
            Ts += list(T[roots]) + list(T[roots + 1])
            dg = np.diff(g)                                  # merging points: extrema of g
            ext = np.nonzero(np.isfinite(dg[:-1]) & np.isfinite(dg[1:]) & (np.sign(dg[:-1]) != np.sign(dg[1:])))[0] + 1
            Ts += list(T[ext][np.isfinite(T[ext])])
        with np.errstate(all="ignore"):
            p0 = lens.real(np.array([1e-12, 1e-9]))
        if np.all(np.isfinite(p0)) and abs(p0[0] - p0[1]) < 1e-6 * (1 + abs(p0[0])):
            Ts.append(0.5 * y * y - p0[0])                   # cusp / core at the centre
        if not Ts:
            Ts = [0.5 * y * y - ps[0]]
        out = (float(min(Ts)), float(max(Ts)))
        if len(self._tdcache) > 64:
            self._tdcache.clear()
        self._tdcache[y] = out
        return out

    def _interp_panels(self, w_lo, w_hi, B):
        """Panel edges: phase B*(width) <= 10 rad and width <= left end (geometric near w = 0)."""
        edges, a = [w_lo], w_lo
        while a < w_hi * (1 - 1e-14):
            h = min(10.0 / B if B > 0 else np.inf, a, w_hi - a)
            a = a + h if w_hi - (a + h) > 1e-3 * h else w_hi
            edges.append(a)
        return np.array(edges)

    def _spectrum_interp(self, wp, y, auto=False, n_p=24):
        """
        F at the frequencies wp by Chebyshev interpolation on panels, or None if not worth it/failed.

        Tested and keepig it here,becoz its faster than splines from scipy. And also accurate

        
        """
        T_lo, T_hi = self._time_delay_range(y)
        Tc = 0.5 * (T_lo + T_hi)
        B = 0.5 * (T_hi - T_lo) * 1.2 + 0.5                 # half bandwidth of F e^{-i w Tc}, with margin
        w_lo, w_hi = wp.min(), wp.max()
        xc = _cheb_points(n_p)
        for attempt in range(4):
            edges = self._interp_panels(w_lo, w_hi, B)
            n_nodes = (edges.size - 1) * n_p
            if (auto and wp.size < max(4000, 3 * n_nodes)) or n_nodes >= wp.size:
                return None                                  # exact is cheaper
            mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
            wn = (mid[:, None] + half[:, None] * xc[None, :]).ravel()
            G = (self._spectrum_exact(wn, y) * np.exp(-1j * wn * Tc)).reshape(-1, n_p)
            F = self._interp_eval(wp, edges, G, xc) * np.exp(1j * wp * Tc)
            # check at a few frequencies where interpolation errors peak (near panel ends)
            k = np.unique(np.linspace(0, edges.size - 2, min(6, edges.size - 1)).astype(int))
            wt = np.clip(np.concatenate([mid[k] + 0.985 * half[k], mid[k] - 0.6 * half[k]]), w_lo, w_hi)
            Ft = self._spectrum_exact(wt, y)
            err = np.max(np.abs(self._interp_eval(wt, edges, G, xc) * np.exp(1j * wt * Tc) - Ft) / np.maximum(1, np.abs(Ft)))
            if err <= self.interp_tol:
                return F
            B *= 2.0                                         # unexpected structure: finer panels
        _warn_once("interp", f"interpolation in w did not reach interp_tol={self.interp_tol:g}; computing all frequencies")
        return None

    @staticmethod
    def _interp_eval(w, edges, G, xc):
        """Evaluate the panel-wise Chebyshev interpolant (values G[panel, node] at the points xc) at w:
        Chebyshev coefficients per panel, then T_k(t) by recurrence on the sorted targets of each panel."""
        n = xc.size
        Tm = np.cos(np.outer(np.arange(n), np.pi * (np.arange(n) + 0.5) / n))
        coef = (2.0 / n) * G @ Tm.T
        coef[:, 0] *= 0.5
        cr, ci = np.ascontiguousarray(coef.real.T), np.ascontiguousarray(coef.imag.T)   # (n, panels)
        order = None if np.all(w[1:] >= w[:-1]) else np.argsort(w, kind="stable")
        ws = w if order is None else w[order]
        cut = np.searchsorted(ws, edges[1:-1], side="right")
        bounds = np.concatenate([[0], cut, [ws.size]])
        res = np.empty(ws.size, dtype=complex)
        for p in range(edges.size - 1):
            i0, i1 = bounds[p], bounds[p + 1]
            if i1 <= i0:
                continue
            t = (2 * ws[i0:i1] - (edges[p] + edges[p + 1])) / (edges[p + 1] - edges[p])
            V = np.empty((n, i1 - i0))                       # V[k] = T_k(t), rows contiguous
            V[0] = 1.0
            V[1] = t
            t2 = 2 * t
            for k in range(2, n):
                np.multiply(t2, V[k - 1], out=V[k])
                V[k] -= V[k - 2]
            res.real[i0:i1] = cr[:, p] @ V
            res.imag[i0:i1] = ci[:, p] @ V
        if order is None:
            return res
        out = np.empty_like(res)
        out[order] = res
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
# in F (far below detector noise). Usage: Solver1D(lens, y, wmin, wmax, y_radius=R, **PE_SETTINGS)
PE_SETTINGS = dict(ngl=24, nlag=24, n_cheb=10, nufft_eps=1e-9)


# ================================================================== convenience functions
def _err_kw(s, kw):
    return dict(kw, ngl=kw.get("ngl", 32) + 8, nlag=kw.get("nlag", 30) + 10, xb_min=1.5 * s.xb0)


def amplification(w, y, lens, estimate_error=False, **kw):
    """
    F(w, y) at one frequency with optional error estimation.
    The function is comparitively slow for many frequencies or source positions; use amplification_many() or amplification_grid() instead
    according to your use case. The amplification_many() is faster for many frequencies at one source position, while amplification_grid() is faster for many source positions at one frequency.
    It supports all keywords of Solver1D, e.g. nlag, ngl, n_cheb, y_radius, threads, n_workers,interpolate etc.

    Parameters:
    -----------
    w : float
        Frequency at which to compute the amplification factor.
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
    Amplification factor F at the specified frequency. If estimate_error is True, also returns an estimate of the error in the computed amplification factor.
    """
    w = float(w)
    if w == 0:
        return (1.0 + 0j, 0.0) if estimate_error else 1.0 + 0j
    with Solver1D(lens, y, w, w, **kw) as s:
        F = s(w)
    if not estimate_error:
        return F
    with Solver1D(lens, y, w, w, **_err_kw(s, kw)) as s2:
        return F, abs(s2(w) - F)


def amplification_many(ws, y, lens, estimate_error=False, **kw):
    """

    F on an array of frequencies for one source position (one solver, batched non-uniform FFTs)
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
    (May be we should call it amplification_many_y?)
    F at one frequency for many source positions ys (any shape).
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


def amplification_grid(ws, ys, lens, **kw):
    """
    Table F[k, m] = F(ws[m], ys[k]) for many frequencies and source positions.
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

