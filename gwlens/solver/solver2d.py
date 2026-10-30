import os
import signal
import weakref
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager, nullcontext
from functools import partial
import numpy as np
from scipy.special import roots_legendre, roots_laguerre, expit
try:
    import finufft                      
except ImportError:                    
    finufft = None
try:
    import numexpr as ne              
except ImportError:
    ne = None
try:
    from threadpoolctl import ThreadpoolController  
except ImportError:
    ThreadpoolController = None

_GL, _LAG = {}, {}



_IMPORT_PID = os.getpid()
_NE_SET = {}
_WORKER = {}
_TPC = {}
_PTS_PER_THREAD = 250_000         


def available_cores():
    """Number of CPU cores this process may use."""
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


def _fork_executor(n, initializer, initargs):
    return ProcessPoolExecutor(n, mp_context=mp.get_context("fork"), initializer=initializer, initargs=initargs)


def _set_ne_threads(n):
    if ne is not None and _NE_SET.get(os.getpid()) != n:
        ne.set_num_threads(n)
        _NE_SET[os.getpid()] = n


def _blas_single():
    """
    Context in which BLAS runs on one thread. The matrix products of a spectrum or a map are small;
    a multithreaded BLAS between FINUFFT calls only competes with FINUFFT's own threads.
    """
    if ThreadpoolController is None:
        return nullcontext()
    pid = os.getpid()
    ctl = _TPC.get(pid)
    if ctl is None:
        try:
            ctl = _TPC[pid] = ThreadpoolController()
        except Exception:
            return nullcontext()
    return ctl.limit(limits=1, user_api="blas")


def _worker_common():
    signal.signal(signal.SIGINT, signal.SIG_IGN)          # Ctrl-C is handled by the parent
    _set_ne_threads(1)


def _thread_map(fn, n, nthreads):
    """[fn(0), ..., fn(n-1)] on up to nthreads threads (order kept). fn must release the GIL for most
    of its work (numpy elementwise operations on large arrays do)."""
    if nthreads <= 1 or n <= 1:
        return [fn(i) for i in range(n)]
    with ThreadPoolExecutor(min(nthreads, n)) as ex:
        return list(ex.map(fn, range(n)))


_THREAD_EFF = 0.5            # FINUFFT threads on one transform: ~half the cores' worth (assumed)


def _init_eval_worker(ref):
    _worker_common()
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
    """Worker: rows of Solver2D.evaluate. by_y: one spectrum per source position; else one map per w."""
    s = _WORKER["solver"]
    by_y, ws, ys = task
    if by_y:
        return np.array([s.spectrum(ws, y) for y in ys]).reshape(len(ys), ws.size)
    return np.array([s.map(w, ys[:, 0], ys[:, 1]) for w in ws]).reshape(ws.size, len(ys)).T


def _shutdown_pool(ex, pid):
    if os.getpid() == pid:
        ex.shutdown(wait=True, cancel_futures=True)


def _gl(n):
    if n not in _GL:
        _GL[n] = roots_legendre(n)
    return _GL[n]


def _lag(n):
    if n not in _LAG:
        _LAG[n] = roots_laguerre(n)
    return _LAG[n]


def _unique_points(pts, tol=1e-12):
    out = []
    for p in pts:
        p = (float(p[0]), float(p[1]))
        if not any(np.hypot(p[0] - q[0], p[1] - q[1]) < tol for q in out):
            out.append(p)
    return out



class Lens:
    """A lens potential psi(x1, x2) plus the list of its singular points.

    psi       callable psi(x1, x2) -> array; analytic operations only (see module docstring)
    singular  iterable of (x1, x2) points where psi is not smooth
    q_inf     kappa + |gamma| of external sheet / shear (None: estimate automatically)
    """

    def __init__(self, psi, singular=(), q_inf=None):
        self._terms = [psi]
        self.singular = _unique_points(singular)
        self.q_inf = q_inf
        self._checked = False

    def psi(self, x1, x2):
        out = self._terms[0](x1, x2)
        for f in self._terms[1:]:
            out = out + f(x1, x2)
        return out

    __call__ = psi

    def __add__(self, other):
        new = Lens.__new__(Lens)
        new._terms = self._terms + other._terms
        new.singular = _unique_points(self.singular + other.singular)
        new.q_inf = None if (self.q_inf is None or other.q_inf is None) else self.q_inf + other.q_inf
        new._checked = False
        return new

    def grad(self, x1, x2, h=1e-20):
        """(d psi/d x1, d psi/d x2) at real points by the complex step (exact to rounding)."""
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        g1 = np.imag(self.psi(x1 + 1j * h, x2 + 0j)) / h
        g2 = np.imag(self.psi(x1 + 0j, x2 + 1j * h)) / h
        return g1, g2

    def check_analytic(self, n=400, seed=0, verbose=False):
        """Compare complex-step and finite-difference gradients at random points.
        A mismatch means psi uses a non-analytic operation (abs, hypot, arctan2, where, ...),
        which would make the rotated tail wrong. Returns the worst relative mismatch."""
        rng = np.random.default_rng(seed)
        P = np.array(self.singular) if self.singular else np.zeros((1, 2))
        c = P.mean(axis=0)
        R = 2.0 + np.max(np.hypot(*(P - c).T))
        x = c + rng.uniform(-R, R, size=(n, 2))
        dmin = np.min(np.hypot(x[:, None, 0] - P[None, :, 0], x[:, None, 1] - P[None, :, 1]), axis=1)
        x = x[dmin > 0.05]
        with np.errstate(all="ignore"):
            try:
                g1, g2 = self.grad(x[:, 0], x[:, 1])
            except TypeError as err:
                raise TypeError("psi(x1, x2) must accept complex arrays (it is evaluated on the "
                                f"rotated tail); use analytic numpy operations only. ({err})") from None
            h = 1e-6
            f = lambda a, b: np.real(self.psi(a + 0j, b + 0j))
            d1 = (f(x[:, 0] + h, x[:, 1]) - f(x[:, 0] - h, x[:, 1])) / (2 * h)
            d2 = (f(x[:, 0], x[:, 1] + h) - f(x[:, 0], x[:, 1] - h)) / (2 * h)
            im = np.imag(self.psi(x[:, 0] + 0j, x[:, 1] + 0j))
            re = np.real(self.psi(x[:, 0] + 0j, x[:, 1] + 0j))
        mism = np.max(np.hypot(g1 - d1, g2 - d2) / (1 + np.hypot(d1, d2)))
        imag = np.max(np.abs(im) / (1 + np.abs(re)))
        bad = (not np.isfinite(mism)) or mism > 1e-5 or imag > 1e-10
        if bad:
            warnings.warn(
                f"psi does not look analytic (gradient mismatch {mism:.1e}, imaginary part on the "
                f"real plane {imag:.1e}). Use only analytic numpy operations (no abs/hypot/"
                "arctan2/where/real/if-else on x); otherwise the rotated tail is invalid.")
        if verbose:
            print(f"check_analytic: gradient mismatch {mism:.1e}, Im psi on real plane {imag:.1e}")
        self._checked = True
        return mism



def point_mass(m=1.0, x0=(0.0, 0.0)):
    """psi = m ln|x - x0|."""
    a, b, m = float(x0[0]), float(x0[1]), float(m)
    return Lens(lambda x1, x2: 0.5 * m * np.log((x1 - a)**2 + (x2 - b)**2), singular=[(a, b)], q_inf=0.0)


def sis(psi0=1.0, x0=(0.0, 0.0)):
    """psi = psi0 |x - x0|."""
    a, b, p = float(x0[0]), float(x0[1]), float(psi0)
    return Lens(lambda x1, x2: p * np.sqrt((x1 - a)**2 + (x2 - b)**2), singular=[(a, b)], q_inf=0.0)


def cis(psi0=1.0, rc=0.05, x0=(0.0, 0.0)):
    """
    Cored isothermal sphere (GLoW convention): psi0 [s + rc ln(2 rc/(s + rc))], s = sqrt(R^2 + rc^2).
    The centre is listed as a singular point so that a small core is resolved like a cusp.
    """
    a, b, p, rc = float(x0[0]), float(x0[1]), float(psi0), float(rc)

    def f(x1, x2):
        R2 = (x1 - a)**2 + (x2 - b)**2
        if rc == 0:
            return p * np.sqrt(R2)
        s = np.sqrt(R2 + rc * rc)
        return p * (s + rc * np.log(2 * rc / (s + rc)))
    return Lens(f, singular=[(a, b)], q_inf=0.0)


def nfw(k=1.0, psi0=1.0, x0=(0.0, 0.0)):
    """NFW: 0.5 psi0 [ln^2(v/2) + arccos^2(1/v)], v = k |x - x0|  (k = xi0 / r_s)."""
    a, b, p, k = float(x0[0]), float(x0[1]), float(psi0), float(k)

    def f(x1, x2):
        v = k * np.sqrt((x1 - a)**2 + (x2 - b)**2 + 0j)       # complex: arccos(1/v) for v < 1
        return 0.5 * p * (np.log(v / 2)**2 + np.arccos(1 / v)**2)
    return Lens(f, singular=[(a, b)], q_inf=0.0)


def external_shear(gamma1=0.0, gamma2=0.0, kappa=0.0):
    """psi = kappa |x|^2/2 + [gamma1 (x1^2 - x2^2) + 2 gamma2 x1 x2]/2."""
    g1, g2, k = float(gamma1), float(gamma2), float(kappa)
    f = lambda x1, x2: 0.5 * (k * (x1 * x1 + x2 * x2) + g1 * (x1 * x1 - x2 * x2) + 2 * g2 * x1 * x2)
    return Lens(f, singular=[], q_inf=k + np.hypot(g1, g2))


# Planck Taper window
def _step(s, s0, b=2.5):
    """
    C-infinity step: 1 for s <= s0, 0 for s >= 1.
    Planck taper 1/(1 + exp(b [1/(1-t) - 1/t])), t = (s - s0)/(1 - s0); b = 2.5 minimises the
    number of Fourier modes (hence trapezoid angles) needed to resolve it to ~1e-12.

    From Sathyaprakash's paper, not using this taper introduced some edge artifacts.
    """
    t = np.clip((np.asarray(s, dtype=float) - s0) / (1.0 - s0), 0.0, 1.0)
    inner = (t > 0) & (t < 1)
    tt = np.where(inner, t, 0.5)
    z = b * (1.0 / (1.0 - tt) - 1.0 / tt)
    return np.where(t <= 0, 1.0, np.where(t >= 1, 0.0, expit(-z)))


def _estimate_q_inf(lens, c):
    th = np.linspace(0, 2 * np.pi, 257)[:-1]
    q = -np.inf
    for R in (1e3, 1e4):
        g1, g2 = lens.grad(c[0] + R * np.cos(th), c[1] + R * np.sin(th))
        q = max(q, np.max((np.cos(th) * g1 + np.sin(th) * g2) / R))
    return float(q)


def _split_radius(lens, c, yc0, Re, thr, xb):
    """Smallest x_b (from the given start, steps of 10 %) with (n.(y - c) + d psi/dr)/x_b <= thr in every
    direction for every source within Re of y; also returns the smallest decay rate 1 - P there."""
    th = np.linspace(0, 2 * np.pi, 257)[:-1]
    for _ in range(400):
        g1, g2 = lens.grad(c[0] + xb * np.cos(th), c[1] + xb * np.sin(th))
        P = np.max((np.cos(th) * (yc0[0] + g1) + np.sin(th) * (yc0[1] + g2) + Re) / xb)
        if P <= thr:
            return float(xb), float(1 - P)
        xb *= 1.1
    raise RuntimeError("could not find a split radius; is the potential growing faster than r^2/2?")


def _region_cost(lens, c, Y, r_out, wmax, ngl, cycles, bw_factor, bw_margin, u_floor, weight=None,
                 ang_req=None, n_oct=12, n_ang=32):
    """
    Rough number of head nodes of a polar region of radius r_out about c (the same rules as
    _plan_nodes, with psi sampled on one ring per octave). Only used to compare candidate centres.
    """
    r_e = r_out * 2.0 ** -np.arange(n_oct, -1, -1)
    th = (np.arange(n_ang) + 0.5) * 2 * np.pi / n_ang
    R, TH = np.meshgrid(r_e, th, indexing="ij")
    x1, x2 = c[0] + R * np.cos(TH), c[1] + R * np.sin(TH)
    with np.errstate(all="ignore"):
        g1, g2 = lens.grad(x1, x2)
        G = np.hypot(g1, g2)
        D = np.abs(R * (-np.sin(TH) * g1 + np.cos(TH) * g2))
        ok = np.isfinite(G) & np.isfinite(D)
        if weight is not None:
            ok &= weight(x1, x2) != 0
    G = np.max(np.where(ok, G, 0.0), axis=1)
    D = np.max(np.where(ok, D, 0.0), axis=1)
    per_piece = 2 * np.pi * cycles / wmax                      # phase per unit w in one radial piece
    n = ngl * np.log(max(0.5 * r_e[0] ** 2 / u_floor, 1.0)) * (bw_factor * wmax * (r_e[0] * Y + D[0]) + bw_margin)
    for k in range(1, r_e.size):
        r_lo, r_hi = r_e[k - 1], r_e[k]
        Gk, Dk = max(G[k - 1], G[k]), max(D[k - 1], D[k])
        dV = 0.5 * (r_hi ** 2 - r_lo ** 2) + (Y + Gk) * (r_hi - r_lo)
        N = bw_factor * wmax * (r_hi * Y + Dk) + bw_margin + (ang_req(r_lo, r_hi) if ang_req else 0.0)
        n += ngl * (np.ceil(dV / per_piece) + 1) * N
    return float(n)



# Resolution of the two planning probes (they only size the grid; psi on the grid is exact). The
# previous values (64 directions, >= 1500 + 3000 radii; 256 x 8 points per octave) cost 30-50 % of a
# build with PE_SETTINGS and gave the same grids to within a few nodes.
_PROBE_N_ANG, _PROBE_N_LIN, _PROBE_N_GEO = 48, 600, 1000
_DPSI_N_ANG, _DPSI_N_R = 128, 4


def _probe_V1(lens, c, yc, u_floor, u_lin, u_hi, wmax, weight=None, n_ang=_PROBE_N_ANG):
    """Worst-direction accumulated phase per unit w, V1(u), on a probe grid in [u_floor, u_hi].
    Dense (>= 3 points per cycle at wmax) up to u_lin, geometric everywhere."""
    n_lin = int(max(_PROBE_N_LIN, 3 * wmax * u_lin / (2 * np.pi)))
    ug = np.unique(np.concatenate([np.linspace(u_floor, u_lin, n_lin),
                                   np.geomspace(u_floor, u_hi, _PROBE_N_GEO)]))
    xg = np.sqrt(2 * ug)
    th = (np.arange(n_ang) + 0.5) * 2 * np.pi / n_ang
    dV = np.zeros(ug.size - 1)
    step = max(1, int(1_000_000 // ug.size))
    with np.errstate(all="ignore"):
        for s in range(0, n_ang, step):
            t = th[s:s + step, None]
            x1, x2 = c[0] + xg[None, :] * np.cos(t), c[1] + xg[None, :] * np.sin(t)
            ph = ug[None, :] - xg[None, :] * (yc[0] * np.cos(t) + yc[1] * np.sin(t)) - np.real(lens.psi(x1, x2))
            d = np.abs(np.diff(ph, axis=1))
            ok = np.isfinite(d)
            if weight is not None:
                m = weight(x1, x2) != 0
                ok &= m[:, 1:] & m[:, :-1]
            dV = np.maximum(dV, np.max(np.where(ok, d, 0.0), axis=0))
    return ug, np.concatenate([[0.0], np.cumsum(dV)])


def _max_dpsi_dth(lens, c, r_lo, r_hi, weight=None, n_ang=_DPSI_N_ANG, n_r=_DPSI_N_R):
    """max over the annulus of |d psi / d theta| (angular derivative about c), masked by weight != 0."""
    rs = np.linspace(max(r_lo, 1e-6 * r_hi), r_hi, n_r)
    R, T = np.meshgrid(rs, (np.arange(n_ang) + 0.5) * 2 * np.pi / n_ang)
    x1, x2 = c[0] + R * np.cos(T), c[1] + R * np.sin(T)
    with np.errstate(all="ignore"):
        g1, g2 = lens.grad(x1, x2)
        d = np.abs(R * (-np.sin(T) * g1 + np.cos(T) * g2))
        ok = np.isfinite(d)
        if weight is not None:
            ok &= weight(x1, x2) != 0
    return float(np.max(np.where(ok, d, 0.0)))


def _even(n):
    return int(2 * np.ceil(n / 2))


def _cheb_points(n):
    k = np.arange(n)
    return np.cos(np.pi * (k + 0.5) / n)


def _bary_matrix(xc, xe):
    """Barycentric interpolation matrix from Chebyshev points (first kind) xc to points xe."""
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


_WARNED = set()


def _interp_mode(value, default):
    """The interpolate setting as False, True or 'auto'. Also accepts what configuration files and ini
    parsers pass on: the strings 'True' / 'False' / 'auto' / 'None' (any case or quoting), numpy booleans
    and 0 / 1. None gives the solver's default."""
    if value is None:
        return default
    if isinstance(value, str):
        v = value.strip().strip("'\"").lower()
        table = {"true": True, "false": False, "auto": "auto", "none": default, "1": True, "0": False}
        if v in table:
            return table[v]
    elif isinstance(value, (bool, np.bool_, int, np.integer)) and value in (0, 1):
        return bool(value)
    raise ValueError(f"interpolate must be False, True or 'auto', not {value!r}")


def _time_origin_mode(value):
    """time_origin as None, 'first_image' or a float (strings from configuration files accepted)."""
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().strip("'\"").lower().replace("-", "_")
        if v in ("none", "", "0"):
            return None
        if v in ("first_image", "first", "minimum", "min"):
            return "first_image"
        try:
            return float(v)
        except ValueError:
            raise ValueError(f"time_origin must be None, 'first_image' or a number, not {value!r}") from None
    return float(value)


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(msg)



def _interp_panels(w_lo, w_hi, B):
    """
    Panel edges on [w_lo, w_hi]: phase B*(width) <= 10 rad, width <= left end (geometric near w = 0).
    """
    edges, a = [w_lo], w_lo
    while a < w_hi * (1 - 1e-14):
        h = min(10.0 / B if B > 0 else np.inf, a, w_hi - a)
        a = a + h if w_hi - (a + h) > 1e-3 * h else w_hi
        edges.append(a)
    return np.array(edges)


def _interp_eval(w, edges, G, xc):
    """
    Evaluate the panel-wise Chebyshev interpolant (values G[panel, node] at the points xc) at w:
    Chebyshev coefficients per panel, then T_k(t) by recurrence on the sorted targets of each panel.
    """
    n = xc.size
    Tm = np.cos(np.outer(np.arange(n), np.pi * (np.arange(n) + 0.5) / n))
    coef = (2.0 / n) * G @ Tm.T
    coef[:, 0] *= 0.5
    cr, ci = np.ascontiguousarray(coef.real.T), np.ascontiguousarray(coef.imag.T)
    order = None if np.all(w[1:] >= w[:-1]) else np.argsort(w, kind="stable")
    ws = w if order is None else w[order]
    bounds = np.concatenate([[0], np.searchsorted(ws, edges[1:-1], side="right"), [ws.size]])
    res = np.empty(ws.size, dtype=complex)
    for p in range(edges.size - 1):
        i0, i1 = bounds[p], bounds[p + 1]
        if i1 <= i0:
            continue
        t = (2 * ws[i0:i1] - (edges[p] + edges[p + 1])) / (edges[p + 1] - edges[p])
        V = np.empty((n, i1 - i0))
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



class Solver2D:
    """
    lens          a Lens (see module docstring or the documentation on how to use this)
    y             centre of the source region (with y_radius = 0: the source position)
    wmin, wmax    frequency range the grid must resolve
    y_radius      radius of the source region (0: a single source position)

    Evaluation (all reuse the same grid; psi is never evaluated again for the head):

      s(w, y=None)            F at one frequency, one source position (default: the centre)
      s.spectrum(ws, y=None)  many frequencies, one source position     -> 1D non-uniform FFT
      s.map(w, Y1, Y2)        one frequency, many source positions       -> 2D non-uniform FFT
      s.evaluate(ws, ys)      table (len(ys), len(ws)); loops over whichever axis is shorter

    center        centre of the polar coordinates. 'auto' (default): the cheapest of the singular points
                  and their centroid, judged by a quick estimate of the grid size (often the centroid
                  for several similar lenses: |y - c| and the split radius are then small, giving
                  2-3x fewer nodes). 'nearest': the singular point nearest to y (the old default).
                  Or a point (x1, x2). info['center_costs'] lists the estimates.

    Accuracy/tuning options (defaults are the validated ones): patch_frac, patch_max, s0,
    cycles, ngl, nlag, n_cheb, theta_factor (2 = convergence test), bw_factor, bw_margin,
    xb_min (move split out = convergence test), xb_base, n_transition, pou_density,
    map_ext (grid built for radius map_ext*y_radius so maps are valid up to y_radius; use 1.0 if
    you never call map(), e.g. in parameter estimation - the grid is then smaller),
    nufft_eps (FINUFFT tolerance; default 1e-14 gives ~1e-12 in F),
    low_band_ratio (width of the geometric frequency bands below w = 20/u_b).

    Parallelliszing 
      n_workers   1 (default), k, or None (all cores). Building the solver: psi is evaluated on the
                  grid by up to n_workers THREADS (worth it for large grids or expensive potentials).
                  evaluate(ws, ys): rows are computed by up to n_workers forked PROCESSES when there
                  are >= 2 rows per process; the pool is started on first use and kept - call close()
                  or use `with Solver2D(...) as s:`. Processes need the 'fork' start method (Linux,
                  macOS); elsewhere, and inside daemonic pool workers, rows run serially.
      threads     threads of ONE transform (a spectrum or a map). 1 (default): one thread - the
                  fastest choice for grids up to ~10^6 nodes and the right one inside a sampler
                  that runs several likelihoods at once (bilby npool). k: FINUFFT may use up to k
                  threads, one per 250 000 points of a transform (smaller transforms stay on one).
                  None / 'auto': the same with k = all available cores. BLAS always runs on one
                  thread inside a transform (needs threadpoolctl; pip install threadpoolctl), and
                  forked or pool-worker processes always use one thread.
    To use many cores, run many independent calls (source positions, likelihood evaluations) in
    parallel, each on one thread, rather than one call on many threads.

    Interpolation in w (for very many frequencies)
      interpolate  False (default): every frequency is computed. True: F is computed on Chebyshev
                   panels sized by the time delays of the images and interpolated to the requested
                   frequencies. 'auto': only when clearly cheaper. Per call: s.spectrum(ws, y,
                   interpolate=True). A spectrum here costs a fixed ~0.1-1 s (spreading the grid
                   nodes) plus ~1-2 us per frequency; interpolation removes the second part, so it
                   pays from ~10^4-10^5 frequencies on.
      interp_tol   checked every call against exact values at a few frequencies (computed in the
                   same transform); panels are refined if the relative error exceeds it.

    Time origin (the phase of F)
      time_origin  None (default): T = |x - y|^2/2 - psi(x) as it stands, no time shift.
                   'first_image': time measured from the arrival of the first image, i.e. F is multiplied
                   by exp(-i w T_min(y)), T_min = first_image_delay(y) (the global minimum of T; GLoW's
                   t_min). A number t0: F is multiplied by exp(-i w t0). Only the phase changes, by a
                   term linear in w (a constant time shift).
    """

    def __init__(self, lens, y, wmin, wmax, y_radius=0.0, center="auto", patch_frac=0.8, patch_max=1.0,
                 s0=0.3, cycles=8, ngl=32, nlag=30, n_cheb=None, theta_factor=1.0, bw_factor=1.3,
                 bw_margin=32, xb_min=0.0, xb_base=2.0, n_transition=3, pou_density=16.0, map_ext=1.5,
                 nufft_eps=1e-14, low_band_ratio=4.0, max_chunk=2_000_000, tail_mode=None,
                 n_workers=1, threads=1, task_nodes=262_144, min_parallel_work=400_000,
                 interpolate=False, interp_tol=1e-10, time_origin="first_image"):
        """See the class docstring formore details"""
        if not lens._checked:
            lens.check_analytic()
        self.interpolate, self.interp_tol = _interp_mode(interpolate, False), float(interp_tol)
        self._tdgrid, self._tdcache = None, {}
        self.time_origin = _time_origin_mode(time_origin)
        self._tmincache, self._tmingrid = {}, None
        self.n_workers = available_cores() if (n_workers is None or n_workers <= 0) else int(n_workers)
        if threads is None or threads == "auto":
            self.threads = "auto"
        else:
            self.threads = max(1, int(threads))
        self.task_nodes, self.min_parallel_work = int(task_nodes), int(min_parallel_work)
        self._pool = self._pool_fin = self._pool_pid = None
        self.lens = lens
        self.y0 = np.asarray(y, dtype=float)
        self.Yr = float(y_radius)
        self.ext = float(map_ext)
        self.Re = self.ext * self.Yr
        self.wmin, self.wmax = float(wmin), float(wmax)
        if not (0 < self.wmin <= self.wmax):
            raise ValueError("need 0 < wmin <= wmax")
        self.cycles, self.ngl, self.nlag = cycles, ngl, nlag
        self.theta_factor, self.bw_factor, self.bw_margin = theta_factor, bw_factor, bw_margin
        self.nufft_eps, self.max_chunk = nufft_eps, max_chunk
        self.u_floor = 1e-15 / max(self.wmax, 1.0)
        self.LX, self.LW = _lag(nlag)
        S = np.array(lens.singular, dtype=float).reshape(-1, 2)

        # ---- patch radii (independent of the centre): each patch holds one singular point
        rho_of = []
        for k, p in enumerate(S):
            others = [np.hypot(*(p - q)) for j, q in enumerate(S) if j != k]
            rho_of.append(min(patch_max, patch_frac * min(others)) if others else patch_max)

        # ---- split threshold (independent of the centre)
        if lens.q_inf is not None:
            q = lens.q_inf
        else:
            q = _estimate_q_inf(lens, S.mean(axis=0) if len(S) else np.zeros(2))
        if q >= 1:
            raise ValueError(f"kappa + |gamma| = {q:.3f} >= 1: the tail cannot be rotated upward in every direction.")
        thr = max(0.5, 0.5 * (1 + q))

        def layout(c):
            """Patches, split radius and smallest decay rate for polar centre c."""
            patches = [(p, r) for p, r in zip(S, rho_of) if np.hypot(*(p - c)) >= 1e-12]
            yc0 = self.y0 - c
            Ymax_c = float(np.hypot(*yc0)) + self.Re
            reach = max([np.hypot(*(p - c)) + r for p, r in patches], default=0.0)
            xb, al = _split_radius(lens, c, yc0, self.Re, thr, max(float(xb_base), 2 * Ymax_c, 1.25 * reach,
                                                                   float(xb_min)))
            return patches, yc0, Ymax_c, xb, al

        def make_ang_req(dists):
            def ang_req(r_lo, r_hi):
                n = 0.0
                for d, rho in dists:
                    if d - rho < r_hi and d + rho > r_lo:
                        n = max(n, pou_density * 2 * np.pi * min(r_hi, d + rho) / ((1 - s0) * rho))
                return n
            return ang_req

        def make_chi0(patches):
            if not patches:
                return None

            def chi0(x1, x2):
                x1, x2 = np.broadcast_arrays(x1, x2)
                out = np.ones(x1.shape)
                for p, rho in patches:                    # each bump only where it is non-zero
                    d2 = (x1 - p[0])**2 + (x2 - p[1])**2
                    m = d2 < rho * rho
                    if m.any():
                        out[m] -= _step(np.sqrt(d2[m]) / rho, s0)
                return out
            return chi0

        # ---- centre: a given point, the singular point nearest to y, or the cheapest candidate
        costs = None
        if isinstance(center, str) and center == "nearest":
            c = S[np.argmin(np.hypot(*(S - self.y0).T))] if len(S) else np.zeros(2)
        elif center is None or (isinstance(center, str) and center == "auto"):
            cands = [p for p in S]
            if len(S) > 1:
                cands.append(S.mean(axis=0))
            if not cands:
                cands = [np.zeros(2)]
            uniq = []
            for cc in cands:
                if not any(np.hypot(*(cc - u)) < 1e-9 for u in uniq):
                    uniq.append(np.asarray(cc, dtype=float))
            if len(uniq) == 1:
                c = uniq[0]
            else:
                costs, patch_cost = [], {}
                for cc in uniq:
                    costs.append((self._center_cost(cc, layout(cc), S, rho_of, patch_cost, make_chi0, make_ang_req,
                                                    s0), tuple(cc)))
                c = np.array(min(costs)[1])
        else:
            c = np.asarray(center, dtype=float)
        self.c = c
        patches, self.yc0, self.Ymax_c, xb, _ = layout(c)
        self.patches = patches
        chi0 = make_chi0(patches)
        self.q_inf, self.xb0, self.ub0 = q, xb, xb * xb / 2
        self.w_switch = 20 / self.ub0
        self.ub_max = max(self.ub0, 20 / self.wmin)

        extra_r, dists = [], []
        for p, rho in patches:
            d = np.hypot(*(p - c))
            n = int(np.ceil(2 * rho / ((1 - s0) * rho / n_transition)))
            extra_r.append(np.linspace(max(0.0, d - rho), d + rho, n + 1))
            dists.append((d, rho))
        ang_req = make_ang_req(dists)

        def touches(r_lo, r_hi):
            return any(d - rho < r_hi and d + rho > r_lo for d, rho in dists)

        # ---- plan the head nodes (y- and w-independent): background (k = 0), then patches (k >= 1).
        # Each region: probe of the worst-direction phase variation V1(u) (+ Re dr for the y-dependent
        # part), then radial nodes and angular counts. Regions are independent -> parallel tasks.
        def plan_region(k):
            if k == 0:
                ug, V1 = _probe_V1(lens, c, self.yc0, self.u_floor, self.ub0, self.ub_max, self.wmax, weight=chi0)
                V1 = V1 + self.Re * (np.sqrt(2 * ug) - np.sqrt(2 * ug[0]))
                blocks, meta = self._plan_nodes(c, self.Ymax_c, 0.0, self.ub0, ug, V1, self.wmax, weight=chi0,
                                                touches=touches, extra_r=np.concatenate(extra_r) if extra_r else (),
                                                ang_req=ang_req if patches else None)
            else:
                p, rho = patches[k - 1]
                ug, V1 = _probe_V1(lens, p, self.y0 - p, self.u_floor, rho * rho / 2, rho * rho / 2, self.wmax)
                V1 = V1 + self.Re * (np.sqrt(2 * ug) - np.sqrt(2 * ug[0]))
                blocks, meta = self._plan_nodes(p, float(np.hypot(*(self.y0 - p))) + self.Re, 0.0, rho * rho / 2,
                                                ug, V1, self.wmax, radial_weight=lambda r: _step(r / rho, s0),
                                                extra_r=np.linspace(s0 * rho, rho, n_transition + 1))
            return (ug, V1) if k == 0 else None, blocks, meta

        n_reg = 1 + len(patches)
        plans = [plan_region(k) for k in range(n_reg)]       # cheap; many small numpy calls (threads do not pay)
        self.ug_bg, self.V1_bg = plans[0][0]
        plans = [(blocks, meta) for _, blocks, meta in plans]
        regions = [dict(kind="background", center=tuple(c), **plans[0][1])]
        regions += [dict(kind="patch", center=tuple(p), rho=rho, **P[1]) for (p, rho), P in zip(patches, plans[1:])]
        self.D_outer = plans[0][1]["D_outer"]

        # ---- plan the frequency bands: main (u_b = ub0) and geometric low bands (u_b = 20 / w_low)
        self.bands = []
        if self.wmax >= self.w_switch:
            self.bands.append(self._plan_band(max(self.wmin, self.w_switch), self.wmax, self.ub0, n_cheb))
        if self.wmin < self.w_switch:
            hi = min(self.w_switch, self.wmax)
            while True:
                lo = max(self.wmin, hi / low_band_ratio)
                self.bands.append(self._plan_band(lo, hi, 20 / lo, n_cheb))
                if lo <= self.wmin * (1 + 1e-12):
                    break
                hi = lo

        # ---- evaluate psi on every planned node and tail point (the expensive part; threads if allowed)
        groups = [[blk for P in plans for blk in P[0]]] + [b.pop("seg_blocks") for b in self.bands]
        filled, nthr = self._fill(groups, self.bands)
        self.hx1, self.hx2, self.hA, self.hW = filled[0]
        for b, seg in zip(self.bands, filled[1:]):
            b["seg"] = seg
        self._tcache = {}
        self.info = dict(center=tuple(c), x_b=self.xb0, q_inf=q, w_switch=self.w_switch, y_center=tuple(self.y0),
                         y_radius=self.Yr, n_head_nodes=int(self.hW.size), regions=regions,
                         bands=[(b["w_lo"], b["w_hi"], b["ub"], b["th"].size, b["seg"][0].size) for b in self.bands],
                         n_tail_points=int(sum(b["th"].size * b["eps"].size for b in self.bands) * self.nlag),
                         setup_threads=nthr,
                         center_costs=None if costs is None else [(cc, float(v)) for v, cc in costs])

    # choice of the polar centre
    def _center_cost(self, c, lay, S, rho_of, patch_cost, make_chi0, make_ang_req, s0):
        """Estimated work (head nodes + 2 x main-band tail points) with polar centre c. The patch costs
        do not depend on c and are cached in patch_cost."""
        patches, yc0, Y, xb, al = lay
        kw = dict(wmax=self.wmax, ngl=self.ngl, cycles=self.cycles, bw_factor=self.bw_factor,
                  bw_margin=self.bw_margin, u_floor=self.u_floor)
        dists = [(np.hypot(*(p - c)), rho) for p, rho in patches]
        n = _region_cost(self.lens, c, Y, xb, weight=make_chi0(patches),
                         ang_req=make_ang_req(dists) if patches else None, **kw)
        for p, rho in patches:
            key = (float(p[0]), float(p[1]))
            if key not in patch_cost:
                patch_cost[key] = _region_cost(self.lens, p, float(np.hypot(*(self.y0 - p))) + self.Re, rho, **kw)
            n += patch_cost[key]
        ub = 0.5 * xb * xb
        w_lo = max(self.wmin, min(self.wmax, 20 / ub))
        r_far = abs(np.sqrt(2 * (ub + 1j * 40 / (self.wmax * max(al, 1e-3)))))
        D = _max_dpsi_dth(self.lens, c, 0.5 * xb, max(xb, r_far), n_ang=64, n_r=3)
        n_dir = self.bw_factor * self.wmax * (r_far * Y + D) + self.bw_margin
        n_cheb = 1 if self.wmax <= w_lo * (1 + 1e-12) else int(np.clip(14 + np.sqrt(self.wmax / w_lo), 16, 32))
        return n + 2.0 * n_dir * n_cheb * self.nlag

    # nodes of one polar region: the plan
    def _plan_nodes(self, c, Yabs, u_lo, u_hi, ug, V1, wtop, weight=None, touches=None, radial_weight=None,
                    extra_r=(), ang_req=None):
        """Radial nodes and angular counts of one polar region (psi is not evaluated on the grid here).
        Returns blocks [(c, r, w_r, N, weight or None)], one per radial octave, and a summary."""
        lens, K = self.lens, self.cycles
        rmax, rmin = np.sqrt(2 * u_hi), np.sqrt(2 * u_lo)
        V_lo, V_hi = np.interp([u_lo, u_hi], ug, V1)
        cuts = np.interp(np.arange(V_lo, V_hi, 2 * np.pi * K / wtop), V1, ug)
        if u_lo == 0:
            geo = np.geomspace(self.u_floor, u_hi, int(np.ceil(np.log(u_hi / self.u_floor))) + 1)
        else:
            geo = np.geomspace(u_lo, u_hi, int(np.ceil(np.log(u_hi / u_lo))) + 2)
        rb = rmax * 2.0 ** -np.arange(12, 0, -1)
        rb = rb[rb > rmin]
        extra_u = np.asarray(extra_r, dtype=float) ** 2 / 2
        edges = np.concatenate([[u_lo, u_hi], geo, cuts, np.linspace(u_lo, u_hi, 5), rb**2 / 2, extra_u])
        edges = np.unique(edges[(edges >= u_lo) & (edges <= u_hi)])
        GX, GW = _gl(self.ngl)
        mid, half = (edges[1:] + edges[:-1]) / 2, (edges[1:] - edges[:-1]) / 2
        uu = (mid[:, None] + half[:, None] * GX[None, :]).ravel()
        wu = (half[:, None] * GW[None, :]).ravel()
        xu = np.sqrt(2 * uu)
        if radial_weight is not None:
            wu = wu * radial_weight(xu)
            keep = wu != 0
            uu, wu, xu = uu[keep], wu[keep], xu[keep]
        band = np.searchsorted(rb, xu)
        r_lo_all, r_hi_all = np.concatenate([[rmin], rb]), np.concatenate([rb, [rmax]])
        blocks, D_outer, bands = [], 0.0, []
        for b in np.unique(band):
            sel = band == b
            wb_n, xb_n = wu[sel], xu[sel]
            r_lo, r_hi = r_lo_all[b], r_hi_all[b]
            D = _max_dpsi_dth(lens, c, r_lo, r_hi, weight)
            if b == band.max():
                D_outer = D
            n = self.bw_factor * wtop * (r_hi * Yabs + D) + self.bw_margin
            if ang_req is not None:
                n += ang_req(r_lo, r_hi)
            N = _even(n * self.theta_factor)
            bands.append((float(r_lo), float(r_hi), int(xb_n.size), N))
            use_w = weight is not None and (touches is None or touches(r_lo, r_hi))
            blocks.append((np.asarray(c, dtype=float), xb_n, wb_n, N, weight if use_w else None))
        return blocks, dict(D_outer=D_outer, bands=bands)

    # psi on the planned nodes
    def _psi_job(self, X1, X2, A, s, e):
        """A[s:e] = |x|^2/2 - psi(x) on the head/segment nodes s..e-1."""
        with np.errstate(all="ignore"):
            x1, x2 = X1[s:e], X2[s:e]
            A[s:e] = 0.5 * (x1 * x1 + x2 * x2) - np.real(self.lens.psi(x1, x2))

    def _weight_job(self, fn, X1, X2, W, idx):
        """Multiply the weights of the nodes idx by the partition-of-unity function fn."""
        with np.errstate(all="ignore"):
            W[idx] = W[idx] * fn(X1[idx], X2[idx])

    def _tail_psi_job(self, b, X1, X2, s, e):
        """Complex psi on the rotated-tail points s..e-1 of band b (flattened directions x eps x Laguerre)."""
        per_dir = b["eps"].size * self.nlag
        x1, x2 = X1[s:e], X2[s:e]
        b["dA"].reshape(-1)[s:e] = (0.5 * (x1 * x1 + x2 * x2) - self.lens.psi(x1, x2)
                                    - b["A0"][np.arange(s, e) // per_dir])

    def _pmax(self, n_tasks):
        """Most processes worth considering for n_tasks independent tasks (1 = run here, serially)."""
        n = min(self.n_workers, available_cores(), int(n_tasks))
        if n <= 1:
            return 1
        if not _can_fork():
            if not mp.current_process().daemon:
                _warn_once("nofork", "n_workers > 1 needs the 'fork' start method (Linux/macOS); running serially.")
            return 1
        return n

    def _build_threads(self, work):
        """Threads for evaluating psi while the solver is built (1 unless n_workers > 1 and the grid is large)."""
        if work < self.min_parallel_work or mp.current_process().daemon:
            return 1
        return max(1, min(self.n_workers, available_cores(), int(work // 100_000)))

    def _fill(self, groups, bands):
        """Evaluate psi on all planned nodes (groups: head, then one segment per band) and on the tail
        points of every band. The node coordinates of all blocks are laid out first; psi and the
        partition-of-unity weight are then evaluated on chunks of ~task_nodes points (few large numpy
        calls instead of many small ones), by up to n_workers threads. The chunks write into disjoint
        slices, so threaded and serial runs give identical grids."""
        sizes = [sum(N * r.size for _, r, _, N, _ in blocks) for blocks in groups]
        n_tail = [b["th"].size * b["eps"].size * self.nlag for b in bands]
        work = sum(sizes) + 3 * sum(n_tail)                     # complex psi ~ 3x
        nthr = self._build_threads(work)
        chunk = int(np.clip(work // (4 * nthr), 32_768, max(32_768, min(self.max_chunk, self.task_nodes))))
        jobs, outs = [], []
        for blocks, n_tot in zip(groups, sizes):
            X1, X2, A, W = (np.empty(n_tot) for _ in range(4))
            weighted, off = {}, 0
            for c, r, wr, N, wgt in blocks:
                t = 2 * np.pi * np.arange(N) / N
                n = N * r.size
                X1[off:off + n] = (c[0] + np.cos(t)[:, None] * r[None, :]).ravel()
                X2[off:off + n] = (c[1] + np.sin(t)[:, None] * r[None, :]).ravel()
                W[off:off + n] = np.broadcast_to(wr[None, :] * (2 * np.pi / N), (N, r.size)).ravel()
                if wgt is not None:
                    weighted.setdefault(id(wgt), (wgt, []))[1].append(np.arange(off, off + n))
                off += n
            jobs += [partial(self._psi_job, X1, X2, A, s, min(n_tot, s + chunk)) for s in range(0, n_tot, chunk)]
            for fn, idx in weighted.values():
                idx = np.concatenate(idx)
                jobs += [partial(self._weight_job, fn, X1, X2, W, idx[s:s + chunk]) for s in range(0, idx.size, chunk)]
            outs.append((X1, X2, A, W))
        c = self.c
        for b, n in zip(bands, n_tail):
            shape = (b["th"].size, b["eps"].size, self.nlag)
            t = self.LX[None, None, :] * b["eps"][None, :, None] / b["al"][:, None, None]
            r = np.sqrt(2 * (b["ub"] + 1j * t))
            b["dr"] = r - b["xb"]
            b["dA"] = np.empty(shape, dtype=complex)
            X1 = (c[0] + r * b["ct"][:, None, None]).ravel()
            X2 = (c[1] + r * b["st"][:, None, None]).ravel()
            jobs += [partial(self._tail_psi_job, b, X1, X2, s, min(n, s + chunk)) for s in range(0, n, chunk)]
        _thread_map(lambda i: jobs[i](), len(jobs), nthr)
        filled = []
        for X1, X2, A, W in outs:                             # drop nodes outside the partition of unity
            m = W != 0
            filled.append((X1[m], X2[m], A[m], W[m]))
        return filled, nthr

    # ------------------------------------------------------------------ frequency band: segment + tail plan
    def _plan_band(self, w_lo, w_hi, ub, n_cheb):
        lens, c = self.lens, self.c
        xb = np.sqrt(2 * ub)
        if ub > self.ub0 * (1 + 1e-12):            # extra radial segment [ub0, ub] (beyond all patches)
            seg_blocks = self._plan_nodes(c, self.Ymax_c, self.ub0, ub, self.ug_bg, self.V1_bg, w_hi)[0]
        else:
            seg_blocks = []
        th0 = np.linspace(0, 2 * np.pi, 257)[:-1]
        g1, g2 = lens.grad(c[0] + xb * np.cos(th0), c[1] + xb * np.sin(th0))
        al0 = 1 - (np.cos(th0) * (self.yc0[0] + g1) + np.sin(th0) * (self.yc0[1] + g2) + self.Re) / xb
        # Directions of the tail sum: its angular bandwidth is at most w (|r| |y - c| + max |d psi/d theta|)
        # along the rotated contour up to ~40 e-folds, |r| <= r_far(w) = |sqrt(2 (u_b + 40 i/(w alpha)))|.
        # Both w r_far(w) and w max|d psi/d theta| grow with w, so the worst case is w = w_hi of the band
        # (previously r_far was taken at w_lo and max|d psi/d theta| over the radii of ALL bands, which
        # for an external shear, |d psi/d theta| ~ gamma r^2, gave 10-100x too many directions).
        r_far = abs(np.sqrt(2 * (ub + 1j * 40 / (w_hi * max(np.min(al0), 1e-3)))))
        D = _max_dpsi_dth(lens, c, 0.5 * xb, max(xb, r_far))
        if ub <= self.ub0 * (1 + 1e-12):
            D = max(D, self.D_outer)
        N = _even((self.bw_factor * w_hi * (r_far * self.Ymax_c + D) + self.bw_margin) * self.theta_factor)
        th = 2 * np.pi * np.arange(N) / N
        ct, st = np.cos(th), np.sin(th)
        g1, g2 = lens.grad(c[0] + xb * ct, c[1] + xb * st)
        al = 1 - (ct * (self.yc0[0] + g1) + st * (self.yc0[1] + g2) + self.Re) / xb
        if np.any(al <= 0):
            raise RuntimeError("tail does not decay in some direction; increase xb_min")
        x0 = np.stack([c[0] + xb * ct, c[1] + xb * st])
        A0 = 0.5 * (x0[0]**2 + x0[1]**2) - np.real(lens.psi(x0[0], x0[1]))
        e_lo, e_hi = 1 / w_hi, 1 / w_lo
        if e_hi - e_lo < 1e-12 * e_hi:
            eps = np.array([e_lo])
        else:
            nc = n_cheb or int(np.clip(14 + np.sqrt(w_hi / w_lo), 16, 32))
            eps = 0.5 * (e_hi + e_lo) + 0.5 * (e_hi - e_lo) * _cheb_points(nc)
        # the tables dA, dr (directions x eps x Laguerre nodes) are filled by _tail_job in _fill
        return dict(w_lo=w_lo, w_hi=w_hi, ub=ub, xb=xb, seg_blocks=seg_blocks, th=th, ct=ct, st=st, al=al, A0=A0,
                    x0=x0, eps=eps, e_lo=e_lo, e_hi=e_hi)

    def _band_index(self, wp):
        idx = np.full(wp.size, -1)
        for i, b in enumerate(self.bands):
            sel = (idx < 0) & (wp >= b["w_lo"] * (1 - 1e-12)) & (wp <= b["w_hi"] * (1 + 1e-12))
            idx[sel] = i
        if np.any(idx < 0):
            raise ValueError(f"frequencies must lie in [{self.wmin:g}, {self.wmax:g}] for this solver")
        return idx

    def _band_of(self, w):
        for i, b in enumerate(self.bands):
            if b["w_lo"] * (1 - 1e-12) <= w <= b["w_hi"] * (1 + 1e-12):
                return i
        raise ValueError(f"w = {w} outside [{self.wmin}, {self.wmax}] of this solver")

    def _as_y(self, y):
        return self.y0 if y is None else np.asarray(y, dtype=float)

    def _check_y(self, y1, y2):
        d = np.hypot(np.asarray(y1) - self.y0[0], np.asarray(y2) - self.y0[1])
        if np.any(d > max(self.Yr, 1e-12) * (1 + 1e-9) + 1e-12):
            raise ValueError(f"source position outside |y - ({self.y0[0]:g}, {self.y0[1]:g})| <= {self.Yr:g}; "
                             "build the solver with a larger y_radius")

    def _check_w(self, wp):
        if np.any(wp < self.wmin * (1 - 1e-12)) or np.any(wp > self.wmax * (1 + 1e-12)):
            raise ValueError(f"frequencies must lie in [{self.wmin:g}, {self.wmax:g}] for this solver")

    # ------------------------------------------------------------------ threads and processes
    def _nthreads(self, n_points=None):
        """Threads for one FINUFFT / numexpr call on n_points points (None: the most this solver may use).
        One thread per _PTS_PER_THREAD points, at most `threads` (all cores for 'auto'); always 1 in
        forked or pool-worker processes."""
        if _forked() or mp.current_process().daemon:
            if _forked() and self.threads != 1:  # OpenMP cannot start threads in a forked child of a process that used it
                _warn_once("forkthreads", "threads > 1 is ignored in a forked process (OpenMP would deadlock).")
            return 1
        if self.threads == 1:
            return 1
        cap = available_cores() if self.threads == "auto" else min(int(self.threads), available_cores())
        if n_points is None:
            return max(1, cap)
        return int(max(1, min(cap, int(n_points) // _PTS_PER_THREAD)))

    def _fopts(self, n_points):
        return {"nthreads": self._nthreads(n_points)}

    def _get_pool(self):
        if self._pool is not None and self._pool_pid != os.getpid():     # inherited from a parent process
            self._pool = self._pool_fin = None
        if self._pool is None:
            n = max(1, min(self.n_workers, available_cores()))
            self._pool = _fork_executor(n, _init_eval_worker, (weakref.ref(self),))
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

    def __getstate__(self):                     # pickling (e.g. with cloudpickle) drops the process pool
        d = self.__dict__.copy()
        d["_pool"] = d["_pool_fin"] = d["_pool_pid"] = None
        return d

    # ------------------------------------------------------------------ sums over nodes
    def _sum_w(self, phi, W, ws):
        """sum_j W_j exp(i w phi_j) for an array of w."""
        if phi.size == 0:
            return np.zeros(ws.size, dtype=complex)
        if finufft is not None and ws.size >= 4:
            return finufft.nufft1d3(phi, W.astype(complex), ws, isign=1, eps=self.nufft_eps, **self._fopts(phi.size))
        if ws.size > 20:
            _warn_once("nufft1", "finufft is not installed: frequency sums are done directly (much slower). "
                                 "pip install finufft")
        out = np.empty(ws.size, dtype=complex)
        for i, w in enumerate(ws):
            acc = 0.0 + 0.0j
            for s in range(0, phi.size, 1_000_000):
                acc += np.sum(W[s:s + 1_000_000] * np.exp(1j * w * phi[s:s + 1_000_000]))
            out[i] = acc
        return out

    def _sum_y(self, X1, X2, C, k1, k2):
        """sum_j C_j exp(-i (k1 X1_j + k2 X2_j)) for arrays of targets (k1, k2)."""
        if finufft is not None and k1.size >= 4:
            return finufft.nufft2d3(X1, X2, C, k1, k2, isign=-1, eps=self.nufft_eps, **self._fopts(X1.size))
        if k1.size > 20:
            _warn_once("nufft2", "finufft is not installed: maps are summed directly (much slower). pip install finufft")
        out = np.empty(k1.size, dtype=complex)
        for i in range(k1.size):
            out[i] = np.sum(C * np.exp(-1j * (k1[i] * X1 + k2[i] * X2)))
        return out

    def _phases(self, A, x1, x2, y):
        if ne is not None and A.size > 100_000:
            _set_ne_threads(self._nthreads(A.size))
            return ne.evaluate("A - x1 * a - x2 * b", local_dict=dict(A=A, x1=x1, x2=x2, a=float(y[0]), b=float(y[1])))
        return A - x1 * y[0] - x2 * y[1]

    # ------------------------------------------------------------------ tail for one y: table in 1/w
    def _tail_table(self, bi, y):
        key = (float(y[0]), float(y[1]))
        hit = self._tcache.get(bi)
        if hit is not None and hit[0] == key:
            return hit[1], hit[2]
        b = self.bands[bi]
        sig = (b["ct"] * y[0] + b["st"] * y[1])[:, None, None]
        wj = (1 / b["eps"])[None, :, None]
        L = self.LX[None, None, :]
        if ne is not None:
            _set_ne_threads(self._nthreads(b["dA"].size))
            E3 = ne.evaluate("exp(1j * wj * (dA - dr * sig) + L)", local_dict=dict(wj=wj, dA=b["dA"], dr=b["dr"], sig=sig, L=L))
        else:
            E3 = np.exp(1j * wj * (b["dA"] - b["dr"] * sig) + L)
        Sv = E3 @ self.LW                                                    # (directions, n_cheb)
        nc = Sv.shape[1]
        if nc == 1:
            coef = Sv
        else:
            k = np.arange(nc)
            Tm = np.cos(np.outer(np.arange(nc), np.pi * (k + 0.5) / nc))
            coef = (2.0 / nc) * Sv @ Tm.T
            coef[:, 0] *= 0.5
        P = b["A0"] - b["x0"][0] * y[0] - b["x0"][1] * y[1]
        D = np.ascontiguousarray(((1j * (2 * np.pi / b["th"].size) / b["al"])[None, :] * coef.T))   # (n_cheb, directions)
        self._tcache[bi] = (key, P, D)
        return P, D

    def _tail_spectrum(self, bi, y, wb):
        b = self.bands[bi]
        P, D = self._tail_table(bi, y)
        nc = D.shape[0]
        if nc == 1:
            T = np.ones((1, wb.size))
        else:
            xx = (2 / wb - (b["e_hi"] + b["e_lo"])) / (b["e_hi"] - b["e_lo"])
            T = np.empty((nc, wb.size)); T[0] = 1.0
            T[1] = xx
            for m in range(2, nc):
                T[m] = 2 * xx * T[m - 1] - T[m - 2]
        if finufft is not None and P.size * wb.size > 60_000:
            R = finufft.nufft1d3(P, D, wb, isign=1, eps=self.nufft_eps, **self._fopts(P.size * nc)).reshape(nc, wb.size)
        else:
            R = D @ np.exp(1j * np.outer(P, wb))
        return np.sum(R * T, axis=0) / wb

    # ------------------------------------------------------------------ public evaluation
    def spectrum(self, ws, y=None, interpolate=None):
        """F at an array of frequencies for one source position y (default: the centre).
        interpolate: None = the solver's setting; True / False / 'auto' (see the class docstring)."""
        with _blas_single():
            F = self._spectrum(ws, y, interpolate)
        if self.time_origin is not None:
            F = F * np.exp(-1j * np.atleast_1d(np.asarray(ws, dtype=float)) * self._t0(self._as_y(y)))
        return F

    def _t0(self, y):
        """Time origin for source position(s) y, shape (..., 2) (see time_origin)."""
        return self.first_image_delay(y) if self.time_origin == "first_image" else self.time_origin

    def first_image_delay(self, y):
        """
        Time delay of the first image, T_min(y) = min over x of T(x) = |x - y|^2/2 - psi(x) (the global
        minimum of the Fermat potential; GLoW's t_min), for source positions y of shape (2,) or (..., 2).
        All local minima of T on a grid over |x - c| <= x_b (where every image lies) are refined by a
        damped Newton iteration to rounding (gradient by the complex step, Hessian by central
        differences of the gradient); the lowest is returned.
        """
        y = np.asarray(y, dtype=float)
        shape = y.shape[:-1]
        yf = y.reshape(-1, 2)
        if yf.shape[0] == 1:
            key = (float(yf[0, 0]), float(yf[0, 1]))
            if key in self._tmincache:
                return self._tmincache[key] if shape == () else np.full(shape, self._tmincache[key])
        if self._tmingrid is None:
            g = np.linspace(-self.xb0, self.xb0, 161)
            X1, X2 = np.meshgrid(self.c[0] + g, self.c[1] + g, indexing="ij")
            with np.errstate(all="ignore"):
                ps = np.real(self.lens.psi(X1, X2))
            self._tmingrid = (X1, X2, np.where(np.isfinite(ps), ps, np.nan))
        X1, X2, ps = self._tmingrid
        n = X1.shape[0]
        out = np.empty(yf.shape[0])
        step = max(1, 4_000_000 // X1.size)
        for a in range(0, yf.shape[0], step):
            yc = yf[a:a + step]
            with np.errstate(all="ignore"):
                T = 0.5 * ((X1[None] - yc[:, 0, None, None]) ** 2 + (X2[None] - yc[:, 1, None, None]) ** 2) - ps[None]
            T = np.where(np.isfinite(T), T, np.inf)
            C = T[:, 1:-1, 1:-1]
            m = np.isfinite(C)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di or dj:
                        m &= C <= T[:, 1 + di:n - 1 + di, 1 + dj:n - 1 + dj]
            k, i, j = np.nonzero(m)
            if k.size == 0 or np.unique(k).size < yc.shape[0]:      # no interior minimum: lowest grid point
                flat = T.reshape(yc.shape[0], -1).argmin(axis=1)
                k = np.concatenate([k, np.arange(yc.shape[0])])
                i = np.concatenate([i, flat // n - 1]); j = np.concatenate([j, flat % n - 1])
            x1, x2, Tm = self._refine_min(X1[i + 1, j + 1], X2[i + 1, j + 1], yc[k, 0], yc[k, 1])
            best = np.full(yc.shape[0], np.inf)
            np.minimum.at(best, k, Tm)
            out[a:a + step] = best
        if yf.shape[0] == 1:
            if len(self._tmincache) > 256:
                self._tmincache.clear()
            self._tmincache[(float(yf[0, 0]), float(yf[0, 1]))] = float(out[0])
        return float(out[0]) if shape == () else out.reshape(shape)

    def _refine_min(self, x1, x2, y1, y2):
        """Damped Newton iteration for minima of T(x) = |x - y|^2/2 - psi(x) from starts (x1, x2)."""
        lens = self.lens
        x1, x2 = np.array(x1, dtype=float), np.array(x2, dtype=float)

        def T(a, b, ya, yb):
            with np.errstate(all="ignore"):
                v = 0.5 * ((a - ya) ** 2 + (b - yb) ** 2) - np.real(lens.psi(a, b))
            return np.where(np.isfinite(v), v, np.inf)

        T0 = T(x1, x2, y1, y2)
        active = np.isfinite(T0)
        for _ in range(50):
            if not active.any():
                break
            a, b, ya, yb = x1[active], x2[active], y1[active], y2[active]
            with np.errstate(all="ignore"):
                g1, g2 = lens.grad(a, b)
                G1, G2 = a - ya - g1, b - yb - g2
                h = 1e-6 * (1.0 + np.hypot(a, b))
                p1a, p2a = lens.grad(a + h, b)
                p1b, p2b = lens.grad(a - h, b)
                q1a, q2a = lens.grad(a, b + h)
                q1b, q2b = lens.grad(a, b - h)
                H11 = 1 - (p1a - p1b) / (2 * h)
                H22 = 1 - (q2a - q2b) / (2 * h)
                H12 = -0.25 * ((p2a - p2b) + (q1a - q1b)) / h
                det = H11 * H22 - H12 * H12
                s1 = (H22 * G1 - H12 * G2) / det
                s2 = (H11 * G2 - H12 * G1) / det
            newton = np.isfinite(s1) & np.isfinite(s2) & (det > 0) & (H11 > 0)
            s1 = np.where(newton, s1, 0.1 * G1)
            s2 = np.where(newton, s2, 0.1 * G2)
            Ta = T0[active]
            lam = np.ones(a.size)
            na, nb = a - s1, b - s2
            Tn = T(na, nb, ya, yb)
            for _ in range(30):                               # backtracking: T must not increase
                worse = Tn > Ta + 1e-15 * np.abs(Ta)
                if not worse.any():
                    break
                lam = np.where(worse, 0.5 * lam, lam)
                na, nb = a - lam * s1, b - lam * s2
                Tn = np.where(worse, T(na, nb, ya, yb), Tn)
            done = (np.hypot(G1, G2) < 1e-13) | (np.hypot(lam * s1, lam * s2) < 1e-15 * (1 + np.hypot(a, b)))
            idx = np.nonzero(active)[0]
            x1[idx], x2[idx], T0[idx] = na, nb, np.minimum(Tn, Ta)
            active[idx[done]] = False
        return x1, x2, T(x1, x2, y1, y2)

    def _spectrum(self, ws, y=None, interpolate=None):
        ws = np.atleast_1d(np.asarray(ws, dtype=float))
        y = self._as_y(y)
        self._check_y(y[0], y[1])
        F = np.ones(ws.size, dtype=complex)
        pos = ws > 0
        wp = ws[pos]
        if wp.size == 0:
            return F
        self._check_w(wp)
        mode = self.interpolate if interpolate is None else _interp_mode(interpolate, self.interpolate)
        if mode and wp.size >= 64:
            Fi = self._spectrum_interp(wp, y, auto=(mode == "auto"))
            if Fi is not None:
                F[pos] = Fi
                return F
        F[pos] = self._spectrum_exact(wp, y)
        return F

    def _spectrum_exact(self, wp, y):
        total = self._sum_w(self._phases(self.hA, self.hx1, self.hx2, y), self.hW, wp)
        band_idx = self._band_index(wp)
        for bi in np.unique(band_idx):
            sel = band_idx == bi
            b = self.bands[bi]
            if b["seg"][0].size:
                sx1, sx2, sA, sW = b["seg"]
                total[sel] += self._sum_w(self._phases(sA, sx1, sx2, y), sW, wp[sel])
            total[sel] += self._tail_spectrum(bi, y, wp[sel])
        return wp / (2j * np.pi) * np.exp(0.5j * wp * (y @ y)) * total

    # optional interpolation in w
    def _stationary_grid(self, n=401):
        """
        The lens potential and its derivative  on a square grid covering the region where images can lie is computed(|x - c| <= x_b:
        beyond it the radial derivative of the time delay is positive for every y of the solver).
        """
        if self._tdgrid is None:
            g = np.linspace(-self.xb0, self.xb0, n)
            X1, X2 = np.meshgrid(self.c[0] + g, self.c[1] + g, indexing="ij")
            with np.errstate(all="ignore"):
                ps = np.real(self.lens.psi(X1, X2))
                g1, g2 = self.lens.grad(X1, X2)
            self._tdgrid = (X1, X2, ps, g1, g2)
        return self._tdgrid

    def _time_delay_range(self, y):
        """
        This is a 2D verison of the smae time delay that you can find in the solver1d.py script.
        (T_min, T_max) of the time delay T(x) = |x - y|^2/2 - psi(x) over its stationary points
        (images; local minima of |grad T| also catch merging/virtual images) and over singular points
        where psi is finite (cusps, cores). F(w) oscillates in w only through exp(i w T) at these.
        
        """
        key = (float(y[0]), float(y[1]))
        if key in self._tdcache:
            return self._tdcache[key]
        X1, X2, ps, g1, g2 = self._stationary_grid()
        with np.errstate(all="ignore"):
            Q = (X1 - y[0] - g1) ** 2 + (X2 - y[1] - g2) ** 2
            T = 0.5 * ((X1 - y[0]) ** 2 + (X2 - y[1]) ** 2) - ps
        Q = np.where(np.isfinite(Q) & np.isfinite(T), Q, np.inf)
        C = Q[1:-1, 1:-1]
        m = np.isfinite(C)
        n1, n2 = Q.shape
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    m &= C <= Q[1 + di:n1 - 1 + di, 1 + dj:n2 - 1 + dj]
        Ts = list(T[1:-1, 1:-1][m])
        for p in self.lens.singular:
            with np.errstate(all="ignore"):
                v = np.real(self.lens.psi(np.array([p[0] + 1e-12, p[0] + 1e-9]), np.array([p[1], p[1]])))
            if np.all(np.isfinite(v)) and abs(v[0] - v[1]) < 1e-6 * (1 + abs(v[0])):
                Ts.append(0.5 * ((p[0] - y[0]) ** 2 + (p[1] - y[1]) ** 2) - v[0])
        if not Ts:
            Ts = [0.5 * ((self.c[0] - y[0]) ** 2 + (self.c[1] - y[1]) ** 2)]
        out = (float(min(Ts)), float(max(Ts)))
        if len(self._tdcache) > 64:
            self._tdcache.clear()
        self._tdcache[key] = out
        return out

    def _spectrum_interp(self, wp, y, auto=False, n_p=24):
        """F at wp by Chebyshev interpolation on panels (nodes and check frequencies in one transform),
        or None if not worth it / not accurate enough after 3 refinements."""
        T_lo, T_hi = self._time_delay_range(y)
        Tc = 0.5 * (T_lo + T_hi)
        B = 0.5 * (T_hi - T_lo) * 1.2 + 0.5                 # half bandwidth of F e^{-i w Tc}, with margin
        w_lo, w_hi = wp.min(), wp.max()
        xc = _cheb_points(n_p)
        for attempt in range(4):
            edges = _interp_panels(w_lo, w_hi, B)
            n_nodes = (edges.size - 1) * n_p
            if (auto and wp.size < max(8000, 3 * n_nodes)) or n_nodes >= wp.size:
                return None                                  # exact is cheaper
            mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
            wn = (mid[:, None] + half[:, None] * xc[None, :]).ravel()
            k = np.unique(np.linspace(0, edges.size - 2, min(6, edges.size - 1)).astype(int))
            wt = np.clip(np.concatenate([mid[k] + 0.985 * half[k], mid[k] - 0.6 * half[k]]), w_lo, w_hi)
            Fall = self._spectrum_exact(np.concatenate([wn, wt]), y)
            G = (Fall[:wn.size] * np.exp(-1j * wn * Tc)).reshape(-1, n_p)
            Ft = Fall[wn.size:]
            err = np.max(np.abs(_interp_eval(wt, edges, G, xc) * np.exp(1j * wt * Tc) - Ft) / np.maximum(1, np.abs(Ft)))
            if err <= self.interp_tol:
                return _interp_eval(wp, edges, G, xc) * np.exp(1j * wp * Tc)
            B *= 2.0                                         # unexpected structure: finer panels
        _warn_once("interp", f"interpolation in w did not reach interp_tol={self.interp_tol:g}; computing all frequencies")
        return None

    many = spectrum                                   # version-2 name

    def map(self, w, y1, y2, n_xi=512, n_cheb_sigma=48):
        """F at one frequency for source positions (y1, y2) (arrays of any common shape)."""
        with _blas_single():
            return self._map(w, y1, y2, n_xi, n_cheb_sigma)

    def _map(self, w, y1, y2, n_xi=512, n_cheb_sigma=48):
        w = float(w)
        y1 = np.asarray(y1, dtype=float); y2 = np.asarray(y2, dtype=float)
        shape = np.broadcast(y1, y2).shape
        y1, y2 = np.broadcast_to(y1, shape).ravel(), np.broadcast_to(y2, shape).ravel()
        self._check_y(y1, y2)
        if w == 0:
            return np.ones(shape, dtype=complex)
        self._check_w(np.array([w]))
        if self.Yr == 0:                              # single-source solver: every target is the centre
            return np.full(shape, self.spectrum([w])[0])
        if self.ext <= 1.0:
            raise ValueError(f"map() needs a solver built with map_ext > 1 (this one has map_ext = {self.ext:g}): the "
                             "tail is windowed over map_ext * y_radius. Use the default map_ext=1.5, or spectrum() "
                             "for each source position.")
        bi = self._band_of(w)
        b = self.bands[bi]
        lens, c, LX, LW = self.lens, self.c, self.LX, self.LW
        X1, X2, C = [self.hx1], [self.hx2], [self.hW * np.exp(1j * w * self.hA)]
        if b["seg"][0].size:
            sx1, sx2, sA, sW = b["seg"]
            X1.append(sx1); X2.append(sx2); C.append(sW * np.exp(1j * w * sA))
        # tail: per direction a smooth ridge function g_i(sigma), sigma = n_i.y, windowed and FFT'd
        ct, st, al, xb, ub = b["ct"], b["st"], b["al"], b["xb"], b["ub"]
        Nt = ct.size
        t = LX[None, :] / (w * al[:, None])
        r = np.sqrt(2 * (ub + 1j * t))
        x1, x2 = c[0] + r * ct[:, None], c[1] + r * st[:, None]
        dA = 0.5 * (x1 * x1 + x2 * x2) - lens.psi(x1, x2) - b["A0"][:, None]
        dr = r - xb
        sig0 = ct * self.y0[0] + st * self.y0[1]
        xc = _cheb_points(n_cheb_sigma) * self.ext
        pref = (2 * np.pi / Nt) * 1j / (w * al)
        g = np.empty((Nt, n_cheb_sigma), dtype=complex)
        for l, xi in enumerate(xc):
            sig = sig0 + self.Yr * xi
            g[:, l] = pref * np.sum(LW[None, :] * np.exp(LX[None, :] + 1j * w * (dA - dr * sig[:, None])), axis=1)
        xu = -self.ext + 2 * self.ext * np.arange(n_xi) / n_xi
        B = _step(np.abs(xu) / self.ext, 1 / self.ext)
        v = (g @ _bary_matrix(xc, xu).T) * B[None, :]
        qs = np.fft.fftfreq(n_xi, 1.0 / n_xi)
        gh = (np.fft.fft(v, axis=1) / n_xi) * np.exp(1j * np.pi * qs)[None, :]
        keep = np.max(np.abs(gh), axis=0) > 1e-17 * np.max(np.abs(gh))
        qk = qs[keep]
        gam = gh[:, keep] * np.exp(1j * w * b["A0"])[:, None] * np.exp(-1j * np.pi * np.outer(sig0, qk) / (self.ext * self.Yr))
        rad = xb - np.pi * qk / (w * self.ext * self.Yr)
        X1.append((c[0] + np.outer(ct, rad)).ravel()); X2.append((c[1] + np.outer(st, rad)).ravel()); C.append(gam.ravel())
        out = self._sum_y(np.concatenate(X1), np.concatenate(X2), np.concatenate(C), w * y1, w * y2)
        F = w / (2j * np.pi) * np.exp(0.5j * w * (y1 * y1 + y2 * y2)) * out
        if self.time_origin is not None:
            F = F * np.exp(-1j * w * self._t0(np.stack([y1, y2], axis=-1)))
        return F.reshape(shape)

    def evaluate(self, ws, ys):
        """Table F[k, m] = F(ws[m], ys[k]); ys has shape (K, 2). Loops over the shorter axis: one
        spectrum per source position, or one map per frequency. With n_workers > 1 the rows are
        computed by worker processes (one FINUFFT thread each)."""
        ws = np.atleast_1d(np.asarray(ws, dtype=float))
        ys = np.atleast_2d(np.asarray(ys, dtype=float))
        self._check_y(ys[:, 0], ys[:, 1])
        self._check_w(ws[ws > 0])
        # maps need the window margin of map_ext > 1; otherwise always one spectrum per source position
        by_y = ys.shape[0] <= ws.size or self.Yr == 0 or self.ext <= 1.0
        n_rows = ys.shape[0] if by_y else ws.size
        out = np.empty((ys.shape[0], ws.size), dtype=complex)
        # A serial row is one FINUFFT transform on all its threads; in a worker it runs on one thread.
        # FINUFFT's threads give at most about half the cores' worth on one transform, so processes
        # only pay off once at least that many rows run at the same time, each process getting >= 2
        # rows (which also pays for starting the processes).
        nproc = self._pmax(n_rows // 2)
        serial_threads = self._nthreads(self.hW.size)
        if nproc > 1 and nproc < _THREAD_EFF * serial_threads:
            nproc = 1
        if nproc > 1:
            n_tasks = min(n_rows, 4 * nproc)
            rows = [np.arange(i, n_rows, n_tasks) for i in range(n_tasks)]   # round robin: balanced costs
            tasks = [(True, ws, ys[r]) if by_y else (False, ws[r], ys) for r in rows]
            try:
                with _quiet_fork():
                    res = list(self._get_pool().map(_eval_rows, tasks))
            except BrokenProcessPool:
                self.close()
                raise
            for r, v in zip(rows, res):
                if by_y:
                    out[r] = v
                else:
                    out[:, r] = v
        elif by_y:
            for k, y in enumerate(ys):
                out[k] = self.spectrum(ws, y)
        else:
            for m, w in enumerate(ws):
                out[:, m] = self.map(w, ys[:, 0], ys[:, 1])
        return out

    def __call__(self, w, y=None):
        return self.spectrum([w], y)[0]


# Lighter settings for parameter estimation: ~2x fewer nodes and faster sums, accuracy ~1e-7 in F
# (far below detector noise). Usage: Solver2D(lens, y, wmin, wmax, y_radius=R, map_ext=1.0, **PE_SETTINGS)
PE_SETTINGS = dict(ngl=24, nlag=24, nufft_eps=1e-9, bw_factor=1.15, bw_margin=20, pou_density=12)



def _err_kw(s, kw):
    kw2 = dict(kw); kw2["theta_factor"] = 2 * kw.get("theta_factor", 1.0)
    kw3 = dict(kw); kw3["xb_min"] = 1.5 * s.xb0
    return kw2, kw3


def amplification(w, y, lens, estimate_error=False, return_info=False, **kw):
    """
    F(w, y) for scalars or arrays, with numpy broadcasting. A source position is a pair (y1, y2) in
    the LAST axis of y: y has shape (..., 2), and w broadcasts against y[..., 0]. One solver serves all
    points, and the fastest path is chosen from the inputs:

        amplification(0.5, (0.3, 0.1), lens)                     one value
        amplification(ws, (0.3, 0.1), lens)                      many frequencies, one y  -> s.spectrum
        amplification(0.5, ys, lens)          ys: (K, 2)          one frequency, many y    -> s.evaluate / s.map
        amplification(0.5, np.stack([Y1, Y2], -1), lens)          map on an image grid, shape Y1.shape
        amplification(ws[None, :], ys[:, None, :], lens)          table (K, len(ws))       -> s.evaluate
        amplification(ws, ys, lens)           ws: (K,), ys: (K, 2) pairs (ws[k], ys[k])

    Pairs are grouped by the coordinate with fewer distinct values (one spectrum per distinct
    position, or one map per distinct w); if they fill most of a grid, the table is computed instead.
    It supports all keywords of Solver2D, e.g. ngl, nlag, y_radius, map_ext, threads, n_workers,
    interpolate, and **PE_SETTINGS.

    Parameters:
    -----------
    w : float or array_like
        Frequencies (w = 0 gives F = 1).
    y : array_like, shape (..., 2)
        Source positions (y1, y2); the leading axes broadcast against w.
    lens : Lens
        Lens model (see Lens and the ready-made lenses).
    estimate_error : bool, optional
        If True, also returns max(|F - F'|, |F - F''|) with F' from twice the tail directions and F''
        from a farther split (an estimate of the error, same shape as F).
    return_info : bool, optional
        If True, also returns the solver's info dict.
    **kw : dict
        Keyword arguments of Solver2D (see the Solver2D class for available options).

    Returns:
    --------
    F with the broadcast shape of w and y[..., 0] (a complex scalar if w is a scalar and y one
    position); then the error (if estimate_error) and the info dict (if return_info), as a tuple.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 0 or y.shape[-1] != 2:
        raise ValueError("y must hold source positions (y1, y2) in its last axis, shape (..., 2)")
    wb, y1b, y2b = np.broadcast_arrays(np.asarray(w, dtype=float), y[..., 0], y[..., 1])
    shape, wf = wb.shape, wb.ravel()
    yf = np.stack([y1b.ravel(), y2b.ravel()], axis=1)
    pos = wf > 0

    def pack(F, E, info):
        F = F.reshape(shape)
        F = F[()] if shape == () else F
        out = [F]
        if estimate_error:
            E = E.reshape(shape)
            out.append(E[()] if shape == () else E)
        if return_info:
            out.append(info)
        return out[0] if len(out) == 1 else tuple(out)

    if not pos.any():
        return pack(np.ones(wf.size, dtype=complex), np.zeros(wf.size), {})
    wlo, whi = wf[pos].min(), wf[pos].max()
    yu = np.unique(yf[pos], axis=0)
    with _solver_for(lens, wlo, whi, yu, kw) as s:
        F = _evaluate_pairs(s, wf, yf)
        info = dict(s.info, n_eval=s.info.get("n_head_nodes")) if return_info else {}
    E = None
    if estimate_error:
        kw2, kw3 = _err_kw(s, kw)
        E = np.zeros(wf.size)
        for k in (kw2, kw3):
            with _solver_for(lens, wlo, whi, yu, k) as s2:
                E = np.maximum(E, np.abs(_evaluate_pairs(s2, wf, yf) - F))
    return pack(F, E, info)


def _solver_for(lens, wmin, wmax, ys, kw):
    """One solver for the frequencies [wmin, wmax] and the distinct source positions ys (shape (K, 2)).
    A y_radius in kw is kept if it is larger than needed."""
    kw = dict(kw)
    r_user = float(kw.pop("y_radius", 0.0))
    if ys.shape[0] == 1:
        return Solver2D(lens, ys[0], wmin, wmax, y_radius=r_user, **kw)
    yc, R = _region_of(ys[:, 0], ys[:, 1])
    return Solver2D(lens, yc, wmin, wmax, y_radius=max(R, r_user), **kw)


def _evaluate_pairs(s, wf, yf):
    """F at the pairs (wf[i], yf[i]) (wf flat, yf shape (N, 2)) with the solver s, by the cheapest path."""
    out = np.ones(wf.size, dtype=complex)
    pos = wf > 0
    wq, yq = wf[pos], yf[pos]
    uw, iw = np.unique(wq, return_inverse=True)
    uy, iy = np.unique(yq, axis=0, return_inverse=True)
    iw, iy = iw.reshape(-1), iy.reshape(-1)
    if uy.shape[0] == 1:                                     # one source position: one spectrum
        vals = s.spectrum(uw, uy[0])[iw]
    elif uw.size * uy.shape[0] <= 4 * wq.size:               # one frequency, or (nearly) a grid: the table
        vals = s.evaluate(uw, uy)[iy, iw]                    # (one map per w or one spectrum per y)
    else:                                                    # scattered pairs
        vals = np.empty(wq.size, dtype=complex)
        if uy.shape[0] <= uw.size:
            for k in range(uy.shape[0]):
                sel = iy == k
                vals[sel] = s.spectrum(wq[sel], uy[k])
        else:
            for k in range(uw.size):
                sel = iw == k
                vals[sel] = s.evaluate(uw[k:k + 1], yq[sel])[:, 0]
    out[pos] = vals
    return out


def amplification_spectrum(ws, y, lens, estimate_error=False, **kw):
    """F at many frequencies for one source position (one grid, one non-uniform FFT): the frequency
    spectrum of the amplification. Formerly amplification_many."""
    ws = np.atleast_1d(np.asarray(ws, dtype=float))
    wp = ws[ws > 0]
    if wp.size == 0:
        F = np.ones(ws.size, dtype=complex)
        return (F, np.zeros(ws.size)) if estimate_error else F
    with Solver2D(lens, y, wp.min(), wp.max(), **kw) as s:
        F = s.spectrum(ws)
    if not estimate_error:
        return F
    kw2, kw3 = _err_kw(s, kw)
    F2 = Solver2D(lens, y, wp.min(), wp.max(), **kw2).spectrum(ws)
    F3 = Solver2D(lens, y, wp.min(), wp.max(), **kw3).spectrum(ws)
    return F, np.maximum(np.abs(F2 - F), np.abs(F3 - F))


def _region_of(y1, y2):
    y1, y2 = np.ravel(y1), np.ravel(y2)
    yc = np.array([(y1.min() + y1.max()) / 2, (y2.min() + y2.max()) / 2])
    R = float(np.max(np.hypot(y1 - yc[0], y2 - yc[1])))
    return yc, (R * (1 + 1e-9) if R > 1e-12 else 0.0)        # one position: single-source solver


def amplification_map(w, y1, y2, lens, **kw):
    """F at one frequency for many source positions (one grid, one 2D non-uniform FFT)."""
    yc, R = _region_of(y1, y2)
    with Solver2D(lens, yc, w, w, y_radius=R, **kw) as s:
        return s.map(w, y1, y2)


def amplification_table(ws, ys, lens, **kw):
    """Table F[k, m] = F(ws[m], ys[k]) for many frequencies and many source positions (ys: shape (K, 2)).
    Formerly amplification_grid.
    Pass n_workers=None (all cores) or n_workers=k to build the grid and compute the rows in parallel."""
    ws = np.atleast_1d(np.asarray(ws, dtype=float))
    ys = np.atleast_2d(np.asarray(ys, dtype=float))
    wp = ws[ws > 0]
    if wp.size == 0:
        return np.ones((ys.shape[0], ws.size), dtype=complex)
    yc, R = _region_of(ys[:, 0], ys[:, 1])
    with Solver2D(lens, yc, wp.min(), wp.max(), y_radius=R, **kw) as s:
        return s.evaluate(ws, ys)


# former names, kept so that existing scripts keep working
def amplification_many(ws, y, lens, estimate_error=False, **kw):
    """Former name of amplification_spectrum (deprecated)."""
    warnings.warn("amplification_many is now amplification_spectrum", DeprecationWarning, stacklevel=2)
    return amplification_spectrum(ws, y, lens, estimate_error=estimate_error, **kw)


def amplification_grid(ws, ys, lens, **kw):
    """Former name of amplification_table (deprecated)."""
    warnings.warn("amplification_grid is now amplification_table", DeprecationWarning, stacklevel=2)
    return amplification_table(ws, ys, lens, **kw)