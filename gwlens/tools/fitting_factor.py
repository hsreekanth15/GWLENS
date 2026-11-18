import numpy as np
import itertools
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pycbc.filter import match
from pycbc import conversions
from pycbc.waveform import get_fd_waveform
from scipy.optimize import minimize, minimize_scalar
from gwlens.solver.lensmodels import get_lens_model, lens_frequency_series


#FINUFFT for faster non uniform FFT 
# Installed along with gwlens...
#But the fallback is added for older systems where the architecture don't support FINUFFT installation...
try:
    import finufft
except ImportError:                                  
    finufft = None

#Setting thread limits here otherwise, GWLENS's solver
#tried to use every cores for FFT, BLAS and FINUFFT and makes the evaluations slower

_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "VECLIB_MAXIMUM_THREADS")
_LIMITS = None

# template parameters theta = (mchirp, q, spin1z, spin2z): bounds and the spread of the extra starts
TEMPLATE_PARAMETERS = ("mchirp", "q", "spin1z", "spin2z")
DEFAULT_BOUNDS = ((3.5, 150.0), (1.0, 20.0), (-0.99, 0.99), (-0.99, 0.99))
DEFAULT_SIGMA = (2.0, 0.2, 0.1, 0.1)


def masses_from_mchirp_q(mc, q):
    """
    Chirpmass and massratio to component masses
    (m1, m2) from chirp mass and q = m1/m2 >= 1 (any mass unit).
    Kept here for For LISA-mbbh
    """
    m2 = mc * (1 + q) ** 0.2 * q ** -0.6
    return q * m2, m2


def mchirp_q_from_masses(m1, m2):

    """
    Masses to chirp-mass
    Kept here for For LISA-mbbh
    
    """
    m1, m2 = max(m1, m2), min(m1, m2)
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2, m1 / m2


#PyCBC (ground based)
@dataclass
class PyCBCSetup:
    """
    The unlensed source and the analysis settings, through PyCBC.

    source      (mchirp, q, spin1z, spin2z) of the injected binary (detector frame; q = m1/m2 >= 1)
    psd         a PyCBC FrequencySeries, or the name of a pycbc.psd analytic PSD
    wf_kwargs   further get_fd_waveform arguments (distance, inclination, coa_phase, ...)
    bounds, sigma   template-parameter bounds and the spread of the extra optimiser starts
    """
    source: tuple
    f_lower: float = 20.0
    f_upper: float = 1024.0
    delta_f: float = 1.0 / 16
    approximant: str = "IMRPhenomXAS"
    psd: object = "aLIGOZeroDetHighPower"
    wf_kwargs: dict = field(default_factory=dict)
    convention: str = "lal"                     # how F multiplies h(f), see LensModel.amplification
    bounds: tuple = DEFAULT_BOUNDS
    sigma: tuple = DEFAULT_SIGMA

    @property
    def length(self):
        return int(round(self.f_upper / self.delta_f)) + 1

    def waveform_kwargs(self):
        kw = dict(approximant=self.approximant, f_lower=self.f_lower, f_final=self.f_upper,
                  delta_f=self.delta_f, distance=100.0, inclination=0.0, coa_phase=0.0)
        kw.update(self.wf_kwargs)
        return kw

    def get_psd(self):
        if isinstance(self.psd, str):
            import pycbc.psd
            if getattr(self, "_psd_cache", None) is None:
                self._psd_cache = getattr(pycbc.psd, self.psd)(self.length, self.delta_f, self.f_lower)
            return self._psd_cache
        return self.psd

    def template(self, theta):
        """
        Unlensed frequency-domain waveform for theta = (mchirp, q, spin1z, spin2z).
        
        """

        mc, q, s1z, s2z = theta
        hp, _ = get_fd_waveform(mass1=conversions.mass1_from_mchirp_q(mc, q),
                                mass2=conversions.mass2_from_mchirp_q(mc, q),
                                spin1z=s1z, spin2z=s2z, **self.waveform_kwargs())
        hp.resize(self.length)
        return hp

    def lensed(self, h, model, params):
        return lens_frequency_series(h, model, params, self.f_lower, self.f_upper, self.convention)

    def match(self, a, b, psd=None):
        return match_value(a, b, self.get_psd() if psd is None else psd, self.f_lower, self.f_upper)


Setup = PyCBCSetup                              


def template(theta, setup):
    """Unlensed template of `setup` for theta = (mchirp, q, spin1z, spin2z)."""
    return setup.template(theta)


def match_value(a, b, psd, f_lower, f_upper):
    """
    PyCBC match of a and b, maximised over time and phase. subsample_interpolation: the peak is
    interpolated between time samples; without it the time is maximised only on the sample grid
    (1 / (2 f_upper)), which underestimates matches by up to ~3e-3 for stellar-mass binaries.
    
    """
    
    m, _ = match(a, b, psd=psd, low_frequency_cutoff=f_lower, high_frequency_cutoff=f_upper,
                 subsample_interpolation=True)
    return float(m)


# arrays (e.g. LISA) 
#Written for Shashwat's LISA-mbbh package...
#Requires installation of lisa-mbbh
# Link to the repo to be added....
def _trapz_weights(f):
    w = np.zeros_like(f)
    d = np.diff(f)
    w[:-1] += d / 2
    w[1:] += d / 2
    return w


def overlap(a, b, psd, f):
    """
    Complex <a|b> = 4 sum_f conj(a) b / S df (trapezoid on the grid f), summed over channels.
    a, b: (n_f,) or (n_channels, n_f); psd broadcast to them; PSD values <= 0, inf or nan get no weight.
    
    """
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    return 4.0 * np.sum(np.conj(a) * b * _inverse_psd_weights(psd, f, a.shape))


def _inverse_psd_weights(psd, f, shape):
    psd = np.broadcast_to(np.atleast_2d(psd), shape)
    ok = np.isfinite(psd) & (psd > 0)
    return np.where(ok, 1.0 / np.where(ok, psd, 1.0), 0.0) * _trapz_weights(np.asarray(f, dtype=float))


def grid_match(a, b, psd, f, dt_max, return_dt=False):
    """
    max over |dt| <= dt_max and over phase of |<a | b exp(-2 pi i f dt)>| / (|a| |b|), on any frequency
    grid. The phase maximisation is analytic (modulus); dt is scanned on a grid of step 1 / (8 f_max)
    (one type-3 NUFFT, or chunked direct sums without FINUFFT) and refined with a bounded 1D search.
    """
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    wt = _inverse_psd_weights(psd, f, a.shape)
    c = 4.0 * np.sum(np.conj(a) * b * wt, axis=0)                # per-frequency integrand
    na = np.sqrt(4.0 * np.sum(np.abs(a) ** 2 * wt))
    nb = np.sqrt(4.0 * np.sum(np.abs(b) ** 2 * wt))
    if na == 0 or nb == 0:
        return (0.0, 0.0) if return_dt else 0.0
    keep = np.abs(c) > 1e-14 * np.abs(c).max()                   # drop the empty part of the band
    fk, ck = np.asarray(f, dtype=float)[keep], c[keep]
    O = lambda dt: abs(np.sum(ck * np.exp(-2j * np.pi * fk * dt)))
    if dt_max <= 0:
        m, dt = O(0.0), 0.0
    else:
        step = 1.0 / (8.0 * fk.max())
        dts = np.arange(-dt_max, dt_max + step, step)
        if finufft is not None and dts.size > 64:
            vals = np.abs(finufft.nufft1d3(2 * np.pi * fk, ck.astype(complex), dts, isign=-1, eps=1e-12))
        else:
            vals = np.concatenate([np.abs(np.exp(-2j * np.pi * np.outer(dts[i:i + 256], fk)) @ ck)
                                   for i in range(0, dts.size, 256)])
        i = int(np.argmax(vals))
        res = minimize_scalar(lambda t: -O(t), bounds=(dts[i] - step, dts[i] + step), method="bounded",
                              options={"xatol": step / 100})
        m, dt = (-res.fun, res.x) if -res.fun >= vals[i] else (vals[i], dts[i])
    m = float(m / (na * nb))
    return (m, float(dt)) if return_dt else m


def check_grid(f, dt_max):
    """Warn if the frequency grid cannot resolve the phase 2 pi f dt of time shifts or delays up to dt_max."""
    worst = np.max(np.diff(f)) * dt_max
    if worst > 0.1:
        warnings.warn(f"max(df) * dt_max = {worst:.2f} > 0.1: the frequency grid may not resolve time "
                      "shifts / lensing delays this large. Use a finer grid or a smaller dt_max.")


@dataclass
class ArraySetup:
    """
    The unlensed source and the analysis settings for any waveform code and noise curve, on any
    frequency grid (e.g. LISA with lisa_mbhb).

    source       (mchirp, q, chi1z, chi2z) of the injected binary, in the units template_fn expects
    f            frequency grid [Hz] (uniform or log-spaced)
    template_fn  template_fn(mchirp, q, chi1z, chi2z) -> h on the grid f: shape (n_f,), or
                 (n_channels, n_f) for stacked channels (e.g. TDI A, E); the extrinsic parameters
                 (sky position, distance, tc, ...) are fixed inside it
    psd          noise PSD on the grid: (n_f,) (e.g. the sky-averaged Sn(f)) or (n_channels, n_f)
    dt_max       largest time shift [s] searched by the match; it must exceed the lensing delays
                 (default: 0.1 / max(df), the largest shift the grid resolves)
    f_lower, f_upper   band where the lens factor is applied (default: the whole grid)
    bounds, sigma      template-parameter bounds and spread of the extra starts; default relative to
                 the source: mchirp in [0.5, 2] x mchirp_0, spread 1 %; q in [1, 20]; spins in +-0.99
    """
    source: tuple
    f: np.ndarray
    template_fn: object
    psd: np.ndarray
    dt_max: float = None
    f_lower: float = None
    f_upper: float = None
    convention: str = "lal"
    bounds: tuple = None
    sigma: tuple = None

    def __post_init__(self):
        self.f = np.asarray(self.f, dtype=float)
        if self.dt_max is None:
            self.dt_max = 0.1 / float(np.max(np.diff(self.f)))
        check_grid(self.f, self.dt_max)
        mc = float(self.source[0])
        if self.bounds is None:
            self.bounds = ((0.5 * mc, 2.0 * mc), (1.0, 20.0), (-0.99, 0.99), (-0.99, 0.99))
        if self.sigma is None:
            self.sigma = (0.01 * mc, 0.1, 0.05, 0.05)
        self.f_lower = float(self.f.min()) if self.f_lower is None else self.f_lower
        self.f_upper = float(self.f.max()) if self.f_upper is None else self.f_upper

    def template(self, theta):
        return np.asarray(self.template_fn(*theta))

    def lensed(self, h, model, params):
        m = get_lens_model(model)
        lens = {k: v for k, v in params.items() if k != "MLz"}
        return h * m.amplification(self.f, params["MLz"], self.f_lower, self.f_upper, self.convention, **lens)

    def match(self, a, b, psd=None):
        return grid_match(a, b, self.psd if psd is None else psd, self.f, self.dt_max)


# fitting factor
def lensed_signal(setup, lens_model, lens_params, h=None):
    """The source waveform multiplied by the lens model's factor (F = 1 outside [f_lower, f_upper]);
    lens_params holds MLz and the lens parameters."""
    h = setup.template(setup.source) if h is None else h
    return setup.lensed(h, lens_model, lens_params)


def _initial_simplex(x0, step, lo, hi):
    """
    Simplex with vertices x0 + step_i e_i (flipped inward at a bound).
    
    """
    simplex = [x0]
    for i in range(len(x0)):
        v = x0.copy()
        v[i] = x0[i] + step[i] if x0[i] + step[i] <= hi[i] else x0[i] - step[i]
        simplex.append(v)
    return np.array(simplex)


def fitting_factor(signal, setup, psd=None, x0=None, bounds=None, sigma=None, n_starts=6, seed=None,
                   xatol=1e-3, fatol=1e-6, maxiter=3000):
    """
    Maximise the match between `signal` and the setup's unlensed templates over (mchirp, q, chi1z, chi2z).
    The first Nelder-Mead start is x0 (default: the true source), the others x0 + N(0, sigma). The search
    runs in coordinates scaled by sigma, so xatol is in units of sigma (the same for a 30 Msun and a 1e6
    Msun binary). bounds / sigma default to the setup's. Returns (FF, best_theta, runs) with
    runs = [(match, theta), ...], best first.
    """
    bounds = setup.bounds if bounds is None else bounds
    sigma = np.asarray(setup.sigma if sigma is None else sigma, dtype=float)
    rng = np.random.default_rng(seed)
    lo, hi = np.array(bounds, dtype=float).T
    x0 = np.asarray(setup.source if x0 is None else x0, dtype=float)
    zlo, zhi = (lo - x0) / sigma, (hi - x0) / sigma                 # scaled coordinates z = (x - x0) / sigma

    def mismatch(z):
        try:
            h = setup.template(x0 + sigma * z)
        except Exception:                     # waveform failed at an edge of parameter space
            return 1.0
        return 1.0 - setup.match(signal, h, psd)

    runs = []
    for k in range(n_starts):
        start = np.clip(np.zeros(4) if k == 0 else rng.normal(0.0, 1.0, 4), zlo + 1e-9, zhi - 1e-9)
        res = minimize(mismatch, start, method="Nelder-Mead", bounds=list(zip(zlo, zhi)),
                       options={"xatol": xatol, "fatol": fatol, "maxiter": maxiter,
                                "initial_simplex": _initial_simplex(start, np.ones(4), zlo, zhi)})
        runs.append((1.0 - float(res.fun), np.clip(x0 + sigma * res.x, lo, hi)))
    runs.sort(key=lambda r: r[0], reverse=True)
    return runs[0][0], runs[0][1], runs


def lens_fitting_factor(setup, lens_model, lens_params, psd=None, **ff_kw):
    """FF for one lens: returns dict(FF, best (theta), match_unlensed (match of the lensed signal with the
    unlensed source itself), time)."""
    t0 = time.time()
    h0 = setup.template(setup.source)
    sig = setup.lensed(h0, lens_model, lens_params)
    m0 = setup.match(sig, h0, psd)
    ff, best, _ = fitting_factor(sig, setup, psd=psd, **ff_kw)
    return dict(FF=ff, best=best, match_unlensed=m0, time=time.time() - t0)


def _grid_point(args):
    setup, lens_model, params, ff_kw = args
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return lens_fitting_factor(setup, lens_model, params, **ff_kw)
        except Exception as e:                     # keep the grid going; report the failure
            return dict(FF=np.nan, best=np.full(4, np.nan), match_unlensed=np.nan, time=0.0, error=repr(e))


def _limit_threads(n):
    """
    Worker initializer:
    At most n threads for OpenMP (FINUFFT, LAL), BLAS and numexpr in this process.
    The environment variables cover libraries that start later; threadpoolctl also limits
    the ones already loaded.
    """
    global _LIMITS
    for v in _THREAD_ENV:
        os.environ[v] = str(n)
    try:
        from threadpoolctl import threadpool_limits
        _LIMITS = threadpool_limits(limits=n)
    except ImportError:
        pass


def fitting_factor_grid(setup, lens_model, grid, fixed=None, n_workers=1, threads_per_worker=1, outfile=None,
                        verbose=True, **ff_kw):
    """
    FF on the Cartesian grid of lens parameters `grid` = {name: values, ...} (axes in this order),
    with the remaining lens parameters in `fixed` = {name: value} (or the model's defaults). MLz may be
    on the grid or fixed, like any other parameter.

    setup       PyCBCSetup or ArraySetup (or any object with source, bounds, sigma, template(theta),
                lensed(h, model, params) and match(a, b, psd=None))
    n_workers   processes running grid points in parallel (None: all cores)
    threads_per_worker  threads each worker may use (OpenMP, BLAS); only applied with n_workers > 1
    outfile     .npz file: the results are saved after every point, and a rerun resumes from it
    ff_kw       passed to fitting_factor (n_starts, seed, bounds, sigma, xatol, ...)

    Returns a dict: 'axes' (names), one array per axis, 'FF', 'match_unlensed' (shape of the grid),
    'best' (grid shape + (4,): mchirp, q, spin1z, spin2z of the best template), 'fixed'.
    """
    model = get_lens_model(lens_model)
    fixed = dict(fixed or {})
    names = tuple(grid)
    axes = [np.atleast_1d(np.asarray(grid[n], dtype=float)) for n in names]
    missing = [p for p in model.all_parameters if p not in names and p not in fixed and p not in model.defaults]
    if missing:
        raise ValueError(f"lens model '{model.name}' needs {missing} on the grid or in `fixed` "
                         f"(parameters: {', '.join(model.all_parameters)})")
    unknown = [p for p in list(names) + list(fixed) if p not in model.all_parameters]
    if unknown:
        raise ValueError(f"{unknown} are not parameters of lens model '{model.name}' ({', '.join(model.all_parameters)})")

    shape = tuple(a.size for a in axes)
    FF, M0 = np.full(shape, np.nan), np.full(shape, np.nan)
    best, done = np.full(shape + (4,), np.nan), np.zeros(shape, dtype=bool)
    if outfile and os.path.exists(outfile):
        old = np.load(outfile, allow_pickle=True)
        if tuple(old["axes"]) == names and all(np.array_equal(old[n], a) for n, a in zip(names, axes)):
            FF, M0, best, done = old["FF"], old["match_unlensed"], old["best"], old["done"]
            if verbose:
                print(f"resuming from {outfile}: {done.sum()} of {done.size} points done")
        else:
            warnings.warn(f"{outfile} holds a different grid; starting afresh (it will be overwritten)")

    todo = [idx for idx in itertools.product(*(range(n) for n in shape)) if not done[idx]]
    if hasattr(setup, "get_psd"):
        setup.get_psd()                        # computed once here and shipped to the workers
    tasks = [(setup, model, {**fixed, **{n: float(a[i]) for n, a, i in zip(names, axes, idx)}}, dict(ff_kw))
             for idx in todo]

    def save():
        if outfile:
            np.savez(outfile, axes=np.array(names), FF=FF, match_unlensed=M0, best=best, done=done,
                     fixed=np.array([fixed], dtype=object), **{n: a for n, a in zip(names, axes)})

    def store(idx, task, r):
        FF[idx], M0[idx], best[idx], done[idx] = r["FF"], r["match_unlensed"], r["best"], True
        save()
        if verbose:
            p = ", ".join(f"{n}={task[2][n]:.4g}" for n in names)
            msg = r.get("error") or f"FF={r['FF']:.4f}  match(unlensed)={r['match_unlensed']:.4f}  [{r['time']:.0f} s]"
            print(f"[{done.sum()}/{done.size}] {p}: {msg}", flush=True)

    if n_workers == 1 or len(tasks) <= 1:
        for idx, task in zip(todo, tasks):
            store(idx, task, _grid_point(task))
    else:
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_limit_threads,
                                 initargs=(int(threads_per_worker),)) as pool:
            for idx, task, r in zip(todo, tasks, pool.map(_grid_point, tasks)):
                store(idx, task, r)
    out = dict(axes=names, FF=FF, match_unlensed=M0, best=best, fixed=fixed)
    out.update({n: a for n, a in zip(names, axes)})
    return out