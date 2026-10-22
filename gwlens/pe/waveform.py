import importlib
import inspect
from collections import OrderedDict
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
from bilby.core.utils import logger
from bilby.gw.waveform_generator import WaveformGenerator
from scipy.constants import G, c
from scipy.interpolate import BSpline, make_interp_spline

SOLAR_MASS = 1.98892e30  # kg

# w = 8 * pi * G * MLz[Msun] * f / c^3
POINT_LENS_W_CONSTANT = 8 * np.pi * G * SOLAR_MASS / c ** 3

DEFAULT_WAVEFORM_KWARGS = dict(
    waveform_approximant="IMRPhenomXAS",
    reference_frequency=50.0,
    minimum_frequency=20.0,
    catch_waveform_errors=False,
    pn_spin_order=-1,
    pn_tidal_order=-1,
    pn_phase_order=-1,
    pn_amplitude_order=0,
)


def _solver_module(solver):
    if solver == "1d":
        from gwlens import solver1d as module
    else:
        from gwlens import solver2d as module
    return module



@dataclass(frozen=True)
class LensModel:
    name: str
    solver: str                     # "1d" or "2d"
    build: object                   # build(m, *shape_parameters) -> Potential / Lens
    shape_parameters: tuple = ()
    axisymmetric: bool = True       # source position y (True) or y1, y2

    @property
    def position_parameters(self):
        return ("y",) if self.axisymmetric else ("y1", "y2")

    @property
    def parameters(self):
        return ("MLz",) + self.position_parameters + tuple(self.shape_parameters)

    def with_solver(self, solver):
        if solver is None or solver == self.solver:
            return self
        if solver == "2d" and self.axisymmetric:
            return replace(self, solver="2d")
        raise ValueError(f"lens model '{self.name}' is not axisymmetric: it needs the 2d solver")

    def position(self, params):
        """Source position as the solver expects it."""
        if not self.axisymmetric:
            return (float(params["y1"]), float(params["y2"]))
        y = abs(float(params["y"]))
        return y if self.solver == "1d" else (y, 0.0)

    def solver_box(self, ranges):
        """Source region for the solver from {parameter: [lo, hi]}."""
        if not self.axisymmetric:
            return tuple((float(ranges[p][0]), float(ranges[p][1])) for p in ("y1", "y2"))
        lo, hi = float(ranges["y"][0]), float(ranges["y"][1])
        return (lo, hi) if self.solver == "1d" else ((lo, hi), (0.0, 0.0))

    def make_lens(self, params):
        return self.build(_solver_module(self.solver), *[float(params[k]) for k in self.shape_parameters])


def _binary_point(m, lens_q, lens_separation):
    m1, m2 = 1.0 / (1.0 + lens_q), lens_q / (1.0 + lens_q)
    return m.point_mass(m1, (-m2 * lens_separation, 0.0)) + m.point_mass(m2, (m1 * lens_separation, 0.0))


LENS_MODELS = {
    "point": LensModel("point", "1d", lambda m: m.point_mass(1.0)),
    "sis": LensModel("sis", "1d", lambda m: m.sis(1.0)),
    "cis": LensModel("cis", "1d", lambda m, rc: m.cis(1.0, rc), ("rc",)),
    "nfw": LensModel("nfw", "1d", lambda m, kappa_s: m.nfw(1.0, 4.0 * kappa_s), ("kappa_s",)),
    "binary_point": LensModel("binary_point", "2d", _binary_point, ("lens_q", "lens_separation"), False),
    "point_shear": LensModel("point_shear", "2d",
                             lambda m, gamma_ext: m.point_mass(1.0) + m.external_shear(gamma_ext, 0.0),
                             ("gamma_ext",), False),
}


def get_lens_model(lens_model, solver=None):
    """LensModel for a built-in name, a dotted path to a lens function, a lens function, or a
    LensModel, computed with ``solver`` ("1d", "2d", or None/"auto" for the model's own)."""
    solver = None if solver in (None, "", "None", "auto") else str(solver).lower()
    if solver not in (None, "1d", "2d"):
        raise ValueError(f"lens solver must be '1d', '2d' or 'auto', not {solver!r}")
    if isinstance(lens_model, LensModel):
        return lens_model.with_solver(solver)
    if callable(lens_model):
        build, name = lens_model, f"{lens_model.__module__}.{lens_model.__qualname__}"
    else:
        name = str(lens_model).strip()
        if name.lower() in LENS_MODELS:
            return LENS_MODELS[name.lower()].with_solver(solver)
        if "." not in name:
            raise ValueError(f"Unknown lens model '{name}'. Built in: {', '.join(LENS_MODELS)}, "
                             "or give the dotted path of your lens function")
        module, attr = name.rsplit(".", 1)
        build = getattr(importlib.import_module(module), attr)
    if solver is None:
        raise ValueError(f"lens model '{name}': give the solver type (lens_solver = '1d' or '2d')")
    shape = tuple(list(inspect.signature(build).parameters)[1:])
    return LensModel(name, solver, build, shape, axisymmetric=(solver == "1d"))




def _coerce(value):
    """Undo bilby_pipe's string conversion of waveform-arguments-dict values."""
    if isinstance(value, dict):
        return {k: _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    if isinstance(value, str):
        s = value.strip().strip("'\"")
        if s.lower() in ("none", ""):
            return None
        if s.lower() in ("true", "false"):
            return s.lower() == "true"
        try:
            return float(s)
        except ValueError:
            return s
    return value


@dataclass
class LensingSettings:
    lens_model: object = None
    solver: str = None
    use_lookup_table: bool = False
    lookup_table: str = None
    ranges: dict = None
    time_origin: str = None           


def pop_lensing_settings(kwargs):
    """Remove the lens_* entries from a waveform-arguments dict and return them as settings."""
    raw = _coerce({k: kwargs.pop(k) for k in list(kwargs) if k.startswith("lens_")})
    return LensingSettings(raw.get("lens_model"), raw.get("lens_solver"),
                           bool(raw.get("lens_use_lookup_table") or False), raw.get("lens_lookup_table"),
                           raw.get("lens_ranges"), time_origin_setting(raw.get("lens_time_origin")))


def time_origin_setting(value):
    """None (T as it stands) or 'first_image', from what an ini or waveform arguments may hold."""
    if value in (None, False) or str(value).strip().strip("'\"").lower() in ("", "none", "false", "0"):
        return None
    v = str(value).strip().strip("'\"").lower().replace("-", "_")
    if v in ("first_image", "first", "minimum", "min"):
        return "first_image"
    raise ValueError(f"lens time origin must be 'none' or 'first_image', not {value!r}")



_SOLVERS = OrderedDict()
_WARNED = set()


def get_solver(model, params, w_range, box, time_origin=None):
    """Solver for one lens (fixed shape parameters), frequencies w_range and source region box;
    cached per process (the last 8)."""
    shape = tuple(float(params[k]) for k in model.shape_parameters)
    key = (model.name, model.solver, shape, tuple(np.round(w_range, 12)), repr(box), time_origin)
    if key in _SOLVERS:
        _SOLVERS.move_to_end(key)
        return _SOLVERS[key]
    m = _solver_module(model.solver)
    lens = model.make_lens(params)
    kw = dict(m.PE_SETTINGS, n_workers=1, interpolate="auto", interp_tol=1e-6, time_origin=time_origin)
    w_lo, w_hi = max(w_range[0], 1e-9 * w_range[1]), w_range[1]
    if model.solver == "1d":
        lo, hi = box
        s = m.Solver1D(lens, y=0.5 * (lo + hi), wmin=w_lo, wmax=w_hi, y_radius=0.5 * (hi - lo) + 1e-12, **kw)
    else:
        (a1, b1), (a2, b2) = box
        s = m.Solver2D(lens, (0.5 * (a1 + b1), 0.5 * (a2 + b2)), w_lo, w_hi,
                       y_radius=0.5 * np.hypot(b1 - a1, b2 - a2) + 1e-12, map_ext=1.0, **kw)
    _SOLVERS[key] = s
    if len(_SOLVERS) > 8:
        _SOLVERS.popitem(last=False)[1].close()
    return s


def _inside(pos, box):
    p, b = np.atleast_1d(pos), np.atleast_2d(box)
    return bool(np.all((p >= b[:, 0] - 1e-9) & (p <= b[:, 1] + 1e-9)))


def _from_solver(model, w, params, ranges, time_origin=None):
    pos = model.position(params)
    w_range = (float(w.min()), float(w.max()))
    single = (pos, pos) if model.solver == "1d" else ((pos[0], pos[0]), (pos[1], pos[1]))   # this position only
    box = None
    if ranges and any(ranges[p][0] != ranges[p][1] for p in model.shape_parameters if p in ranges):
        # sampled shape parameters: every call has a new lens, so its solver is never reused; build it for
        # this source position and these frequencies only (several times cheaper than the whole prior region)
        return np.conj(get_solver(model, params, w_range, single, time_origin).spectrum(w, pos))
    if ranges:
        W0, W1 = ranges["w"]
        if W0 * (1 - 1e-9) <= w_range[0] and w_range[1] <= W1 * (1 + 1e-9):
            w_range = (W0, W1)
        box = model.solver_box(ranges)
    if box is None or not _inside(pos, box):
        if ranges and "range" not in _WARNED:
            _WARNED.add("range")
            logger.warning(f"GWLENS: source position {pos} outside lens_ranges: extra solver built")
        box = single
    return np.conj(get_solver(model, params, w_range, box, time_origin).spectrum(w, pos))



def _bspline_basis(t, k, x):
    """First non-zero B-spline index at x and the k+1 non-zero values (Cox-de Boor)."""
    n = t.size - k - 1
    m = min(max(int(np.searchsorted(t, x, side="right")) - 1, k), n - 1)
    N, left, right = np.zeros(k + 1), np.zeros(k + 1), np.zeros(k + 1)
    N[0] = 1.0
    for j in range(1, k + 1):
        left[j], right[j] = x - t[m + 1 - j], t[m + j] - x
        saved = 0.0
        for r in range(j):
            tmp = N[r] / (right[r + 1] + left[j - r])
            N[r] = saved + right[r + 1] * tmp
            saved = left[j - r] * tmp
        N[j] = saved
    return m - k, N


class LookupTable:
    """F (LAL convention) on a grid (w, y, ...) or (w, y1, y2, ...), cubic spline in every axis. The
    axes after the source position are sampled shape parameters of the lens (e.g. rc); fixed ones
    are attributes.

    HDF5 file: <axis>_values for every axis, results (n_w, n_y, [n_y2,] [n_shape, ...] 2) with real
    and imaginary parts; attributes 'axes' (default ["w", "y"]), 'lens_model', and the fixed shape
    parameters of the lens.
    """

    def __init__(self, axes, coords, values, attrs=None, path=None):
        self.axes, self.attrs, self.path = list(axes), dict(attrs or {}), path
        self.coords = [np.asarray(x, dtype=float) for x in coords]
        C = np.asarray(values, dtype=complex)
        self.knots, self.degrees = [], []
        for j, x in enumerate(self.coords):
            if x.size == 1:
                self.knots.append(None)
                self.degrees.append(0)
                continue
            spl = make_interp_spline(x, np.moveaxis(C, j, 0), k=min(3, x.size - 1), axis=0)
            C = np.moveaxis(spl.c, 0, j)
            self.knots.append(spl.t)
            self.degrees.append(spl.k)
        self.C = np.ascontiguousarray(C)

    @classmethod
    def load(cls, path):
        import h5py
        import json
        with h5py.File(path, "r") as f:
            attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
            axes = json.loads(attrs["axes"]) if "axes" in attrs else ["w", "y"]
            coords = [f[f"{a}_values"][:] for a in axes]
            data = f["results"][:]
        return cls(axes, coords, data[..., 0] + 1j * data[..., 1], attrs, path)

    def _clip(self, j, v):
        lo, hi = self.coords[j][0], self.coords[j][-1]
        if (np.any(v < lo * (1 - 1e-9)) or np.any(v > hi * (1 + 1e-9))) and (self.path, j) not in _WARNED:
            _WARNED.add((self.path, j))
            logger.warning(f"GWLENS: {self.axes[j]} outside the lookup table [{lo:.4g}, {hi:.4g}]; clipped")
        return np.clip(v, lo, hi)

    def __call__(self, w, params):
        idx, weights = [slice(None)], []
        for j in range(1, len(self.axes)):
            if self.degrees[j] == 0:
                idx.append(slice(0, 1))
                weights.append(np.ones(1))
                continue
            i0, b = _bspline_basis(self.knots[j], self.degrees[j], float(self._clip(j, params[self.axes[j]])))
            idx.append(slice(i0, i0 + self.degrees[j] + 1))
            weights.append(b)
        sub = self.C[tuple(idx)]
        for b in reversed(weights):
            sub = sub @ b
        if self.degrees[0] == 0:
            return np.full(np.shape(w), sub[0], dtype=complex)
        return BSpline(self.knots[0], sub, self.degrees[0], extrapolate=False)(self._clip(0, np.asarray(w)))


@lru_cache(maxsize=4)
def load_lookup_table(path):
    return LookupTable.load(path)


def _from_table(model, w, params, path, time_origin=None):
    table = load_lookup_table(path)
    made = time_origin_setting(table.attrs.get("time_origin"))
    if made != time_origin:
        raise ValueError(f"lookup table {path} was made with time origin {made or 'none'}, not "
                         f"{time_origin or 'none'}; regenerate it (lensed-lookup-generation = True)")
    for name in model.shape_parameters:
        if name in table.axes:                               # sampled: interpolated like y
            continue
        if name not in table.attrs:
            raise ValueError(f"lookup table {path} has neither a fixed value nor an axis for {name}")
        if abs(float(table.attrs[name]) - float(params[name])) > 1e-6:
            raise ValueError(f"lookup table {path} was made for {name} = {table.attrs[name]}, not {params[name]}; "
                             f"to sample {name}, generate a table with {name} in the prior range")
    p = dict(params)
    if model.axisymmetric:
        p["y"] = abs(float(p["y"]))
    return table(w, p)



def amplification(frequency_array, lens_model, parameters, use_lookup_table=False, lookup_table=None,
                  ranges=None, solver=None, minimum_frequency=None, maximum_frequency=None, time_origin=None):
    """F(f) in the LAL convention (the waveform is multiplied by it); F = 1 outside
    [minimum_frequency, maximum_frequency] (where LAL sets the waveform to zero anyway)."""
    model = get_lens_model(lens_model, solver)
    f = np.asarray(frequency_array, dtype=float)
    F = np.ones(f.shape, dtype=complex)
    band = f > 0
    if minimum_frequency is not None:
        band &= f >= float(minimum_frequency)
    if maximum_frequency is not None:
        band &= f <= float(maximum_frequency)
    if not np.any(band):
        return F
    w = POINT_LENS_W_CONSTANT * float(parameters["MLz"]) * f[band]
    if use_lookup_table:
        F[band] = _from_table(model, w, parameters, lookup_table, time_origin)
    else:
        F[band] = _from_solver(model, w, parameters, ranges, time_origin)
    return F



def _base_function(base_model):
    """Unlensed source model from a function, a bilby.gw.source name or a dotted path."""
    if callable(base_model):
        return base_model
    import bilby
    name = str(base_model)
    fn = getattr(bilby.gw.source, name.replace("bilby.gw.source.", ""), None)
    if fn is None:
        module, attr = name.rsplit(".", 1)
        fn = getattr(importlib.import_module(module), attr)
    return fn


def lensed_source_model(model, base, settings):
    """bilby source function: base(frequency_array, ...) * F, with the lens parameters appended to the
    signature (bilby passes only the parameters that appear in it)."""
    P = inspect.Parameter
    base_params = [p.name for p in list(inspect.signature(base).parameters.values())[1:]
                   if p.kind in (P.POSITIONAL_OR_KEYWORD, P.KEYWORD_ONLY)]
    names = ["frequency_array"] + base_params + list(model.parameters)
    sig = inspect.Signature([P(n, P.POSITIONAL_OR_KEYWORD) for n in names] + [P("kwargs", P.VAR_KEYWORD)])

    def source_model(*args, **kwargs):
        bound = sig.bind(*args, **kwargs).arguments
        extra = {k: v for k, v in bound.pop("kwargs", {}).items() if not k.startswith("lens_")}
        f = bound.pop("frequency_array")
        lens_params = {k: bound.pop(k) for k in model.parameters}
        waveform_kwargs = dict(DEFAULT_WAVEFORM_KWARGS, maximum_frequency=f[-1])
        waveform_kwargs.update(extra)
        pol = base(f, **bound, **waveform_kwargs)
        if pol is None:
            return None
        F = amplification(f, model, lens_params, settings.use_lookup_table, settings.lookup_table,
                          settings.ranges, None, waveform_kwargs.get("minimum_frequency"),
                          waveform_kwargs.get("maximum_frequency"), settings.time_origin)
        return {key: value * F for key, value in pol.items()}

    source_model.__signature__ = sig
    source_model.__name__ = source_model.__qualname__ = f"{model.name}_lensed_{base.__name__}"
    source_model.__module__ = __name__
    return source_model


class LensedWaveformGenerator(WaveformGenerator):
    """bilby WaveformGenerator for any unlensed frequency-domain source model times F (see module doc).

    lens_model        built-in name, dotted path to a lens function, or the function
    lens_solver       "1d", "2d" or None (the model's own)
    use_lookup_table  True: interpolate lookup_table; False: compute F with the solver
    time_origin       None (T as it stands) or "first_image" (time from the first image's arrival)
    priors / ranges   region of the lens parameters ({parameter: [lo, hi], "w": [lo, hi]}); the solver is
                      built once for it. Without it, a solver is built for each new source position.
    Each can also be given in waveform_arguments with a lens_ prefix (as gwlens_pipe does).
    """
    LAL_SIM_INSPIRAL_SPINS_FLOW = 1

    def __init__(self, duration=None, sampling_frequency=None, start_time=0, frequency_domain_source_model=None,
                 time_domain_source_model=None, parameters=None, parameter_conversion=None,
                 waveform_arguments=None, lens_model=None, lens_solver=None, use_lookup_table=None,
                 lookup_table=None, priors=None, ranges=None, time_origin=None):
        wfa = dict(waveform_arguments or {})
        s = pop_lensing_settings(wfa)
        s.lens_model = lens_model if lens_model is not None else s.lens_model
        s.solver = lens_solver if lens_solver is not None else s.solver
        s.use_lookup_table = bool(use_lookup_table) if use_lookup_table is not None else s.use_lookup_table
        s.lookup_table = lookup_table if lookup_table is not None else s.lookup_table
        s.time_origin = time_origin_setting(time_origin) if time_origin is not None else s.time_origin
        if s.lens_model is None:
            raise ValueError("LensedWaveformGenerator needs lens_model")
        if s.use_lookup_table and not s.lookup_table:
            raise ValueError("use_lookup_table=True needs lookup_table")
        self.lens_model = get_lens_model(s.lens_model, s.solver)
        if priors is not None or ranges is not None:
            from .utils import lens_parameter_ranges
            f_hi = wfa.get("maximum_frequency") or (sampling_frequency / 2 if sampling_frequency else None)
            band = (float(wfa.get("minimum_frequency", 20.0)), float(f_hi)) if f_hi else None
            s.ranges = lens_parameter_ranges(self.lens_model, priors=priors, frequency_band=band, overrides=ranges)
        self.lensing = s
        self._base = _base_function(frequency_domain_source_model or "lal_binary_black_hole")
        if parameter_conversion is None and "black_hole" in self._base.__name__:
            from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters as parameter_conversion
        elif parameter_conversion is None and "neutron_star" in self._base.__name__:
            from bilby.gw.conversion import convert_to_lal_binary_neutron_star_parameters as parameter_conversion
        super().__init__(duration=duration, sampling_frequency=sampling_frequency, start_time=start_time,
                         frequency_domain_source_model=lensed_source_model(self.lens_model, self._base, s),
                         parameters=parameters, parameter_conversion=parameter_conversion, waveform_arguments=wfa)
        if self._base.__name__.startswith("lal_") and "waveform_approximant" in wfa and "reference_frequency" in wfa:
            from bilby.gw.waveform_generator import LALCBCWaveformGenerator
            LALCBCWaveformGenerator.validate_reference_frequency(self)

    def __getstate__(self):                # picklable: the lens model is stored by name / function
        state = self.__dict__.copy()
        m = self.lens_model
        state["lens_model"] = (m.name if m.name in LENS_MODELS or "." in m.name else m.build, m.solver)
        state["frequency_domain_source_model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.lens_model = get_lens_model(*state["lens_model"])
        self.frequency_domain_source_model = lensed_source_model(self.lens_model, self._base, self.lensing)

    def __repr__(self):
        mode = f"lookup_table={self.lensing.lookup_table}" if self.lensing.use_lookup_table else "solver"
        return (super().__repr__()[:-1] + f", lens_model={self.lens_model.name}, "
                f"lens_solver={self.lens_model.solver}, {mode})")