"""
gwlens_pipe utilities: lens-parameter ranges from the prior and the injection, and lookup tables.
"""
import json
import os
import time

import numpy as np

from .waveform import LookupTable, _coerce, _solver_module, get_lens_model, logger


# ================================================================== ini helpers
def parse_dict_option(value):
    """A dict-valued ini option (bilby_pipe keeps them as strings) as a dict."""
    if value in (None, "", "None"):
        return {}
    if isinstance(value, dict):
        return _coerce(dict(value))
    from bilby_pipe.utils import convert_string_to_dict
    return _coerce(convert_string_to_dict(str(value)) or {})


def _frequency(value, reduce):
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(reduce(float(v) for v in parse_dict_option(value).values()))


def frequency_band(args):
    """(f_min, f_max) of the analysis, as bilby_pipe sets them."""
    f_lo = _frequency(getattr(args, "minimum_frequency", None), min) or 20.0
    f_hi = _frequency(getattr(args, "maximum_frequency", None), max)
    return f_lo, f_hi or float(getattr(args, "sampling_frequency", 4096.0)) / 2


def read_prior_strings(args):
    """{name: prior string} from prior-dict or prior-file (lines only: other priors are not built)."""
    if getattr(args, "prior_dict", None) not in (None, "", "None"):
        from bilby_pipe.utils import convert_prior_string_input
        return {k.replace("-", "_"): v for k, v in convert_prior_string_input(args.prior_dict).items()}
    path = getattr(args, "prior_file", None)
    if path in (None, "", "None") or not os.path.isfile(str(path)):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            if "=" in line and not line.lstrip().startswith("#"):
                key, val = line.split("=", 1)
                out[key.strip()] = val.strip()
    return out


def read_injection_values(args):
    """Parameters of injection-dict (empty if not given)."""
    return parse_dict_option(getattr(args, "injection_dict", None))


def prior_range(name, spec):
    """(min, max) of a prior given as a number, a bilby Prior or a prior string."""
    import bilby
    if isinstance(spec, (int, float)):
        return float(spec), float(spec)
    p = spec if isinstance(spec, bilby.core.prior.Prior) else bilby.core.prior.PriorDict({name: spec})[name]
    if isinstance(p, bilby.core.prior.DeltaFunction):
        return float(p.peak), float(p.peak)
    return float(p.minimum), float(p.maximum)


def lens_parameter_ranges(lens_model, priors=None, frequency_band=None, injection=None, overrides=None):
    """{parameter: [lo, hi], 'w': [lo, hi]} from the priors, widened to include the injection values;
    overrides ({parameter: [lo, hi] or value}) take precedence. w from MLz and the frequency band."""
    model = get_lens_model(lens_model)
    priors, injection, overrides = priors or {}, injection or {}, _coerce(dict(overrides or {}))
    ranges = {}
    for name in model.parameters:
        lohi = None
        if name in overrides:
            v = np.atleast_1d(overrides[name]).astype(float)
            lohi = (v.min(), v.max())
        elif name in priors:
            lohi = prior_range(name, priors[name])
        if name in injection:
            v = float(injection[name])
            lohi = (v, v) if lohi is None else (min(lohi[0], v), max(lohi[1], v))
        if lohi is None:
            raise ValueError(f"GWLENS: lens parameter '{name}' of '{model.name}' is neither in the prior nor "
                             "in the injection (fixed values are fine, e.g. 'rc = 0.1')")
        ranges[name] = [float(lohi[0]), float(lohi[1])]
    if model.axisymmetric:                                  # F depends on |y| only
        lo, hi = ranges["y"]
        ranges["y"] = [0.0, max(-lo, hi)] if lo < 0 < hi else sorted([abs(lo), abs(hi)])
    if "w" in overrides:
        ranges["w"] = [float(v) for v in overrides["w"]]
    elif frequency_band is not None:
        from .waveform import POINT_LENS_W_CONSTANT as C
        ranges["w"] = [C * ranges["MLz"][0] * frequency_band[0], C * ranges["MLz"][1] * frequency_band[1]]
    return ranges


# ================================================================== lookup tables
def _w_grid(w_lo, w_hi, dw):
    """Geometric steps (10%) at small w, then steps of dw."""
    nodes = [w_lo]
    while nodes[-1] < w_hi:
        nodes.append(min(nodes[-1] + min(0.1 * nodes[-1], dw), w_hi))
    return np.array(nodes)


def generate_lookup_table(lens_model, ranges, path, n_workers=1, dw=0.05, dy=None, max_entries=5e7,
                          n_check=8):
    """Compute F (LAL convention) on a grid covering ``ranges`` and write it to ``path``.

    Grid: w with steps of 10% up to dw, then dw; each source-position axis with step dy (default
    min(0.01, 0.15 / w_max)). Shape parameters must be fixed (lo = hi in ranges). After the table is
    built, it is compared with the solver at n_check random source positions (200 frequencies each).
    """
    import h5py
    model = get_lens_model(lens_model)
    sampled = [p for p in model.shape_parameters if ranges[p][0] != ranges[p][1]]
    if sampled:
        raise ValueError(f"GWLENS: lookup tables need fixed shape parameters; {sampled} vary in the prior. "
                         "Fix them, or set use-lookup-table = False.")
    fixed = {p: float(ranges[p][0]) for p in model.shape_parameters}
    w_hi = float(ranges["w"][1])
    w_lo = max(float(ranges["w"][0]), 1e-6 * w_hi)
    dy = dy or min(0.01, 0.15 / w_hi)
    ws = _w_grid(w_lo, w_hi, dw)
    axes = []
    for p in model.position_parameters:
        lo, hi = ranges[p]
        axes.append(np.linspace(lo, hi, max(4, int(np.ceil((hi - lo) / dy)) + 1)) if hi > lo else np.array([lo]))
    shape = (ws.size,) + tuple(a.size for a in axes)
    if np.prod(shape) > max_entries:
        raise ValueError(f"GWLENS: lookup table of {shape} = {np.prod(shape):,} entries is too large "
                         f"(max_entries = {max_entries:,.0f}); use larger dw / dy in lens-lookup-kwargs, a "
                         "narrower prior, or use-lookup-table = False.")
    logger.info(f"GWLENS: computing lookup table {dict(zip(['w', *model.position_parameters], shape))}")
    t0 = time.perf_counter()

    m = _solver_module(model.solver)
    lens = model.make_lens(fixed)
    box = model.solver_box(ranges)
    if model.solver == "1d":
        solver = m.Solver1D(lens, y=0.5 * (box[0] + box[1]), wmin=w_lo, wmax=w_hi,
                            y_radius=0.5 * (box[1] - box[0]) + 1e-12, n_workers=n_workers, **m.PE_SETTINGS)
    else:
        (a1, b1), (a2, b2) = box
        solver = m.Solver2D(lens, (0.5 * (a1 + b1), 0.5 * (a2 + b2)), w_lo, w_hi,
                            y_radius=0.5 * np.hypot(b1 - a1, b2 - a2) + 1e-12, n_workers=n_workers, **m.PE_SETTINGS)
    names = list(model.position_parameters)
    with solver:
        grid = np.meshgrid(*axes, indexing="ij")
        P = np.array([model.position(dict(zip(names, v))) for v in zip(*(g.ravel() for g in grid))])
        values = np.conj(solver.evaluate(ws, P)).T.reshape(shape)
        # check against the solver at random points
        table = LookupTable(["w"] + names, [ws] + axes, values)
        rng = np.random.default_rng(1)
        err = 0.0
        for _ in range(n_check):
            p = {n: rng.uniform(*ranges[n]) for n in names}
            wt = np.sort(rng.uniform(w_lo, w_hi, 200))
            err = max(err, float(np.max(np.abs(table(wt, p) - np.conj(solver.spectrum(wt, model.position(p)))))))
    (logger.warning if err > 1e-3 else logger.info)(
        f"GWLENS: lookup table done in {time.perf_counter() - t0:.0f} s; max |F_table - F| = {err:.1e} at "
        f"{n_check} random positions (decrease dw / dy in lens-lookup-kwargs if too large)")

    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("w_values", data=ws)
        for n, a in zip(names, axes):
            f.create_dataset(f"{n}_values", data=a)
        f.create_dataset("results", data=np.stack([values.real, values.imag], axis=-1))
        f.attrs["axes"] = json.dumps(["w"] + names)
        f.attrs["lens_model"] = model.name
        f.attrs["convention"] = "lal"
        f.attrs["max_check_error"] = err
        for k, v in fixed.items():
            f.attrs[k] = v
    return path


def check_lookup_table(path, lens_model, ranges):
    """Problems (strings) if the table does not cover the ranges or was made for other parameters."""
    model = get_lens_model(lens_model)
    t = LookupTable.load(path)
    problems = []
    for name, x in zip(t.axes, t.coords):
        lo, hi = ranges.get(name, (x[0], x[-1]))
        if lo < x[0] * (1 - 1e-6) or hi > x[-1] * (1 + 1e-6):
            problems.append(f"{name} range [{lo:.4g}, {hi:.4g}] is not inside the table [{x[0]:.4g}, {x[-1]:.4g}]")
    for name in model.shape_parameters:
        if name in t.attrs and ranges[name] != [float(t.attrs[name])] * 2:
            problems.append(f"table made for {name} = {t.attrs[name]}, prior range {ranges[name]}")
    return problems