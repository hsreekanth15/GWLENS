"""
gwlens_pipe utilities: lens-parameter ranges from the prior and the injection, and lookup tables.
"""
import json
import os
import time

import numpy as np

from .waveform import LookupTable, _coerce, _solver_module, get_lens_model, logger, time_origin_setting


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


def _make_solver(model, lens, ranges, w_lo, w_hi, n_workers=1, **kw):
    """Solver for the source region of ``ranges`` and the frequencies [w_lo, w_hi] (PE settings)."""
    m = _solver_module(model.solver)
    box = model.solver_box(ranges)
    if model.solver == "1d":
        return m.Solver1D(lens, y=0.5 * (box[0] + box[1]), wmin=w_lo, wmax=w_hi,
                          y_radius=0.5 * (box[1] - box[0]) + 1e-12, n_workers=n_workers, **m.PE_SETTINGS, **kw)
    (a1, b1), (a2, b2) = box
    return m.Solver2D(lens, (0.5 * (a1 + b1), 0.5 * (a2 + b2)), w_lo, w_hi,
                      y_radius=0.5 * np.hypot(b1 - a1, b2 - a2) + 1e-12, n_workers=n_workers, **m.PE_SETTINGS, **kw)


def _shape_step(model, ranges, fixed, name, w_hi, n_probe=5):
    """Default grid step of a sampled shape parameter: the phase step w_hi * dy of the default source-
    position grid (dy = min(0.01, 0.15 / w_hi); the time delays move ~1 per unit y), divided by S = how
    fast the time delays of the images (and of a core) move with the parameter at a few source
    positions; and at least 8 intervals over its range."""
    lo, hi = ranges[name]
    vals = np.linspace(lo, hi, n_probe)
    pos_names = list(model.position_parameters)
    probes = [dict(zip(pos_names, v)) for v in
              np.array(np.meshgrid(*[np.unique([ranges[q][0], 0.5 * sum(ranges[q]), ranges[q][1]])
                                     for q in pos_names], indexing="ij")).reshape(len(pos_names), -1).T]
    T = []
    for v in vals:
        s = _make_solver(model, model.make_lens(dict(fixed, **{name: v})), ranges, w_hi, w_hi)
        T.append([s._time_delay_range(np.asarray(model.position(q)) if model.solver == "2d" else model.position(q))
                  for q in probes])
        s.close()
    T = np.array(T)                                                # (n_probe, positions, 2)
    S = float(np.max(np.abs(np.diff(T, axis=0)) / np.diff(vals)[:, None, None]))
    step = (hi - lo) / 8
    return min(step, min(0.01, 0.15 / w_hi) / S) if S > 0 else step


def generate_lookup_table(lens_model, ranges, path, n_workers=1, dw=0.05, dy=None, max_entries=5e7,
                          n_check=8, time_origin=None, **shape_steps):
    """Compute F (LAL convention) on a grid covering ``ranges`` and write it to ``path``.

    Grid: w with steps of 10% up to dw, then dw; each source-position axis with step dy (default
    min(0.01, 0.15 / w_max)). Shape parameters that are fixed (lo = hi in ranges) are stored as
    attributes; sampled ones (e.g. rc of the cis lens) become further axes of the table, with step
    d<name> (e.g. drc = 0.005; default: the phase step of the default dy, from how fast the image time
    delays move with the parameter; at least 8 intervals). One solver is built per grid value of the
    sampled shape parameters. After the table is built, it is compared with the solver at n_check
    random points (200 frequencies each). time_origin: None or "first_image" (see waveform.py); stored in
    the file, and a table is only used with the same setting.
    """
    import h5py
    import itertools
    time_origin = time_origin_setting(time_origin)
    model = get_lens_model(lens_model)
    unknown = [k for k in shape_steps if not (k.startswith("d") and k[1:] in model.shape_parameters)]
    if unknown:
        raise ValueError(f"GWLENS: unknown lens-lookup-kwargs {unknown}; grid steps of the shape parameters "
                         f"are d<name> for {list(model.shape_parameters)}")
    sampled = [p for p in model.shape_parameters if ranges[p][0] != ranges[p][1]]
    fixed = {p: float(ranges[p][0]) for p in model.shape_parameters if p not in sampled}
    w_hi = float(ranges["w"][1])
    w_lo = max(float(ranges["w"][0]), 1e-6 * w_hi)
    dy = dy or min(0.01, 0.15 / w_hi)
    ws = _w_grid(w_lo, w_hi, dw)
    names = list(model.position_parameters)
    axes = []
    for p in names:
        lo, hi = ranges[p]
        axes.append(np.linspace(lo, hi, max(4, int(np.ceil((hi - lo) / dy)) + 1)) if hi > lo else np.array([lo]))
    mid = dict(fixed, **{p: 0.5 * sum(ranges[p]) for p in sampled})
    shape_axes = []
    for p in sampled:
        lo, hi = ranges[p]
        step = float(shape_steps.get(f"d{p}") or _shape_step(model, ranges, mid, p, w_hi))
        shape_axes.append(np.linspace(lo, hi, max(4, int(np.ceil((hi - lo) / step)) + 1)))
    shape = (ws.size,) + tuple(a.size for a in axes + shape_axes)
    if np.prod(shape) > max_entries:
        raise ValueError(f"GWLENS: lookup table of {shape} = {np.prod(shape):,} entries is too large "
                         f"(max_entries = {max_entries:,.0f}); use larger dw / dy"
                         f"{''.join(f' / d{p}' for p in sampled)} in lens-lookup-kwargs, a narrower prior, "
                         "or use-lookup-table = False.")
    all_names = ["w"] + names + sampled
    logger.info(f"GWLENS: computing lookup table {dict(zip(all_names, shape))}"
                + (f", one solver for each of {int(np.prod([a.size for a in shape_axes]))} values of {sampled}"
                   if sampled else ""))
    t0 = time.perf_counter()

    grid = np.meshgrid(*axes, indexing="ij")
    P = np.array([model.position(dict(zip(names, v))) for v in zip(*(g.ravel() for g in grid))])
    pos_shape = shape[:1 + len(names)]
    values = np.empty(shape, dtype=complex)
    for idx in itertools.product(*[range(a.size) for a in shape_axes]):
        params = dict(fixed, **{p: float(a[k]) for p, a, k in zip(sampled, shape_axes, idx)})
        with _make_solver(model, model.make_lens(params), ranges, w_lo, w_hi, n_workers,
                          time_origin=time_origin) as solver:
            values[(Ellipsis,) + idx] = np.conj(solver.evaluate(ws, P)).T.reshape(pos_shape)

    # check against the solver at random points
    table = LookupTable(all_names, [ws] + axes + shape_axes, values)
    rng = np.random.default_rng(1)
    err = 0.0
    for _ in range(n_check):
        p = {n: rng.uniform(*ranges[n]) for n in names + sampled}
        wt = np.sort(rng.uniform(w_lo, w_hi, 200))
        with _make_solver(model, model.make_lens(dict(fixed, **{q: p[q] for q in sampled})), ranges, w_lo, w_hi,
                          time_origin=time_origin) as s:
            err = max(err, float(np.max(np.abs(table(wt, p) - np.conj(s.spectrum(wt, model.position(p)))))))
    (logger.warning if err > 1e-3 else logger.info)(
        f"GWLENS: lookup table done in {time.perf_counter() - t0:.0f} s; max |F_table - F| = {err:.1e} at "
        f"{n_check} random points (decrease dw / dy{''.join(f' / d{p}' for p in sampled)} in "
        "lens-lookup-kwargs if too large)")

    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("w_values", data=ws)
        for n, a in zip(names + sampled, axes + shape_axes):
            f.create_dataset(f"{n}_values", data=a)
        f.create_dataset("results", data=np.stack([values.real, values.imag], axis=-1))
        f.attrs["axes"] = json.dumps(all_names)
        f.attrs["lens_model"] = model.name
        f.attrs["convention"] = "lal"
        f.attrs["max_check_error"] = err
        f.attrs["time_origin"] = time_origin or "none"
        for k, v in fixed.items():
            f.attrs[k] = v
    return path


def check_lookup_table(path, lens_model, ranges, time_origin=None):
    """Problems (strings) if the table does not cover the ranges or was made for other parameters."""
    model = get_lens_model(lens_model)
    t = LookupTable.load(path)
    problems = []
    made = time_origin_setting(t.attrs.get("time_origin"))
    if made != time_origin_setting(time_origin):
        problems.append(f"table made with time origin {made or 'none'}, run uses {time_origin or 'none'}")
    for name, x in zip(t.axes, t.coords):
        lo, hi = ranges.get(name, (x[0], x[-1]))
        if lo < x[0] * (1 - 1e-6) or hi > x[-1] * (1 + 1e-6):
            problems.append(f"{name} range [{lo:.4g}, {hi:.4g}] is not inside the table [{x[0]:.4g}, {x[-1]:.4g}]")
    for name in model.shape_parameters:
        if name in t.attrs and ranges[name] != [float(t.attrs[name])] * 2:
            problems.append(f"table made for {name} = {t.attrs[name]}, prior range {ranges[name]}")
        elif name not in t.attrs and name not in t.axes:
            problems.append(f"table has neither a fixed value nor an axis for {name}")
    return problems