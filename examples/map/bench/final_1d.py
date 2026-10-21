"""GLoW vs gwlens (1D solver, axially symmetric lenses) runtime with the SAME number of CPU cores for both.
Companion of benchmark_same_cpus.py (2D solver): same tests, same CPU control, same plot.

  (a) fixed source position y, n frequencies                               (n = n_w)
  (b) fixed frequency w, n source positions                                (n = n_y)
  (c) K = 4 x NCPU source positions (near y), n frequencies each           (n = n_w)

    python benchmark_same_cpus_1d.py 1
    python benchmark_same_cpus_1d.py 8
    srun -c 8 --cpu-bind=cores python benchmark_same_cpus_1d.py 8          # on the cluster

How the core count is enforced (all of it happens BEFORE numpy / finufft / glow / gwlens are imported,
because OpenMP, OpenBLAS and numexpr read their settings when they are first loaded):
  * the process is pinned to NCPU cores (os.sched_setaffinity): no thread or worker process of either
    code can run anywhere else;
  * OMP_NUM_THREADS = NCPU (and the OPENBLAS / MKL / NUMEXPR limits): GLoW's OpenMP loops use at most
    NCPU threads; GLoW gets parallel = (NCPU > 1);
  * gwlens may use NCPU worker processes (n_workers=NCPU, used by evaluate() in (c)). A single gwlens
    call runs on ONE thread (threads=1, --gw-threads to change): the 1D transforms have ~10^4 points,
    and extra FINUFFT threads made a 1000-frequency spectrum 2-3x slower in our tests.
Each timing also reports "cores busy" = CPU time / wall time (this process, all its threads and its
finished worker processes).

What one timed call does
  (a) GLoW:   time-domain integral I(tau) for y (It_SingleIntegral_C by default), FFT, F at n frequencies.
      gwlens: amplification_many(ws, y): build the solver for [w_min, w_max], one spectrum.
  (b) GLoW:   for each of the n source positions: I(tau), FFT, F at the one frequency w. For
              n > --glow-max-ny this would take very long, so it is timed up to --glow-max-ny and
              continued as n x (time per source position); those points are drawn hollow.
      gwlens: amplification_profile(w, ys): one solver for the range of y, then F at all n positions.
  (c) GLoW:   (a) repeated for each of the K source positions (one after the other, each on NCPU threads).
      gwlens: one solver for the range of the K positions, then evaluate(ws, ys): the K spectra are
              computed by up to NCPU worker processes (one core each).
  All times are per call, i.e. for the whole task (all n or K source positions).

Same lens in both codes: before timing, F from both codes is compared at y and 60 frequencies (GLoW's
time origin is the first image: F_gwlens = F_GLoW exp(i w T_min)); the largest difference is printed
and shown in each panel. NOTE the NFW mapping: GLoW Psi_NFW(psi0, xs) = gwlens nfw(k=1/xs, psi0=psi0).
Source positions where GLoW fails are dropped for both codes and reported.
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("ncpu", type=int, nargs="?", default=int(os.environ.get("NCPU", 1)), help="number of CPU cores for both codes")
ap.add_argument("--repeats", type=int, default=5, help="timed repeats (fewer for calls longer than ~2 s)")
ap.add_argument("--number", type=int, default=5, help="calls per timed repeat (fewer for slow calls); times are per call")
ap.add_argument("--nw", type=int, nargs="+", default=[1, 10, 100, 1000, 10000], help="numbers of frequencies (a), (c)")
ap.add_argument("--ny", type=int, nargs="+", default=[1, 10, 100, 1000, 10000], help="numbers of source positions (b)")
ap.add_argument("--y-source", type=float, default=0.1, help="source position of (a) and centre of the positions of (c)")
ap.add_argument("--w-fixed", type=float, default=10, help="frequency of (b)")
ap.add_argument("--y-range", type=float, nargs=2, default=[0.1, 1.0], help="source positions of (b): linspace")
ap.add_argument("--glow-max-ny", type=int, default=1000, help="largest n_y timed for GLoW; larger n_y are extrapolated")
ap.add_argument("--n-batch", type=int, default=None, help="K, the number of source positions of (c) (default 4 x NCPU)")
ap.add_argument("--glow-method", choices=["SingleIntegral", "SingleContour"], default="SingleIntegral")
ap.add_argument("--gw-threads", type=int, default=1, help="FINUFFT threads of one gwlens call")
ap.add_argument("--pe", action="store_true", help="gwlens with PE_SETTINGS (accuracy ~1e-8) instead of the defaults (~1e-12)")
args = ap.parse_args()

NCPU = args.ncpu
_cores = sorted(os.sched_getaffinity(0))
if NCPU > len(_cores):
    raise SystemExit(f"asked for {NCPU} cores but only {len(_cores)} are available to this process")
os.sched_setaffinity(0, _cores[:NCPU])
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = str(NCPU)

import time                                         # noqa: E402  (imports only after the limits are set)
import numpy as np                                  # noqa: E402
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from gwlens import solver1d                         # noqa: E402
from glow import lenses, time_domain_c, freq_domain_c   # noqa: E402
try:
    from gwlens.tools import plotting               # noqa: E402
    plotting.apply_figure_style()
except ImportError:
    pass

GLOW_COLOR, GW_COLOR = "#4682a8", "#a0553f"
GW_multi_color = "#e08a00"
GLOW_multi_color = "#2a9d8f"
ItClass = getattr(time_domain_c, f"It_{args.glow_method}_C")


# --------------------------------------------------
# Lens builders: return (gwlens_lens, glow_lens) - the SAME lens in both codes
# --------------------------------------------------
def build_sis(psi0=1.0):
    return solver1d.sis(psi0), lenses.Psi_SIS(p_phys={'psi0': psi0})


def build_nfw(psi0=1.0, xs=0.1):
    # GLoW: psi0/2 [ln^2(u/2) + arccos^2(1/u)], u = x/xs;  gwlens: psi0/2 [ln^2(v/2) + arccos^2(1/v)], v = k x
    return solver1d.nfw(k=1.0 / xs, psi0=psi0), lenses.Psi_NFW(p_phys={'psi0': psi0, 'xs': xs})


def build_cis(psi0=1.0, rc=0.05):
    return solver1d.cis(psi0, rc), lenses.Psi_CIS(p_phys={'psi0': psi0, 'rc': rc})


models = [("SIS", build_sis), ("NFW", build_nfw), ("CIS", build_cis)]

# --------------------------------------------------
# Benchmark configuration
# --------------------------------------------------
y_source, w_fixed = args.y_source, args.w_fixed
ws_full = np.logspace(-4, 2, 1000)
P_T = {'tmin': 1e-2, 'tmax': 1e9, 'Nt': 5000, 'sampling': 'log', 'parallel': NCPU > 1}
P_FFT = {'wmin': 0.6 * min(ws_full.min(), w_fixed), 'wmax': max(ws_full.max(), w_fixed)}
# interp_tol must not be below the solver's own accuracy, or interpolation in w gives up (and costs time)
GW_ACC = dict(solver1d.PE_SETTINGS, interp_tol=1e-7) if args.pe else {}            # accuracy settings of every gwlens call
GW_KW = dict(n_workers=NCPU, threads=args.gw_threads, **GW_ACC)
SERIES = ("GLoW_w", "GWLENS_w", "GLoW_y", "GWLENS_y", "GLoW_yw", "GWLENS_yw")
K_BATCH = args.n_batch or 4 * NCPU
y_batch = np.linspace(y_source - 0.04, y_source + 0.04, K_BATCH) if K_BATCH > 1 else np.array([y_source])


def cpu_seconds():
    """CPU time used so far by this process (all its threads; high resolution) and by its finished child
    processes (the worker processes of evaluate(), joined when each solver is closed; 10 ms resolution)."""
    t = os.times()
    return time.process_time() + t.children_user + t.children_system


def benchmark_pipeline(func):
    """Median wall time per call, all per-call wall times, median busy cores. One untimed warm-up call;
    calls longer than ~0.2 s are repeated fewer times so that slow cases stay affordable."""
    t0 = time.perf_counter()
    func()
    t_warm = time.perf_counter() - t0
    number = max(1, min(args.number, int(1.0 / max(t_warm, 1e-9))))
    repeats = args.repeats if t_warm < 2.0 else min(args.repeats, 3)
    wall, cpu = [], []
    for _ in range(repeats):
        c0, t0 = cpu_seconds(), time.perf_counter()
        for _ in range(number):
            func()
        wall.append((time.perf_counter() - t0) / number)
        cpu.append((cpu_seconds() - c0) / number)
    wall, cpu = np.asarray(wall), np.asarray(cpu)
    return float(np.median(wall)), wall, float(np.median(cpu / wall))


def glow_It(lens_glow, y):
    return ItClass(lens_glow, max(float(y), 1e-6), dict(P_T))


def glow_F(lens_glow, y, ws):
    return freq_domain_c.Fw_FFT_C(glow_It(lens_glow, y), dict(P_FFT)).eval_Fw(ws)


_GLOW_OK = {}


def glow_usable(lens_name, lens_glow, ys):
    """Source positions at which GLoW runs. Each position is tried once (untimed); positions where GLoW
    raises are dropped for BOTH codes and reported, so that both always compute the same positions."""
    ok, dropped = [], []
    for y in np.atleast_1d(ys):
        key = (lens_name, round(float(y), 12))
        if key not in _GLOW_OK:
            try:
                glow_F(lens_glow, y, [w_fixed])
                _GLOW_OK[key] = True
            except Exception as err:                     # noqa: BLE001
                _GLOW_OK[key] = False
                print(f"  GLoW failed at y = {float(y):.6g} ({type(err).__name__}: {err}); position dropped for both codes")
        (ok if _GLOW_OK[key] else dropped).append(float(y))
    return np.array(ok), dropped


def timed(fn, label):
    """benchmark_pipeline(fn), or NaN (with a note) if the code raises during the timed calls."""
    try:
        med, _, cores = benchmark_pipeline(fn)
        return med, cores
    except Exception as err:                             # noqa: BLE001
        print(f"  {label}: failed during timing ({type(err).__name__}: {err}); recorded as NaN")
        return np.nan, np.nan


def same_lens_check(lens_gw, lens_glow):
    """max |F_gwlens - F_GLoW exp(i w T_min)| at y_source on 60 frequencies in [1e-2, 1e2]."""
    ws = np.logspace(-2, 2, 60)
    It = glow_It(lens_glow, y_source)
    Fg = freq_domain_c.Fw_FFT_C(It, dict(P_FFT)).eval_Fw(ws) * np.exp(1j * ws * It.tmin)
    return float(np.max(np.abs(solver1d.amplification_many(ws, y_source, lens_gw, **GW_ACC) - Fg)))


# --------------------------------------------------
# Run benchmarks
# --------------------------------------------------
print(f"NCPU = {NCPU}: pinned to cores {sorted(os.sched_getaffinity(0))}, OMP_NUM_THREADS = {os.environ['OMP_NUM_THREADS']}")
print(f"GLoW It_{args.glow_method}_C, parallel = {P_T['parallel']}; gwlens 1D {GW_KW}; (a) y = {y_source}; "
      f"(b) w = {w_fixed}, y in [{args.y_range[0]}, {args.y_range[1]}]; (c) K = {K_BATCH} positions in "
      f"[{y_batch.min():g}, {y_batch.max():g}]")
results = {}
for lens_name, builder in models:
    print(f"\n{lens_name}")
    lens_gw, lens_glow = builder()
    r = results[lens_name] = {k: [] for k in SERIES}
    r["cores"] = {k: [] for k in SERIES}
    r["GLoW_y_measured"] = []
    r["check"] = same_lens_check(lens_gw, lens_glow)
    print(f"  same lens in both codes: max |F_gwlens - F_GLoW| = {r['check']:.1e} at y = {y_source} "
          f"(GLoW's accuracy at these settings)")
    y_b, dropped_b = glow_usable(lens_name, lens_glow, y_batch)       # (c): same positions for both codes
    r["dropped"] = sorted(set(dropped_b))

    # (a) fixed source position, n frequencies  and  (c) K source positions, n frequencies each
    for n_w in args.nw:
        ws = np.logspace(-4, 2, n_w)
        for key, fn in (("GWLENS_w", lambda: solver1d.amplification_many(ws, y_source, lens_gw, interpolate=True, **GW_KW)),
                        ("GLoW_w", lambda: glow_F(lens_glow, y_source, ws))):
            med, cores = timed(fn, key)
            r[key].append(med)
            r["cores"][key].append(cores)
        print(f"  fixed y, n_w={n_w:6d} | GWLENS {r['GWLENS_w'][-1]:9.4g} s ({r['cores']['GWLENS_w'][-1]:.1f} cores) | "
              f"GLoW {r['GLoW_w'][-1]:9.4g} s ({r['cores']['GLoW_w'][-1]:.1f} cores)", flush=True)

        def gw_batch():
            wp = ws[ws > 0]
            with solver1d.Solver1D(lens_gw, y=0.5 * (y_b.min() + y_b.max()), wmin=wp.min(), wmax=wp.max(),
                                   y_radius=0.5 * (y_b.max() - y_b.min()) + 1e-9, n_workers=NCPU,
                                   threads=args.gw_threads, interpolate=True, **GW_ACC) as s:
                return s.evaluate(ws, y_b)
        for key, fn in (("GWLENS_yw", gw_batch), ("GLoW_yw", lambda: [glow_F(lens_glow, y, ws) for y in y_b])):
            med, cores = timed(fn, key)
            r[key].append(med)
            r["cores"][key].append(cores)
        print(f"  {y_b.size:3d} y x n_w={n_w:6d} | GWLENS {r['GWLENS_yw'][-1]:9.4g} s ({r['cores']['GWLENS_yw'][-1]:.1f} cores) | "
              f"GLoW {r['GLoW_yw'][-1]:9.4g} s ({r['cores']['GLoW_yw'][-1]:.1f} cores)", flush=True)

    # (b) fixed frequency, n source positions
    per_source = None
    for n_y in args.ny:
        y1 = np.linspace(args.y_range[0], args.y_range[1], n_y) if n_y > 1 else np.array([y_source])
        if n_y <= args.glow_max_ny:                      # GLoW is timed here: same usable positions for both
            y1, dropped = glow_usable(lens_name, lens_glow, y1)
            r["dropped"] = sorted(set(r["dropped"]) | set(dropped))
        med, cores = timed(lambda: solver1d.amplification_profile(w_fixed, y1, lens_gw, **GW_KW), "GWLENS_y")
        r["GWLENS_y"].append(med)
        r["cores"]["GWLENS_y"].append(cores)
        if n_y <= args.glow_max_ny:
            med, cores = timed(lambda: [glow_F(lens_glow, y, [w_fixed]) for y in y1], "GLoW_y")
            if np.isfinite(med):
                per_source = med / y1.size
            r["GLoW_y"].append(med)
            r["cores"]["GLoW_y"].append(cores)
            r["GLoW_y_measured"].append(True)
            note = f"({cores:.1f} cores)"
        else:
            r["GLoW_y"].append(n_y * per_source if per_source else np.nan)
            r["cores"]["GLoW_y"].append(np.nan)
            r["GLoW_y_measured"].append(False)
            note = "(= n_y x time per source)"
        print(f"  fixed w, n_y={n_y:6d} | GWLENS {r['GWLENS_y'][-1]:9.4g} s ({r['cores']['GWLENS_y'][-1]:.1f} cores) | "
              f"GLoW {r['GLoW_y'][-1]:9.4g} s {note}", flush=True)

np.savez(f"pipeline_benchmark_1d_{NCPU}cpu{'_pe' if args.pe else ''}.npz", ncpu=NCPU, n_w=np.array(args.nw), n_y=np.array(args.ny),
         w_fixed=w_fixed, y_source=y_source,
         **{f"{name}|{key}": np.array(v[key]) for name, v in results.items() for key in SERIES},
         **{f"{name}|{key}|cores": np.array(v["cores"][key]) for name, v in results.items() for key in SERIES},
         **{f"{name}|GLoW_y_measured": np.array(v["GLoW_y_measured"]) for name, v in results.items()},
         **{f"{name}|dropped_y": np.array(v["dropped"], dtype=float) for name, v in results.items()},
         **{f"{name}|same_lens_check": v["check"] for name, v in results.items()})
for name, v in results.items():
    if v["dropped"]:
        print(f"{name}: positions dropped because GLoW failed there: {', '.join(f'{y:.4g}' for y in v['dropped'])}")

# --------------------------------------------------
# Plot: 3 rows x 1 column, legend on top, one y label
# --------------------------------------------------
STYLE = {"GLoW_w": dict(color=GLOW_COLOR, marker="s", ls="-"),
         "GWLENS_w": dict(color=GW_COLOR, marker="o", ls="-"),
         "GLoW_y": dict(color=GLOW_COLOR, marker="s", ls="--"),
         "GWLENS_y": dict(color=GW_COLOR, marker="o", ls="--"),
         "GLoW_yw": dict(color=GLOW_multi_color, marker="D", ls=":"),
         "GWLENS_yw": dict(color=GW_multi_color, marker="D", ls=":")}
LABEL = {"GLoW_w": f"GLoW: fixed $y = {y_source:g}$, $n$ frequencies",
         "GWLENS_w": f"GWLENS: fixed $y = {y_source:g}$, $n$ frequencies",
         "GLoW_y": f"GLoW: fixed $w = {w_fixed:g}$, $n$ source positions",
         "GWLENS_y": f"GWLENS: fixed $w = {w_fixed:g}$, $n$ source positions",
         "GLoW_yw": f"GLoW: {K_BATCH} source positions, $n$ frequencies each",
         "GWLENS_yw": f"GWLENS: {K_BATCH} source positions, $n$ frequencies each"}
fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(6, 8), sharex=True, layout="constrained")
for ax, (lens_name, _) in zip(axes, models):
    r = results[lens_name]
    for key, n in (("GLoW_w", args.nw), ("GWLENS_w", args.nw), ("GWLENS_y", args.ny),
                   ("GLoW_yw", args.nw), ("GWLENS_yw", args.nw)):
        ax.loglog(n, r[key], lw=2, ms=6, **STYLE[key])
    meas = np.array(r["GLoW_y_measured"], dtype=bool)
    ny, ty = np.array(args.ny), np.array(r["GLoW_y"])
    st = dict(STYLE["GLoW_y"])
    ax.loglog(ny, ty, lw=2, color=st["color"], ls="--")
    ax.loglog(ny[meas], ty[meas], ls="none", marker="s", ms=6, color=st["color"])
    ax.loglog(ny[~meas], ty[~meas], ls="none", marker="s", ms=6, mfc="white", mec=st["color"], mew=1.5)
    ax.text(0.02, 0.95, lens_name, transform=ax.transAxes, ha="left", va="top", fontsize=14, fontweight="bold")
    #note = f"same lens: max $|F_\\mathrm{{gwlens}} - F_\\mathrm{{GLoW}}|$ = {r['check']:.0e}"
    #if r["dropped"]:
    #    note += "\nGLoW failed at y = " + ", ".join(f"{y:.3g}" for y in r["dropped"]) + " (excluded)"
    #ax.text(0.02, 0.80, note, transform=ax.transAxes, ha="left", va="top", fontsize=14, color="0.35")
    ax.grid(True, which="both", alpha=0.3)
    def _tick_label(k):
        e = int(round(np.log10(k)))
        return rf"$10^{{{e}}}$" if k == 10**e else str(k)

    nn = sorted(set(args.nw) | set(args.ny))
    ax.set_xticks(nn)
    ax.set_xticklabels([_tick_label(k) for k in nn], fontsize=14)
axes[-1].set_xlabel(r"$n$")
fig.supylabel(f"Median time per call (s), {NCPU} CPU core{'s' if NCPU > 1 else ''} each")
# legend columns: GLoW on the left, gwlens on the right (matplotlib fills legends column by column)
handles = [Line2D([], [], lw=2, ms=6, **STYLE[k]) for k in ("GLoW_w", "GLoW_y", "GLoW_yw")]
handles.append(Line2D([], [], ls="none", marker="s", ms=6, mfc="white", mec=GLOW_COLOR, mew=1.5))
handles += [Line2D([], [], lw=2, ms=6, **STYLE[k]) for k in ("GWLENS_w", "GWLENS_y", "GWLENS_yw")]
labels = [LABEL["GLoW_w"], LABEL["GLoW_y"], LABEL["GLoW_yw"], r"GLoW: $n\,\times$ (time per source position), not timed",
          LABEL["GWLENS_w"], LABEL["GWLENS_y"], LABEL["GWLENS_yw"]]
#fig.legend(handles, labels, loc="upper center", ncol=1, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 1.0))

# outside legend: constrained layout reserves the space above the axes
fig.legend(handles, labels, loc="outside upper center", ncol=1, frameon=False,
           fontsize=14, handlelength=2.5, columnspacing=2.5, handletextpad=0.6)

#fig.tight_layout(rect=(0.02, 0, 1, 0.92))
fig.savefig(f"pipeline_1dbenchmark_{NCPU}cpu{'_pe' if args.pe else ''}.png", dpi=300, bbox_inches="tight")
fig.savefig(f"pipeline_1dbenchmark_{NCPU}cpu{'_pe' if args.pe else ''}.pdf", bbox_inches="tight")
print(f"\nsaved pipeline_1dbenchmark_{NCPU}cpu{'_pe' if args.pe else ''}.png and .npz")