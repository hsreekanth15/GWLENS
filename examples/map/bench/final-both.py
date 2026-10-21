"""GLoW vs gwlens (2D solver) runtime with the SAME number of CPU cores for both, in three settings:

  (a) fixed source position y, n frequencies                          (n = n_w)
  (b) fixed frequency w, n source positions on the x1-axis            (n = n_y)
  (c) K = 4 x NCPU source positions (near y, on the x1-axis), n frequencies each   (n = n_w)

    python benchmark_same_cpus.py 1
    python benchmark_same_cpus.py 8

On the cluster, reserve the cores so nobody else uses them:
    srun -c 8 --cpu-bind=cores python benchmark_same_cpus.py 8
    for n in 1 2 4 8 16; do srun -c $n --cpu-bind=cores python benchmark_same_cpus.py $n; done

How the core count is enforced (all of it happens BEFORE numpy / finufft / glow / gwlens are imported,
because OpenMP, OpenBLAS and numexpr read their settings when they are first loaded):
  * the process is pinned to NCPU cores (os.sched_setaffinity): no thread of either code can run
    anywhere else, whatever it asks for;
  * OMP_NUM_THREADS = NCPU: GLoW's OpenMP loops (parallel=True) and FINUFFT use at most NCPU threads;
    OPENBLAS / MKL / NUMEXPR limits are set to the same value;
  * gwlens gets threads=NCPU (FINUFFT/numexpr threads of one transform) and n_workers=NCPU (threads
    that evaluate psi while the grid is built); GLoW gets parallel = (NCPU > 1).
Each timing also reports "cores busy" = CPU time / wall time (this process and all its threads), so you
can see how many of the NCPU cores each code actually used.

What one timed call does
  (a) GLoW:   time-domain integral I(tau) for y, FFT, F at the n frequencies.
      gwlens: amplification_many(ws, y): build the grid for [w_min, w_max], one spectrum.
  (b) GLoW:   for each of the n source positions: I(tau), FFT, F at the one frequency w (GLoW has
              nothing to reuse between source positions). For n > --glow-max-ny this would take hours,
              so it is measured up to --glow-max-ny and continued as n x (time per source position);
              those points are drawn hollow.
      gwlens: amplification_map(w, y1, y2): build one grid for the disc containing all n positions,
              then one 2D non-uniform FFT gives F at all of them.
  (c) GLoW:   (a) repeated for each of the K source positions (one after the other, each on NCPU threads).
      gwlens: one grid for the disc containing the K positions, then evaluate(ws, ys): the K spectra are
              computed by NCPU worker processes, one core each (n_workers=NCPU, threads=1). This is the
              "many sources, many frequencies" case of parameter estimation or lookup tables.
  All times are per call, i.e. for the whole task (all n or K source positions).
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("ncpu", type=int, nargs="?", default=int(os.environ.get("NCPU", 1)), help="number of CPU cores for both codes")
ap.add_argument("--repeats", type=int, default=5, help="timed repeats (fewer for calls longer than ~2 s)")
ap.add_argument("--number", type=int, default=5, help="calls per timed repeat (fewer for slow calls); times are per call")
ap.add_argument("--nw", type=int, nargs="+", default=[1, 10, 100, 1000, 10000], help="numbers of frequencies (a)")
ap.add_argument("--ny", type=int, nargs="+", default=[1, 10, 100, 1000, 10000], help="numbers of source positions (b)")
ap.add_argument("--y-source", type=float, default=0.1, help="source position of (a), on the x1-axis")
ap.add_argument("--w-fixed", type=float, default=10.0, help="frequency of (b)")
ap.add_argument("--y-range", type=float, nargs=2, default=[0.1, 1.0], help="source positions of (b): linspace on the x1-axis")
ap.add_argument("--glow-max-ny", type=int, default=1000, help="largest n_y timed for GLoW; larger n_y are extrapolated")
ap.add_argument("--n-batch", type=int, default=None, help="K, the number of source positions of (c) (default 4 x NCPU)")
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
from gwlens import  solver2d         # noqa: E402  the new lens2d_general.py
from glow import lenses, time_domain_c, freq_domain_c   # noqa: E402
try:
    from gwlens.tools import plotting               # noqa: E402
    plotting.apply_figure_style()
except ImportError:
    pass

GLOW_COLOR, GW_COLOR = "#4682a8", "#a0553f"
GW_multi_color = "#e08a00"
GLOW_multi_color = "#2a9d8f"

def distribute_centers(n, R=1.0, seed=0):
    """Identical recipe to glow_worker_msis.distribute_centers (own RNG, centre of mass at 0)."""
    rng = np.random.RandomState(seed)
    Rs = R * np.sqrt(rng.random_sample(n))
    ths = 2 * np.pi * rng.random_sample(n)
    x1, x2 = Rs * np.cos(ths), Rs * np.sin(ths)
    return x1 - x1.mean(), x2 - x2.mean()


def build_4sis(seed=1235, R=1.0, n=4):
    xc1, xc2 = distribute_centers(n, R, seed)
    gw = None
    for a, b in zip(xc1, xc2):
        p = solver2d.sis(1.0 / n, (a, b))
        gw = p if gw is None else gw + p
    gl = lenses.CombinedLens({'lenses': [
        lenses.Psi_offcenterSIS(p_phys={'psi0': 1.0 / n, 'xc1': a, 'xc2': b})
        for a, b in zip(xc1, xc2)]})
    return gw, gl


def build_4cis(rc=0.05, n=4):
    cen = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    gw = sum((solver2d.cis(1.0 / n, rc, c) for c in cen[1:]),
             solver2d.cis(1.0 / n, rc, cen[0]))
    gl = lenses.CombinedLens({'lenses': [
        lenses.Psi_offcenterCIS(p_phys={'psi0': 1.0 / n, 'rc': rc, 'xc1': a, 'xc2': b})
        for a, b in cen]})
    return gw, gl


def build_binary():
    m, cen = [0.5, 0.5], [(-0.5, 0.0), (0.5, 0.0)]
    gw = solver2d.point_mass(m[0], cen[0]) + solver2d.point_mass(m[1], cen[1])
    gl = lenses.CombinedLens({'lenses': [
        lenses.Psi_offcenterPointLens(p_phys={'psi0': mi, 'xc1': a, 'xc2': b})
        for mi, (a, b) in zip(m, cen)]})
    return gw, gl


# --------------------------------------------------
# Benchmark configuration
# --------------------------------------------------
y_source, w_fixed = args.y_source, args.w_fixed
ws_full = np.logspace(-4, 2, 1000)
P_T = {'tmin': 1e-2, 'tmax': 1e9, 'Nt': 5000, 'sampling': 'log', 'parallel': NCPU > 1}
P_FFT = {'wmin': 0.6 * min(ws_full.min(), w_fixed), 'wmax': max(ws_full.max(), w_fixed)}
GW_KW = dict(n_workers=NCPU, threads=NCPU, nlag=16,ngl=24)
models = [("4 SIS", build_4sis), ("4 CIS", build_4cis), ("Binary lens", build_binary)]
SERIES = ("GLoW_w", "GWLENS_w", "GLoW_y", "GWLENS_y", "GLoW_yw", "GWLENS_yw")
K_BATCH = args.n_batch or 4 * NCPU
y_batch = np.linspace(y_source - 0.04, y_source + 0.04, K_BATCH) if K_BATCH > 1 else np.array([y_source])
ys_batch = np.column_stack([y_batch, np.zeros_like(y_batch)])


def cpu_seconds():
    """CPU time used so far by this process (all its threads) and by its finished child processes."""
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


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


def glow_F(lens_glow, y, ws):
    It = time_domain_c.It_MultiContour_C(lens_glow, max(float(y), 1e-6), dict(P_T))
    return freq_domain_c.Fw_FFT_C(It, dict(P_FFT)).eval_Fw(ws)


# --------------------------------------------------
# Run benchmarks
# --------------------------------------------------
print(f"NCPU = {NCPU}: pinned to cores {sorted(os.sched_getaffinity(0))}, OMP_NUM_THREADS = {os.environ['OMP_NUM_THREADS']}")
print(f"GLoW parallel = {P_T['parallel']}; gwlens {GW_KW}; (a) y = {y_source}; (b) w = {w_fixed}, "
      f"y in [{args.y_range[0]}, {args.y_range[1]}] on the x1-axis; (c) K = {K_BATCH} positions in "
      f"[{y_batch.min():g}, {y_batch.max():g}] on the x1-axis")
results = {}
for lens_name, builder in models:
    print(f"\n{lens_name}")
    lens_gw, lens_glow = builder()
    r = results[lens_name] = {k: [] for k in SERIES}
    r["cores"] = {k: [] for k in SERIES}
    r["GLoW_y_measured"] = []

    # (a) fixed source position, n frequencies
    for n_w in args.nw:
        ws = np.logspace(-4, 2, n_w)
        for key, fn in (("GWLENS_w", lambda: solver2d.amplification_many(ws, (y_source, 0.0), lens_gw, interpolate=True, **GW_KW)),
                        ("GLoW_w", lambda: glow_F(lens_glow, y_source, ws))):
            med, _, cores = benchmark_pipeline(fn)
            r[key].append(med)
            r["cores"][key].append(cores)
        print(f"  fixed y, n_w={n_w:6d} | GWLENS {r['GWLENS_w'][-1]:9.4g} s ({r['cores']['GWLENS_w'][-1]:.1f} cores) | "
              f"GLoW {r['GLoW_w'][-1]:9.4g} s ({r['cores']['GLoW_w'][-1]:.1f} cores)", flush=True)

        # (c) K source positions, n frequencies each
        def gw_batch():
            wp = ws[ws > 0]
            with solver2d.Solver2D(lens_gw, (y_source, 0.0), wp.min(), wp.max(), y_radius=0.04 + 1e-9,
                                   n_workers=NCPU, threads=1, nlag=16, interpolate=True) as s:
                return s.evaluate(ws, ys_batch)
        for key, fn in (("GWLENS_yw", gw_batch), ("GLoW_yw", lambda: [glow_F(lens_glow, y, ws) for y in y_batch])):
            med, _, cores = benchmark_pipeline(fn)
            r[key].append(med)
            r["cores"][key].append(cores)
        print(f"  {K_BATCH:3d} y x n_w={n_w:6d} | GWLENS {r['GWLENS_yw'][-1]:9.4g} s ({r['cores']['GWLENS_yw'][-1]:.1f} cores) | "
              f"GLoW {r['GLoW_yw'][-1]:9.4g} s ({r['cores']['GLoW_yw'][-1]:.1f} cores)", flush=True)

    # (b) fixed frequency, n source positions
    per_source = None
    for n_y in args.ny:
        y1 = np.linspace(args.y_range[0], args.y_range[1], n_y) if n_y > 1 else np.array([y_source])
        y2 = np.zeros_like(y1)
        med, _, cores = benchmark_pipeline(lambda: solver2d.amplification_map(w_fixed, y1, y2, lens_gw, **GW_KW))
        r["GWLENS_y"].append(med)
        r["cores"]["GWLENS_y"].append(cores)
        if n_y <= args.glow_max_ny:
            med, _, cores = benchmark_pipeline(lambda: [glow_F(lens_glow, y, [w_fixed]) for y in y1])
            per_source = med / n_y
            r["GLoW_y"].append(med)
            r["cores"]["GLoW_y"].append(cores)
            r["GLoW_y_measured"].append(True)
            note = f"({cores:.1f} cores)"
        else:
            r["GLoW_y"].append(n_y * per_source)
            r["cores"]["GLoW_y"].append(np.nan)
            r["GLoW_y_measured"].append(False)
            note = "(= n_y x time per source)"
        print(f"  fixed w, n_y={n_y:6d} | GWLENS {r['GWLENS_y'][-1]:9.4g} s ({r['cores']['GWLENS_y'][-1]:.1f} cores) | "
              f"GLoW {r['GLoW_y'][-1]:9.4g} s {note}", flush=True)

np.savez(f"pipeline_benchmark_{NCPU}cpu.npz", ncpu=NCPU, n_w=np.array(args.nw), n_y=np.array(args.ny),
         w_fixed=w_fixed, y_source=y_source,
         **{f"{name}|{key}": np.array(v[key]) for name, v in results.items() for key in SERIES},
         **{f"{name}|{key}|cores": np.array(v["cores"][key]) for name, v in results.items() for key in SERIES},
         **{f"{name}|GLoW_y_measured": np.array(v["GLoW_y_measured"]) for name, v in results.items()})

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

fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(6, 8),
                         sharex=True, layout="constrained")
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
    ax.loglog(ny[~meas], ty[~meas], ls="none", marker="s", ms=6, mfc="white",
              mec=st["color"], mew=1.5)
    ax.text(0.02, 0.95, lens_name, transform=ax.transAxes, ha="left", va="top",
            fontsize=14, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    def _tick_label(k):
        e = int(round(np.log10(k)))
        return rf"$10^{{{e}}}$" if k == 10**e else str(k)

    nn = sorted(set(args.nw) | set(args.ny))
    ax.set_xticks(nn)
    ax.set_xticklabels([_tick_label(k) for k in nn], fontsize=14)
    ax.tick_params(labelsize=14)

axes[-1].set_xlabel(r"$n$", fontsize=14)
fig.supylabel(f"Median time per call (s), {NCPU} CPU core{'s' if NCPU > 1 else ''} each",
              fontsize=14)

handles = [Line2D([], [], lw=2, ms=6, **STYLE[k]) for k in ("GLoW_w", "GLoW_y", "GLoW_yw")]
handles.append(Line2D([], [], ls="none", marker="s", ms=6, mfc="white",
                      mec=GLOW_COLOR, mew=1.5))
handles += [Line2D([], [], lw=2, ms=6, **STYLE[k]) for k in ("GWLENS_w", "GWLENS_y", "GWLENS_yw")]
labels = [LABEL["GLoW_w"], LABEL["GLoW_y"], LABEL["GLoW_yw"],
          LABEL["GWLENS_w"], LABEL["GWLENS_y"], LABEL["GWLENS_yw"]]

# outside legend: constrained layout reserves the space above the axes
fig.legend(handles, labels, loc="outside upper center", ncol=1, frameon=False,
           fontsize=14, handlelength=2.5, columnspacing=2.5, handletextpad=0.6)

fig.savefig(f"pipeline_benchmark_{NCPU}cpu.png", dpi=300, bbox_inches="tight")
fig.savefig(f"pipeline_benchmark_{NCPU}cpu.png", dpi=300, bbox_inches="tight")
print(f"\nsaved pipeline_benchmark_{NCPU}cpu.png and .npz")

"""
fig, axes = plt.subplots(nrows=len(models), ncols=1, figsize=(8, 10), sharex=True)
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
    ax.grid(True, which="both", alpha=0.3)
    nn = sorted(set(args.nw) | set(args.ny))
    ax.set_xticks(nn)
    ax.set_xticklabels([str(k) for k in nn])
axes[-1].set_xlabel(r"$n$",fontsize=14)
fig.supylabel(f"Median time per call (s), {NCPU} CPU core{'s' if NCPU > 1 else ''} each",fontsize=14)
# legend columns: GLoW on the left, gwlens on the right (matplotlib fills legends column by column)
handles = [Line2D([], [], lw=2, ms=6, **STYLE[k]) for k in ("GLoW_w", "GLoW_y", "GLoW_yw")]
handles.append(Line2D([], [], ls="none", marker="s", ms=6, mfc="white", mec=GLOW_COLOR, mew=1.5))
handles += [Line2D([], [], lw=2, ms=6, **STYLE[k]) for k in ("GWLENS_w", "GWLENS_y", "GWLENS_yw")]
labels = [LABEL["GLoW_w"], LABEL["GLoW_y"], LABEL["GLoW_yw"], r"GLoW: $n\,\times$ (time per source position), not timed",
          LABEL["GWLENS_w"], LABEL["GWLENS_y"], LABEL["GWLENS_yw"]]
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0.02, 0, 1, 0.92))
fig.savefig(f"pipeline_benchmark_{NCPU}cpu.png", dpi=300, bbox_inches="tight")
print(f"\nsaved pipeline_benchmark_{NCPU}cpu.png and .npz")
"""