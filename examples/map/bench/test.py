"""GLoW vs gwlens (2D solver) runtime with the SAME number of CPU cores for both.

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
Each timing also reports "cores busy" = CPU time / wall time (this process, all its threads, and the
worker processes of evaluate()), so you can see how many of the NCPU cores each code actually used.

Two gwlens numbers are reported:
  * GWLENS (one source): one call = build + spectrum for one source position on NCPU cores, the same
    task as one GLoW call. A single call is one fine-grained transform and keeps only ~1-1.5 cores busy.
  * GWLENS (NCPU sources at once): a solver for a small disc of source positions computes 4*NCPU
    spectra with evaluate() on NCPU processes (one core each); the time is per source, build
    included. This is how gwlens uses many cores (e.g. a sampler with npool workers), and the fair
    throughput comparison with GLoW, which has to redo its time-domain integral for every source.
"""
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("ncpu", type=int, nargs="?", default=int(os.environ.get("NCPU", 1)), help="number of CPU cores for both codes")
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--number", type=int, default=5, help="calls per timed repeat (times are reported per call)")
ap.add_argument("--nw", type=int, nargs="+", default=[1, 5, 10, 100, 1000, 10000], help="numbers of frequencies")
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
from gwlens import solver2d         # noqa: E402  the new lens2d_general.py
from glow import lenses, time_domain_c, freq_domain_c   # noqa: E402
try:
    from gwlens.tools import plotting               # noqa: E402
    plotting.apply_figure_style()
except ImportError:
    pass

BENCH_COLORS = ["#4682a8", "#a0553f", "#2a9d8f"]   # GLoW blue / gwlens brown / gwlens batch green


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
n_w_values = args.nw
y_source = 0.1
ws_full = np.logspace(-4, 2, 1000)

P_T = {'tmin': 1e-2, 'tmax': 1e9, 'Nt': 5000, 'sampling': 'log', 'parallel': NCPU > 1}
P_FFT = {'wmin': 0.6 * ws_full.min(), 'wmax': ws_full.max()}
GW_KW = dict(n_workers=NCPU, threads=NCPU, interpolate=True, nlag=16)

models = [("4 SIS", build_4sis), ("4 CIS", build_4cis), ("Binary Lens", build_binary)]


def cpu_seconds():
    """CPU time used so far by this process (all its threads) and by its finished child processes
    (the worker processes of evaluate(), which are joined when each solver is closed)."""
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def benchmark_pipeline(func, repeats=args.repeats, number=args.number):
    """Median wall time per call, all per-call wall times, and the median number of busy cores
    (CPU time of this process and its worker processes / wall time)."""
    wall, cpu = [], []
    for _ in range(repeats):
        c0, t0 = cpu_seconds(), time.perf_counter()
        for _ in range(number):
            func()
        wall.append((time.perf_counter() - t0) / number)
        cpu.append((cpu_seconds() - c0) / number)
    wall, cpu = np.asarray(wall), np.asarray(cpu)
    return float(np.median(wall)), wall, float(np.median(cpu / wall))


# --------------------------------------------------
# Run benchmarks
# --------------------------------------------------
print(f"NCPU = {NCPU}: pinned to cores {sorted(os.sched_getaffinity(0))}, OMP_NUM_THREADS = {os.environ['OMP_NUM_THREADS']}")
print(f"GLoW parallel = {P_T['parallel']}; gwlens {GW_KW}")
results = {}
for lens_name, builder in models:
    print(f"\nBenchmarking {lens_name}...")
    lens_gw, lens_glow = builder()
    KEYS = ("GWLENS", "GWLENS_batch", "GLoW")
    results[lens_name] = {k: [] for k in KEYS}
    results[lens_name]["cores"] = {k: [] for k in KEYS}
    results[lens_name]["raw"] = {k: [] for k in KEYS}
    ys_batch = np.column_stack([y_source + 0.04 * np.cos(np.linspace(0, 2 * np.pi, 4 * NCPU, endpoint=False)),
                                0.04 * np.sin(np.linspace(0, 2 * np.pi, 4 * NCPU, endpoint=False))])
    for n_w in n_w_values:
        ws = np.logspace(-4, 2, n_w)

        def run_gwlens():
            return solver2d.amplification_many(ws, (y_source, 0.0), lens_gw, **GW_KW)

        def run_gwlens_batch():
            wp = ws[ws > 0]
            with solver2d.Solver2D(lens_gw, (y_source, 0.0), wp.min(), wp.max(), y_radius=0.05,
                                   n_workers=NCPU, threads=1, nlag=16) as s:
                return s.evaluate(ws, ys_batch)

        def run_glow():
            It = time_domain_c.It_MultiContour_C(lens_glow, max(y_source, 1e-6), dict(P_T))
            Fw = freq_domain_c.Fw_FFT_C(It, dict(P_FFT))
            return Fw.eval_Fw(ws)

        run_gwlens()                                  # warm-up, outside the timed repetitions
        run_gwlens_batch()
        run_glow()
        for key, fn, per in (("GWLENS", run_gwlens, 1), ("GWLENS_batch", run_gwlens_batch, len(ys_batch)),
                             ("GLoW", run_glow, 1)):
            med, raw, cores = benchmark_pipeline(fn)
            results[lens_name][key].append(med / per)
            results[lens_name]["raw"][key].append(raw / per)
            results[lens_name]["cores"][key].append(cores)
        r = results[lens_name]
        print(f"n_w={n_w:5d} | GWLENS one source {r['GWLENS'][-1]:.4g} s ({r['cores']['GWLENS'][-1]:.1f} cores busy) | "
              f"GWLENS {len(ys_batch)} sources {r['GWLENS_batch'][-1]:.4g} s per source ({r['cores']['GWLENS_batch'][-1]:.1f} cores busy) | "
              f"GLoW {r['GLoW'][-1]:.4g} s ({r['cores']['GLoW'][-1]:.1f} cores busy)")

np.savez(f"pipeline_benchmark_{NCPU}cpu.npz", ncpu=NCPU, n_w=np.array(n_w_values),
         **{f"{name}|{key}": np.array(v[key]) for name, v in results.items() for key in KEYS},
         **{f"{name}|{key}|cores": np.array(v["cores"][key]) for name, v in results.items() for key in KEYS})

# --------------------------------------------------
# Plot: 3 rows x 1 column
# --------------------------------------------------
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True, constrained_layout=True)
for ax, (lens_name, _) in zip(axes, models):
    ax.semilogx(n_w_values, results[lens_name]["GLoW"], marker="s", linewidth=2, markersize=6, linestyle="--",
                label="GLoW", color=BENCH_COLORS[0])
    ax.semilogx(n_w_values, results[lens_name]["GWLENS"], marker="o", linewidth=2, markersize=6,
                label="GWLENS, one source per call", color=BENCH_COLORS[1])
    ax.semilogx(n_w_values, results[lens_name]["GWLENS_batch"], marker="D", linewidth=2, markersize=6, linestyle=":",
                label=f"GWLENS, {4 * NCPU} sources on {NCPU} cores (per source)", color=BENCH_COLORS[2])
    ax.set_title(lens_name)
    ax.set_ylabel("Median time per call (s)")
    ax.set_xscale("log")
    ax.set_xticks(n_w_values)
    ax.set_xticklabels([str(n) for n in n_w_values])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
axes[-1].set_xlabel(r"$n_w$ (number of frequency samples)")
fig.suptitle(f"Runtime comparison: GWLENS vs GLoW, {NCPU} CPU core{'s' if NCPU > 1 else ''} each "
             f"({args.repeats} repetitions)", fontsize=14)
fig.savefig(f"pipeline_benchmark_{NCPU}cpu.png", dpi=300, bbox_inches="tight")