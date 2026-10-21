
import timeit
import numpy as np
import matplotlib.pyplot as plt
from gwlens.tools import plotting
from gwlens import solver2d_old
from glow import lenses, time_domain_c, freq_domain_c
plotting.apply_figure_style()

peset = solver2d_old.PE_SETTINGS

COLORS = ["#7f7f7f", "#8e1b9e", "#b22222", "#1f77b4", "#e08a00", "#2a9d8f"]
BENCH_COLORS = ["#4682a8", "#a0553f", "#7f7f7f", "#8e1b9e", "#e08a00", "#2a9d8f"]   # blue / brown
MARKERS = ["^", "o", "d", "s", "v", "P"]


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
        p = solver2d_old.sis(1.0 / n, (a, b))
        gw = p if gw is None else gw + p
    gl = lenses.CombinedLens({'lenses': [
        lenses.Psi_offcenterSIS(p_phys={'psi0': 1.0 / n, 'xc1': a, 'xc2': b})
        for a, b in zip(xc1, xc2)]})
    return gw, gl

def build_4cis(rc=0.05, n=4):
    cen = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    gw = sum((solver2d_old.cis(1.0 / n, rc, c) for c in cen[1:]),
             solver2d_old.cis(1.0 / n, rc, cen[0]))
    gl = lenses.CombinedLens({'lenses': [
        lenses.Psi_offcenterCIS(p_phys={'psi0': 1.0 / n, 'rc': rc, 'xc1': a, 'xc2': b})
        for a, b in cen]})
    return gw, gl

def build_binary():
    m, cen = [0.5, 0.5], [(-0.5, 0.0), (0.5, 0.0)]
    gw = solver2d_old.point_mass(m[0], cen[0]) + solver2d_old.point_mass(m[1], cen[1])
    gl = lenses.CombinedLens({'lenses': [
        lenses.Psi_offcenterPointLens(p_phys={'psi0': mi, 'xc1': a, 'xc2': b})
        for mi, (a, b) in zip(m, cen)]})
    return gw, gl

# --------------------------------------------------
# Benchmark configuration
# --------------------------------------------------
n_w_values = [1, 5, 10, 100, 1000,10000]
n_repeats = 5
y_source = 0.1

# Common frequency grid
ws_full = np.logspace(-4, 2, 1000)

P_T = {
    'tmin': 1e-2,
    'tmax': 1e9,
    'Nt': 5000,
    'sampling': 'log',
    'parallel': False
}

P_FFT = {
    'wmin': 0.6 * ws_full.min(),
    'wmax': ws_full.max()
}

models = [
    ("4 SIS", build_4sis),
    ("4 CIS", build_4cis),
    ("Binary Lens", build_binary),
]


# --------------------------------------------------
# Benchmark helper
# --------------------------------------------------
def benchmark_pipeline(func, repeats=5):
    """Return median wall-clock runtime and all timings."""
    timings = timeit.repeat(
        stmt=func,
        repeat=repeats,
        number=5
    )
    return np.median(timings), np.asarray(timings)


# --------------------------------------------------
# Run benchmarks
# --------------------------------------------------
results = {}

for lens_name, builder in models:
    print(f"\nBenchmarking {lens_name}...")

    lens_gw, lens_glow = builder()

    results[lens_name] = {
        "GWLENS": [],
        "GLoW": [],
        "raw": {"GWLENS": [], "GLoW": []}
    }

    # Precompute the GLoW time-domain integral and FFT-based
    # frequency-domain object once per lens. This isolates the
    # cost of evaluating different numbers of frequencies.

    for n_w in n_w_values:

        # Select n_w frequencies distributed across the full range.
        # This ensures even n_w=1 samples the frequency range
        # meaningfully, rather than always selecting its lowest end.
        #indices = np.linspace(  0, len(ws_full) - 1, n_w, dtype=int )
        #ws = ws_full[indices]
        ws = np.logspace(-4, 2, n_w)
        def run_gwlens():
            return solver2d_old.amplification_many( ws,(y_source, 0.0), lens_gw,n_workers=-1,threads=None,interpolate=True,nlag=16)

        def run_glow():
            It = time_domain_c.It_MultiContour_C( lens_glow,max(y_source, 1e-6),dict(P_T))
            Fw = freq_domain_c.Fw_FFT_C(It,dict(P_FFT))

            return Fw.eval_Fw(ws)

        # Warm-up, outside the timed repetitions
        run_gwlens()
        run_glow()

        median_gw, raw_gw = benchmark_pipeline(
            run_gwlens, n_repeats
        )
        median_glow, raw_glow = benchmark_pipeline(
            run_glow, n_repeats
        )

        results[lens_name]["GWLENS"].append(median_gw)
        results[lens_name]["GLoW"].append(median_glow)

        results[lens_name]["raw"]["GWLENS"].append(raw_gw)
        results[lens_name]["raw"]["GLoW"].append(raw_glow)

        print(
            f"n_w={n_w:4d} | "
            f"GWLENS median = {median_gw:.6g} s | "
            f"GLoW median = {median_glow:.6g} s"
        )


# --------------------------------------------------
# Plot: 3 rows x 1 column
# --------------------------------------------------
fig, axes = plt.subplots( nrows=3, ncols=1, figsize=(8, 10),sharex=True,constrained_layout=True)

for ax, (lens_name, _) in zip(axes, models):

    gw_times = results[lens_name]["GWLENS"]
    glow_times = results[lens_name]["GLoW"]

    ax.semilogx(
        n_w_values, glow_times,
        marker="s",
        linewidth=2,
        markersize=6,
        linestyle="--",
        label="GLoW",
        color= BENCH_COLORS[0]
    )

    ax.semilogx(
        n_w_values, gw_times,
        marker="o",
        linewidth=2,
        markersize=6,
        label="GWLENS",
        color= BENCH_COLORS[1]
    )


    ax.set_title(lens_name)
    ax.set_ylabel("Median time (s)")
    ax.set_xscale("log")
    ax.set_xticks(n_w_values)
    ax.set_xticklabels([str(n) for n in n_w_values])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)

axes[-1].set_xlabel(r"$n_w$ (number of frequency samples)")

fig.suptitle(
    f"Runtime comparison: GWLENS vs GLoW ({n_repeats} repetitions)",
    fontsize=14
)

fig.savefig(
    "pipeline_benchmark.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
