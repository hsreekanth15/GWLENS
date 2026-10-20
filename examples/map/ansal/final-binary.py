
import timeit
import numpy as np
import matplotlib.pyplot as plt
from gwlens import solver2d as G

seps = np.linspace(0.3, 1.6, 25)
w, L, n = 100.0, 0.8, 500

g = np.linspace(-L, L, n)
Y1, Y2 = np.meshgrid(g, g)

R = L * np.sqrt(2) * (1 + 1e-6)

n_repeat = 5
runs = np.empty(n_repeat)
maps = None

for i in range(n_repeat):
    current_maps = np.empty((len(seps), n, n))

    t0 = timeit.default_timer()

    for k, d in enumerate(seps):
        lens = (
            G.point_mass(0.5, (-d / 2, 0.0))
            + G.point_mass(0.5, (d / 2, 0.0))
        )

        #with G.Solver2D( lens,(0, 0),w,w,y_radius=R,n_workers=1,threads=1,   **G.PE_SETTINGS) as s:
        #using n_threads=None is faster because it used FINUFFT's default process level parallelism 
        current_maps[k] = np.abs(G.amplification_map(w,Y1, Y2,lens,n_workers =1,threads=1, **G.PE_SETTINGS))

    runs[i] = timeit.default_timer() - t0

    # Keep the most recent complete run for plotting.
    maps = current_maps

print(
    f"w = {w:>4}: best {runs.min():.3f} s | "
    f"mean {runs.mean():.3f} ± {runs.std(ddof=1):.3f} s "
    f"({n_repeat} runs) \n"
)
print(runs)
def apply_style(usetex=False, fontsize=15):
    """Serif/STIX look with boxed axes and outward ticks."""
    plt.rcParams.update({
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "text.usetex": bool(usetex),
        "font.size": fontsize,
        "axes.labelsize": fontsize + 1,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "axes.linewidth": 1.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.major.size": 6,
        "ytick.major.size": 6,
    })

apply_style(usetex=False, fontsize=11)

fig, axes = plt.subplots(
    5, 5,
    figsize=(10, 10),
    constrained_layout=True
)

vmin = 0
vmax = np.percentile(maps, 99.5)

for k, ax in enumerate(axes.flat):
    ax.imshow(
        maps[k],
        extent=[-L, L, -L, L],
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal"
    )

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)

    ax.text(
        0.05, 0.95,
        rf"$d={seps[k]:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="white",
        bbox=dict(
            facecolor="black",
            alpha=0.5,
            edgecolor="none",
            pad=2
        )
    )

fig.suptitle(
    rf"Equal-mass binary point lens ($w={w:g}$)",
    fontsize=14
)

fig.savefig("gwlens_binary_map_test.png", dpi=300, bbox_inches="tight")
fig.savefig("gwlens_binary_map_test.pdf", bbox_inches="tight")

plt.show()
