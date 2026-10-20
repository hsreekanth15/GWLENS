import timeit
import numpy as np
import matplotlib.pyplot as plt
from gwlens import solver2d as G
from gwlens.tools import plotting

NS   = [1, 10, 20]       # rows: number of SIS  (must match the GLoW cell)
SEED = 7                # same random lens as the GLoW cell
R    = 1.0              # disc radius the SIS centres are drawn from
WS   = [1.0, 10.0, 50.0] # columns (must match glow_worker_msis.WS)

L, N = 1.5, 500         # source plane [-L, L]^2, N x N pixel CENTRES (same as the GLoW cell)
q = (np.arange(N) + 0.5) * 2 * L / N - L
Y1, Y2 = np.meshgrid(q, q)
R_SRC = float(np.hypot(Y1, Y2).max()) * (1 + 1e-6)     # source disc that contains every pixel


def distribute_centers(n, R=1.0, seed=0):
    """Identical recipe to glow_worker_msis.distribute_centers (own RNG, centre of mass at 0)."""
    rng = np.random.RandomState(seed)
    Rs = R * np.sqrt(rng.random_sample(n))
    ths = 2 * np.pi * rng.random_sample(n)
    x1, x2 = Rs * np.cos(ths), Rs * np.sin(ths)
    return x1 - x1.mean(), x2 - x2.mean()


def multi_sis(n, seed, R):
    xc1, xc2 = distribute_centers(n, R, seed)
    lens = None
    for a, b in zip(xc1, xc2):
        piece = G.sis(1.0 / n, (a, b))                 # psi0 = 1/n each
        lens = piece if lens is None else lens + piece
    return lens



def compute_maps():
    """Compute all multi-SIS magnification maps."""
    maps = np.empty((len(NS), len(WS), N, N))

    for i, n_sis in enumerate(NS):
        lens = multi_sis(n_sis, SEED, R)

        for j, w in enumerate(WS):
            with G.Solver2D( lens,(0.0, 0.0),w,w,y_radius=R_SRC,n_workers=1,threads=1,**G.PE_SETTINGS) as s:
                maps[i, j] = np.abs(s.map(w, Y1, Y2))

    return maps


times = timeit.repeat( stmt="compute_maps()",
    globals=globals(),
    repeat=2,
    number=1
)

print("Execution times for five runs:")
for i, t in enumerate(times, start=1):
    print(f"Run {i}: {t:.2f} s")

print(f"\nMinimum time: {min(times):.2f} s")
print(f"Maximum time: {max(times):.2f} s")
print(f"Mean time:    {np.mean(times):.2f} s")
print(f"Std. dev.:    {np.std(times, ddof=1):.2f} s")

# Compute once more for plotting
maps_gw = compute_maps()


# ---- Plot ----
plotting.apply_figure_style()

fig, axes = plt.subplots(
    len(NS), len(WS),
    figsize=(9.5, 8.5),
    sharex=True,
    sharey=True,
    constrained_layout=True
)
axes = np.atleast_2d(axes)

for i, n_sis in enumerate(NS):
    vmax = np.nanpercentile(maps_gw[i], 99.5)

    for j, w in enumerate(WS):
        ax = axes[i, j]

        im = ax.imshow(
            maps_gw[i, j],
            extent=[-L, L, -L, L],
            origin="lower",
            cmap="viridis",
            interpolation="nearest",
            vmin=0,
            vmax=vmax
        )

        ax.grid(False)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)

        if i == 0:
            ax.set_title(rf"$w={w:g}$")
        if i == len(NS) - 1:
            ax.set_xlabel(r"$y_1$")
        if j == 0:
            ax.set_ylabel(rf"{n_sis} SIS" + "\n" + r"$y_2$")

    cb = fig.colorbar(im, ax=axes[i, :], pad=0.02)
    cb.set_label(r"$|F(w,\mathbf{y})|$")

    step = max(1, int(round(vmax / 5)))
    cb.set_ticks(np.arange(0, vmax, step))

fig.savefig("gwlens_multi_sis1.pdf", dpi=300, bbox_inches="tight")
plt.show()
