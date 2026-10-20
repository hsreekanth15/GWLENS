import numpy as np
import timeit
from gwlens import solver2d
import matplotlib.pyplot as plt
from gwlens.tools import plotting

plotting.apply_figure_style()

# Source-plane grid
x = np.linspace(-1.0, 1.0, 1000)
Y1, Y2 = np.meshgrid(x, x)

# Binary lenses
lens1 = solver2d.point_mass(0.5, (-0.5, 0.0)) + solver2d.point_mass(0.5, (0.5, 0.0))


lens = lens1          # choose which lens to map
w_values = [10, 50, 100]
n_repeat = 5          # number of timed repetitions per w

# Optional warm-up call (removes JIT/compilation or caching overhead
# from the first timed measurement). Uses a tiny grid so it's quick.
_ = solver2d.amplification_map(w_values[0], Y1[:2, :2], Y2[:2, :2], lens, n_workers=1)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
timings = {}

for ax, w in zip(axes, w_values):
    runs = np.empty(n_repeat)
    for i in range(n_repeat):
        t0 = timeit.default_timer()
        Fm = solver2d.amplification_map(w, Y1, Y2, lens, n_workers=1)
        runs[i] = timeit.default_timer() - t0
    timings[w] = runs  # Fm from the last run is reused for plotting

    print(
        f"w = {w:>4}: best {runs.min():.3f} s | "
        f"mean {runs.mean():.3f} ± {runs.std(ddof=1) if n_repeat > 1 else 0.0:.3f} s "
        f"({n_repeat} runs)"
    )

    im = ax.imshow(
        np.abs(Fm),
        extent=(-1, 1, -1, 1),
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )

    ax.set_title(f"$w={w}$ ,t = {runs.mean():.1f} s")
    ax.set_xlabel("$y_1$")
    ax.set_ylabel("$y_2$")

# One shared colorbar for all subplots
fig.colorbar(im, ax=axes, label="$|F|$", pad=0.03, shrink=1.0)

# Show y2 axis only on the leftmost subplot
for ax in axes[1:]:
    ax.set_ylabel("")
    ax.tick_params(axis="y", left=False, labelleft=False)

# Summary table
print("\n   w    best [s]   mean [s]   std [s]")
for w, runs in timings.items():
    std = runs.std(ddof=1) if n_repeat > 1 else 0.0
    print(f"{w:>4}   {runs.min():8.3f}   {runs.mean():8.3f}   {std:7.3f}")

plt.savefig("binary_map.png", dpi=300, bbox_inches="tight")
plt.show()