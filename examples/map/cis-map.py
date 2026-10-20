import numpy as np
from gwlens import solver2d
import timeit
import matplotlib.pyplot as plt
from gwlens.tools import plotting

plotting.apply_figure_style()


x = np.linspace(-1.0, 1.0, 1000)
Y1, Y2 = np.meshgrid(x, x)


norm = 1
pos1 = 2.0
pos2 = 2.0

lens_cis = solver2d.cis(norm,0.05,(0,0)) + solver2d.cis(norm,0.05,(pos1,pos2))+solver2d.cis(norm,0.05,(-pos1,pos2)) + solver2d.cis(norm,0.05,(pos1,-pos2))+solver2d.cis(norm,0.05,(-pos1,-pos2)) 

w_values = [10, 50, 100]

n_repeat = 5

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
timings = {}

for ax, w in zip(axes, w_values):
    runs = np.empty(n_repeat)
    for i in range(n_repeat):
        t0 = timeit.default_timer()
        Fm = solver2d.amplification_map(w, Y1, Y2, lens_cis,n_workers=1)
        runs[i] = timeit.default_timer() - t0
    timings[w] = runs

    print(
        f"w = {w:>4}: best {runs.min():.3f} s | "
        f"w = {w:>4}: worst {runs.max():.3f} s | "
        f"mean {runs.mean():.3f} ± {runs.std(ddof=1) if n_repeat > 1 else 0.0:.3f} s "
        f"({n_repeat} runs)"
    )

    im = ax.imshow(
        np.abs(Fm),
        extent=(-1, 1, -1, 1),
        origin="lower",
        cmap="viridis",
        aspect="equal"
    )

    ax.set_title(f"$w={w}$")
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

#plt.tight_layout()
plt.savefig("cis_map.png", dpi=300, bbox_inches="tight")
plt.show()