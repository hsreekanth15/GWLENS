import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, AutoMinorLocator, MaxNLocator
from gwlens import solver1d
from gwlens.tools import plotting
from glow import lenses, time_domain_c, freq_domain_c

# Uncomment below if LaTeX works
plotting.apply_figure_style()
# plt.rcParams.update({'text.usetex': False})

# ---------------- common setup ----------------
ws = np.logspace(-4, 2, 2000)
y_values = [0.1, 0.5, 2.0]

P_T   = {'tmin': 1e-2, 'tmax': 1e9, 'Nt': 2000, 'sampling': 'log'}
P_FFT = {'wmin': 0.6 * ws.min(), 'wmax': ws.max()}
# ---------------- lens definitions: (label, gwlens lens, glow lens) ----------------
models = [
    ("Point Mass", solver1d.point_mass(), lenses.Psi_PointLens()),
    ("SIS",        solver1d.sis(),        lenses.Psi_SIS()),
    ("CIS",        solver1d.cis(),        lenses.Psi_CIS()),
]

# ---------------- evaluate & store ----------------
RES = {}   # RES[label] = {"gw": {y: |F|}, "glow": {y: |F|}}
for label, lens_gw, lens_glow in models:
    RES[label] = {"gw": {}, "glow": {}}
    for y in y_values:
        RES[label]["gw"][y] = np.abs(
            solver1d.amplification_many(ws, y, lens_gw, n_workers=1))
        It = time_domain_c.It_SingleIntegral_C(lens_glow, y, dict(P_T))
        Fw = freq_domain_c.Fw_FFT_C(It,dict(P_FFT))
        RES[label]["glow"][y] = np.abs(Fw(ws))

# ---------------- plot ----------------
cmap = plt.get_cmap("Oranges")
cols = [cmap(s) for s in np.linspace(0.45, 0.95, len(y_values))]
DASH = (0, (4, 2.2))
FLOOR = 1e-8          # avoids log(0) in the residual panel

fig, axes = plt.subplots(
    2, 3, figsize=(15, 6.8), sharex=True, sharey="row",
    gridspec_kw={"height_ratios": [3.4, 1.2], "hspace": 0.05, "wspace": 0.14},
)

for j, (label, _, _) in enumerate(models):
    ax, axr = axes[0, j], axes[1, j]
    for y, c in zip(y_values, cols):
        Fg, Fl = RES[label]["gw"][y], RES[label]["glow"][y]
        ax.plot(ws, Fg, color=c,   lw=3.0, solid_capstyle="round", zorder=2)
        ax.plot(ws, Fl, color="k", lw=1.1, ls=DASH, zorder=3)

        rel = 100.0 * np.abs(Fl - Fg) / Fg
        axr.plot(ws, np.maximum(rel, FLOOR), color=c, lw=1.2)

    ax.text(0.04, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=17)
    axr.set_xlabel(r"$w$")

# ---- shared x axis (set once, propagates through sharex) ----
axes[0, 0].set_xscale("log")
axes[0, 0].set_xlim(ws.min(), ws.max())
axes[0, 0].xaxis.set_major_locator(LogLocator(base=10, numticks=10))
axes[0, 0].xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=50))
axes[0, 0].xaxis.set_minor_formatter(NullFormatter())

# ---- shared |F| axis ----
axes[0, 0].yaxis.set_major_locator(MaxNLocator(6))
axes[0, 0].yaxis.set_minor_locator(AutoMinorLocator(5))
axes[0, 0].set_ylabel(r"$|F(w,y)|$")

# ---- shared residual axis (log) ----
axr0 = axes[1, 0]
axr0.set_yscale("log")
axr0.set_ylim(1e-6, 1e1)
axr0.yaxis.set_major_locator(LogLocator(base=10, numticks=4))
axr0.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=8))
axr0.yaxis.set_minor_formatter(NullFormatter())
axr0.set_ylabel(r"$|\Delta F|/|F|\ [\%]$")

# ---- hide inner tick labels ----
for j in (1, 2):
    for i in (0, 1):
        axes[i, j].tick_params(labelleft=False)
for j in range(3):
    axes[0, j].tick_params(labelbottom=False)

# ---- one shared legend row on top ----
handles = [Line2D([], [], color=c, lw=3.0, label=fr"$y={y}$") for y, c in zip(y_values, cols)]
handles += [Line2D([], [], color="0.45", lw=3.0, label="GWLENS"),
            Line2D([], [], color="k", lw=1.4, ls=DASH, label="GLoW")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.92),
           ncol=5, frameon=False, handlelength=1.9, columnspacing=2.0, handletextpad=0.6)

fig.align_ylabels(axes[:, 0])
fig.savefig("1dlens-comparison.png", bbox_inches="tight", dpi=300)
# fig.savefig("1dlens-comparison.pdf", bbox_inches="tight")
plt.show()