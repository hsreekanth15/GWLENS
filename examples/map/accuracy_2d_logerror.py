import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter, AutoMinorLocator, MaxNLocator
from gwlens import solver2d
from glow import lenses, time_domain_c, freq_domain_c

def apply_style(usetex=True, fontsize=15):
    plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    })

apply_style(usetex=False, fontsize=15)

# ---------------- common setup ----------------
y_values = [0.1, 0.5, 2.0]
# one grid for both codes: log at low w, linear (fringe-resolving) at high w
#ws = np.unique(np.concatenate([np.logspace(-4, 1, 500), np.linspace(10, 100, 3000)]))
ws= np.logspace(-4, 2, 1000)
P_T   = {'tmin': 1e-2, 'tmax': 1e9, 'Nt': 20000, 'sampling': 'log'}
P_FFT = {'wmin': 0.6 * ws.min(), 'wmax': ws.max()}

# ---------------- lens definitions: each returns (gwlens lens, glow lens) ----------------

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

models = [("4 SIS", build_4sis), ("4 CIS", build_4cis), ("Binary Lens", build_binary)]

# ---------------- evaluate & store ----------------
RES = {}   # RES[label] = {"gw": {y: F}, "glow": {y: F}}
for label, builder in models:
    lens_gw, lens_glow = builder()
    RES[label] = {"gw": {}, "glow": {}}
    for y in y_values:
        RES[label]["gw"][y] = np.abs(
            solver2d.amplification_many(ws, (y, 0.0), lens_gw, n_workers=8))
        It = time_domain_c.It_MultiContour_C(lens_glow, max(y, 1e-6), dict(P_T))
        Fw = freq_domain_c.Fw_FFT_C(It, dict(P_FFT))
        RES[label]["glow"][y] = np.abs(Fw.eval_Fw(ws))

# ---------------- plot ----------------
cmap = plt.get_cmap("cividis_r")
cols = [cmap(s) for s in np.linspace(0.45, 0.95, len(y_values))]
DASH = (0, (4, 2.2))
FLOOR = 1e-8          # avoids log(0) error

fig, axes = plt.subplots(
    2, 3, figsize=(15, 6.8), sharex=True, sharey="row",
    gridspec_kw={"height_ratios": [3.4, 1.2], "hspace": 0.05, "wspace": 0.14},
)

for j, (label, _) in enumerate(models):
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
axes[0, 0].set_xlim(1e-4, 1e2)
axes[0, 0].xaxis.set_major_locator(LogLocator(base=10, numticks=10))
axes[0, 0].xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=50))
axes[0, 0].xaxis.set_minor_formatter(NullFormatter())

# ---- shared |F| axis ----
axes[0, 0].yaxis.set_major_locator(MaxNLocator(6))
axes[0, 0].yaxis.set_minor_locator(AutoMinorLocator(5))
axes[0, 0].set_ylabel(r"$|F|$")

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
handles = [Line2D([], [], color=c, lw=3.0, label=fr"$y=({y},0)$") for y, c in zip(y_values, cols)]
handles += [Line2D([], [], color="0.45", lw=3.0, label="GWLENS"),
            Line2D([], [], color="k", lw=1.4, ls=DASH, label="GLoW")]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.92),
           ncol=5, frameon=False, handlelength=1.9, columnspacing=2.0, handletextpad=0.6)

fig.align_ylabels(axes[:, 0])
#fig.savefig("amplification_all_gwlens_vs_glow.pdf", bbox_inches="tight")
fig.savefig("2dlens-comparison.png", bbox_inches="tight", dpi=300)
plt.show()