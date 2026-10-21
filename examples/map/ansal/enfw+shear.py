import timeit
import numpy as np
import matplotlib.pyplot as plt
from gwlens import solver2d as G
from gwlens.tools import plotting
from matplotlib.ticker import MaxNLocator
from scipy.integrate import quad

def nfw_Mc_to_Rs_alphaRs(M200, c200, z_lens, z_source, H0=70.0, Om0=0.3):
    """(M200 [Msun], c200) -> (Rs [arcsec], alpha_Rs [arcsec]); flat LCDM."""
    G_, c_light = 4.30091727e-9, 299792.458            # G [Mpc (km/s)^2 / Msun], c [km/s]
    E  = lambda z: np.sqrt(Om0 * (1 + z)**3 + 1 - Om0)
    Dc = lambda z: (c_light / H0) * quad(lambda t: 1 / E(t), 0, z)[0]   # comoving distance [Mpc]
    Dd  = Dc(z_lens) / (1 + z_lens)
    Ds  = Dc(z_source) / (1 + z_source)
    Dds = (Dc(z_source) - Dc(z_lens)) / (1 + z_source)
    rho_c = 3 * (H0 * E(z_lens))**2 / (8 * np.pi * G_)                  # critical density [Msun/Mpc^3]
    r200  = (3 * M200 / (4 * np.pi * 200 * rho_c))**(1 / 3)
    Rs    = r200 / c200
    rho0  = (200 / 3) * rho_c * c200**3 / (np.log(1 + c200) - c200 / (1 + c200))
    Sigma_cr = c_light**2 / (4 * np.pi * G_) * Ds / (Dd * Dds)          # [Msun/Mpc^2]
    arcsec = np.pi / 180 / 3600
    return Rs / Dd / arcsec, rho0 * 4 * Rs**2 * (1 + np.log(0.5)) / Sigma_cr / Dd / arcsec

def enfw_lens(Rs, alpha_Rs, e1=0.2, e2=0.0):
    """Spherical NFW potential at the 'square-average' elliptical radius (lenstronomy convention)."""
    phi  = 0.5 * np.arctan2(e2, e1)
    cmod = min(np.hypot(e1, e2), 0.9999)
    q = (1 - cmod) / (1 + cmod)
    e = abs(1 - q**2) / (1 + q**2)
    cp, sp = np.cos(phi), np.sin(phi)
    amp = 0.5 * Rs * alpha_Rs / (1 + np.log(0.5))          # = 2 rho0 Rs^3
    def psi(x1, x2):
        xa = (cp * x1 + sp * x2) * np.sqrt(1 - e)
        ya = (-sp * x1 + cp * x2) * np.sqrt(1 + e)
        X = np.sqrt(xa * xa + ya * ya + 0j) / Rs
        return amp * (np.log(X / 2)**2 + np.arccos(1 / X)**2)   # complex-safe for X < 1
    return G.Lens(psi, singular=[(0.0, 0.0)], q_inf=0.0)


M200, z_l, z_s = 1e13, 0.5, 1.5
cs, WS = [15.0, 30.0, 60.0], [1.0, 10.0, 100.0]
shear = G.external_shear(gamma1=0.03, gamma2=0.01)
L, n = 1.5, 500

g = np.linspace(-L, L, n)
Y1, Y2 = np.meshgrid(g, g)
R = L * np.sqrt(2) * (1 + 1e-6)


def compute_maps():
    """Compute all nine magnification maps."""
    maps = np.empty((3, 3, n, n))

    for i, c in enumerate(cs):
        Rs, aRs = nfw_Mc_to_Rs_alphaRs(M200, c, z_l, z_s)
        lens = enfw_lens(Rs, aRs, e1=0.2, e2=0.0) + shear

        for j, w in enumerate(WS):
            with G.Solver2D(
                lens, (0, 0), w, w,
                y_radius=R,
                n_workers=1,
                threads=1,
                **G.PE_SETTINGS
            ) as s:
                maps[i, j] = np.abs(s.map(w, Y1, Y2))

    return maps


# Benchmark five complete runs
times = timeit.repeat(
    stmt="compute_maps()",
    globals=globals(),
    repeat=1,
    number=1
)

print("Execution times for five runs:")
for i, t in enumerate(times, start=1):
    print(f"Run {i}: {t:.2f} s")

print(f"\nMinimum time: {min(times):.2f} s")
print(f"Maximum time: {max(times):.2f} s")
print(f"Mean time:    {np.mean(times):.2f} s")
print(f"Std. dev.:    {np.std(times, ddof=1):.2f} s")

# Compute maps once more for plotting
maps = compute_maps()

# Continue with your original plotting code here.



plotting.apply_figure_style()
plt.rcParams["axes.grid"] = False
fig, axes = plt.subplots(3, 3, figsize=(9.5, 8.5), sharex=True, sharey=True, constrained_layout=True)
for i, c in enumerate(cs):
    for j, w in enumerate(WS):
        ax = axes[i, j]
        im = ax.imshow(maps[i, j], extent=[-L, L, -L, L], origin="lower", cmap="cividis",
                       vmin=maps[i].min(), vmax=maps[i].max())     # shared scale per row
        ax.grid(False)
        if i == 0: ax.set_title(rf"$w={w:g}$")
        if i == 2: ax.set_xlabel(r"$y_1$")
        if j == 0: ax.set_ylabel(rf"$c_{{200}}={c:g}$" + "\n" + r"$y_2$")
    cb = fig.colorbar(im, ax=axes[i, :], pad=0.02)
    cb.set_label(r"$|F(w,\mathbf{y})|$")
    vmax = maps[i].max()
    step = 1 if vmax < 8 else 2
    cb.set_ticks(np.arange(step, vmax, step))  
    #fig.suptitle(r"Elliptical NFW, $M_{200}=10^{13} M_\odot$, $z_l=0.5$, $z_s=1.5$")
fig.savefig("enfw_shear_map.pdf", bbox_inches="tight")
fig.savefig("enfw_shear_map.png", bbox_inches="tight",dpi=300)
plt.show()
