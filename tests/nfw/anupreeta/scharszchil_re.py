import numpy as np
from gwlens.lens import nfw
import matplotlib.pyplot as plt
from gwlens.tools import plotting
from gwlens.tools import halo
import astropy.units as u
from astropy.constants import G, c
plotting.apply_figure_style()

lens_virial_mass = np.logspace(3, 16, 100) #This should be the x-axis 
colors =  ["green", "orange", "red", "purple", "brown", "pink"] 

fig, ax = plt.subplots(figsize=(12, 8))

source_redshift = 2.0
for i,lenszl in enumerate([0.5]):

    Einstein_radius = []
    Deff = []
    for m in lens_virial_mass:
        Halo =halo.NFW(M200=m, zl = lenszl)
        
        x = 1
        M_at_xrs = Halo.mass_enclosed_at_xrs(x)

        
        theta_E = np.sqrt((12*M_at_xrs*G/(c**2)) * Halo.effective_distance(zs = source_redshift)   ).decompose()
        #print(theta_E.to(u.kpc).value)
        Einstein_radius.append(theta_E.to(u.kpc).value)
        Deff.append(Halo.effective_distance(zs = source_redshift).to(u.kpc).value)


w = 1.0  # This is the value of w for which we want to plot the relationship
def lambda_from_RE(Einstein_radius, Deff):
    """
    Calculate wavelength λ from Einstein radius.
    λ = (2π/w) * (R_E^2 / D_eff)
    """
    return (2 * np.pi / w) * ((Einstein_radius**2) / Deff)


lambda_values = lambda_from_RE(np.array(Einstein_radius), np.array(Deff))

ax.plot(lambda_values, Einstein_radius, 'b-', linewidth=2.5)
 
# Add regions for w < 1 and w > 1
# These are vertical bands showing where w < 1 and w > 1 would apply
# For visualization, we'll mark them as different shaded regions
mid_lambda = np.mean(lambda_values)
ax.axvspan(lambda_values.min(), mid_lambda, alpha=0.1, color='red', label='w > 1 region')
ax.axvspan(mid_lambda, lambda_values.max(), alpha=0.1, color='green', label='w < 1 region')
 
# Labels and formatting
ax.set_xlabel('$\\lambda_{gw} (kpc)$ ', fontsize=12, fontweight='bold')
ax.set_ylabel('$R_E$ (kpc)', fontsize=12, fontweight='bold')
#ax.set_title('Gravitational Lens: Wavelength vs Einstein Radius (w=1)', 
             #fontsize=14, fontweight='bold', pad=20)
 
# Create secondary y-axis for mass
ax2 = ax.twinx()
ax2_values = lens_virial_mass #np.linspace(R_E_min, R_E_max, len(mass_values))
#mass_at_RE = mass_from_RE(ax2_values, D_eff)
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(ax.get_yticks())
ax2.set_yticklabels([f'{ax2_values[i]:.1e}' for i in range(len(ax.get_yticks()))])
#ax2.set_ylabel('Mass of lens [M☉]', fontsize=12, fontweight='bold', color='darkred')
#ax2.tick_params(axis='y', labelcolor='darkred')
 
# Create secondary x-axis for D_eff (wavelength to frequency conversion)
ax3 = ax.twiny()
ax3.set_xlim(ax.get_xlim())
ax3.set_xlabel('D_eff relationship', fontsize=12, fontweight='bold')
 
# Grid
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
 
# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
 
# Add annotation
#textstr = f'Parameters:\nw = {w}\nD_eff = {Deff:.2e} m'
#ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,verticalalignment='bottom', horizontalalignment='right',bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
 
plt.tight_layout()
plt.show()
