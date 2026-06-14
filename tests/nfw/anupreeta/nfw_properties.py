import numpy as np
from gwlens.lens import nfw
import matplotlib.pyplot as plt
from gwlens.tools import plotting
from gwlens.tools import halo
import astropy.units as u
from astropy.constants import G, c
plotting.apply_figure_style()


lens_virial_mass = np.logspace(3, 16, 100) #This should be the x-axis 
impact_parameter = np.logspace(-2, 1.0, 6) #This should be the y-axis
colors =  ["green", "orange", "red", "purple", "brown", "pink"] 

source_redshift = 2.0
for i,lenszl in enumerate([0.3,0.5,0.7,1.0]):

    Einstein_radius = []

    for m in lens_virial_mass:
        Halo =halo.NFW(M200=m, zl = lenszl)
        
        x = 1
        M_at_xrs = Halo.mass_enclosed_at_xrs(x)

        
        theta_E = np.sqrt((12*M_at_xrs*G/(c**2)) * Halo.effective_distance(zs = source_redshift)   ).decompose()
        #print(theta_E.to(u.kpc).value)
        Einstein_radius.append(theta_E.to(u.kpc).value)



#help(Halo)


    plt.plot(lens_virial_mass, Einstein_radius,color=colors[i], lw = 3, label=f"$z_l = {lenszl}, z_s = {source_redshift}$")
plt.xscale('symlog')
plt.yscale('symlog')
plt.xlabel("$M_{vir} (M_\\odot)$")
plt.ylabel("$R_E$ (kpc)")
plt.legend()
#plt.savefig("Einstein_radius_vs_virial_mass.png",dpi=300)
#plt.show()
#plt.close()

R_schr = 2*G*lens_virial_mass*(u.M_sun)/c**2
R_schr = R_schr.decompose().to(u.kpc).value
plt.plot(lens_virial_mass, R_schr,color="blue", lw = 3, label="Schwarzschild radius")
plt.xscale('symlog')
plt.yscale('symlog')
plt.xlabel("$M_{vir} (M_\\odot)$")
plt.ylabel("$R_{sch} (kpc)$")
plt.legend()
plt.show()
#plt.savefig("Schwarzschild_radius_vs_virial_mass.png",dpi=300)