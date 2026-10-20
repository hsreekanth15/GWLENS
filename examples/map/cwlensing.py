import numpy as np
import matplotlib.pyplot as plt
from gwlens import solver2d
from gwlens import microlensing
from scipy.constants import G, c
from gwlens.tools import plotting

plotting.apply_figure_style()



# Separate colormaps
pml_cmap = plt.get_cmap("Blues")
binary_cmap = plt.get_cmap("Oranges")



Dl = 10
Ds = 20
v = 200
M = 50
freq= 1000

SOLAR_MASS = 1.98892e30  
POINT_LENS_W_CONSTANT = 8 * np.pi * G * SOLAR_MASS / c ** 3

w = POINT_LENS_W_CONSTANT*M*freq
print(w)

TE = microlensing.EinsteinCrossingTime(Dl,Ds,v,M)
print(TE)
closest_approach = [0,0.1]

#for colors
colors = np.linspace(0.4, 0.9, len(closest_approach))

t0 = TE
t = np.linspace(0-100,2*TE+100,1000)
lens_pointmass = solver2d.point_mass(1.0,(0,0))
lens_binary= solver2d.point_mass(0.5,(-0.5,0)) + solver2d.point_mass(0.5,(0,0))


for i,y0_i in enumerate(closest_approach):




    #y = np.linspace(-2,2,1000)
    y0 = y0_i
    y = microlensing.MicrolensingImpactparameter(t,y0,t0,TE)
    F_pointmass = solver2d.amplification_map(w, y1 = 0, y2= y, lens=lens_pointmass)


    plt.plot(t,np.abs(F_pointmass), label=f"PML, $y_0$= {y0}", color=pml_cmap(colors[i]))

for i,y0_i in enumerate(closest_approach):
    y0 = y0_i

    y = microlensing.MicrolensingImpactparameter(t,y0,t0,TE)
    F_binary = solver2d.amplification_map(w, y1 = 0, y2= y, lens=lens_binary)
    plt.plot(t,np.abs(F_binary), label=f"Binary, $y_0$= {y0}",color= binary_cmap(colors[i]))

plt.xlabel("Time (days)")
plt.ylabel("$|F|$")
plt.legend()
plt.tight_layout()
plt.savefig(fname="pml-binary_cwlensing.png",dpi=300)