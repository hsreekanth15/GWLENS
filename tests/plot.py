import numpy as np
import mpmath
import matplotlib.pyplot as plt
from gwlens.tools import utilities
mpmath.mp.dps = 50  # Set desired decimal precision

sis_interpolation_file = "outdirsis/interpolation/sis.hdf5"


#parameters

w =  np.linspace(0.01,89,1000)
y = 0.3

F = utilities.interpolate_2D_spline(sis_interpolation_file,w,y)

plt.semilogx(w,np.abs(F), label= y)
plt.xlabel("w")
plt.ylabel("F(w)")
plt.title("SIS amplification factor")
plt.legend()
plt.savefig(fname ="amplification_sis.png",dpi=300)

