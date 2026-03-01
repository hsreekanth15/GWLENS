import numpy as np
import h5py
from scipy.interpolate import RectBivariateSpline
from astropy import units as u
from astropy.constants import G,c
import matplotlib.pyplot as plt


Msun = 1.988e+30
constant = (8*np.pi*G/c**3) * Msun
print(G)
print(c)


file = "outdir/interpolation/out.hdf5"
with h5py.File(file, 'r') as file:
    w_values = file['w_values'][:]
    y_values = file['y_values'][:]
    data = file['results'][:]  # (len(w), len(y), 2)

real_data = data[:, :, 0]
imag_data = data[:, :, 1]

real_spline = RectBivariateSpline(w_values,y_values, real_data, kx=3, ky=3)
imag_spline = RectBivariateSpline(w_values, y_values, imag_data, kx=3, ky=3)


#print(w_values)
#print(y_values)
#print(data)

frequency_array = np.linspace(0.01,1000,300)
MLz = 100
y = 8
ws = constant*MLz*frequency_array
F = real_spline(ws,y)[:,0] + 1j*imag_spline(ws,y)[:,0]

plt.plot(ws,np.abs(F))
plt.xscale("log")
plt.show()