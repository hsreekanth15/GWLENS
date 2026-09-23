#author:Sreekanth Harikumar

#This scripts generates the lens.bin files for dressed primordial black holes using the interpolation file generated with GWLENS.
#The script reads the interpolation data, evaluates the amplification factor for different impact parameters,
#and writes the results to binary files for each segment.
#This creates amplification and phase correction files

import numpy as np
import h5py
from scipy.constants import c,G
#from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import CubicSpline
from astropy import units as u
from gwlens import microlensing
import os

#constants

M_sun = 1.98e+30
constant = (8*np.pi*G)/c**3 * M_sun

def spline_eval_chunked(real_spline, imag_spline, w, y, chunk_size=500_000):
    """
    Evaluate bivariate splines in chunks to avoid allocating huge arrays.
    """
    n = len(y)
    real_out = np.empty(n, dtype=np.float32)
    imag_out = np.empty(n, dtype=np.float32)

    for i in range(0, n, chunk_size):
        j = i + chunk_size
        real_out[i:j] = real_spline(w, y[i:j], grid=False)
        imag_out[i:j] = imag_spline(w, y[i:j], grid=False)

    return real_out + 1j * imag_out


#Path below is to dressed primordial black hole interpolation file generated with GWLENS
file = "/work/sreekanth/codes/mini-halo/dressed_pbh/ampfac/outdir/interpolation/result_100M_varfpbh_1kfreq.hdf5"


with h5py.File(file,'r') as file:
     w_values= file['w_values'][:]
     y_values = file['y_values'][:]
     f_pbh_values = file['f_pbh_values'][:]
     m_pbh_values = file['m_pbh_values'][:]
     data = file['results'][:]


print(data.shape)
print("f_pbh:",f_pbh_values)
print("m_pbh:",m_pbh_values)
print("w_values:",w_values)

fpbh_index = 3  # Index for f_pbh_values

#For anirudh
#the data is structured as [w_index, y_index, f_pbh_index, 0 for real, 1 for imag]
#So below we are extracting the real and imaginary parts for the first f_pbh value across a fixed w and f_Pbh  and varying y values
#y_values are changing due to micorlensing impact parameter changes with time, so we will be evaluating the amplification factor for different y values at a fixed w and f_Pbh

real_data = data[0,:,fpbh_index, 0, 0]
imag_data = data[0,:,fpbh_index, 0, 1]

#Interpolating the real and imaginary parts of the amplification factor using RectBivariateSpline
real_spline = CubicSpline(y_values,real_data)
imag_spline = CubicSpline(y_values,imag_data)

#Choosing a arbitrary freqeuncy to be consistent with earlier analysis of pbh
frequency = 1000.1
MLz = m_pbh_values[0] # Using the first mass value from the file, can be changed as needed
bandwidth = 0.25
dt = 1/(2*bandwidth)
Dl = 10
Ds = 20
v = 200
M = MLz

w = constant*MLz*frequency

segment_days = 6


t_obs = 120*segment_days

print("Number of days in the segment:",segment_days)
te = microlensing.EinsteinCrossingTime(Dl,Ds,v,M)

print(te)


TE = te

print("Einstein crossing time",TE)

N = 2*t_obs*86164/dt
print(N)

t = np.linspace(TE-t_obs/2,TE+t_obs/2,int(N))
#Choosing the source to go exactly behind the lens
y0 = 0
t0 = te
y = microlensing.MicrolensingImpactparameter(t,y0,t0,TE)



#F = real_spline(w,y, grid=False) + 1j*imag_spline(w,y,grid=False)

#F = spline_eval_chunked(real_spline,imag_spline,w,y,chunk_size=500000)

#Fabs = np.abs(F)

total_segments = 120 #2*TE/segment_days

print("Total number of Segments",total_segments)


chunks = np.array_split(y,total_segments)

print("Prepared amplification factor chucks:",len(chunks))

data_directory = ""

for i, chunk in enumerate(chunks, start=1):
    segment_directory = f"{i:03d}"
    folder_path = os.path.join(data_directory,segment_directory)
    print("Writing to segment:",i,"in directory",segment_directory)

    for det in ["H1"]:
        lensdata_path = os.path.join(folder_path,det,f"lens_{fpbh_index}.bin")
        F = real_spline(chunk) + 1j*imag_spline(chunk)
        phase = np.angle(F)
        Fabs = np.abs(F)

        # Create interleaved array [amp, phase, amp, phase, ...]
        interleaved = np.empty(2 * len(Fabs), dtype=np.float64)
        interleaved[0::2] = Fabs
        interleaved[1::2] = phase

        interleaved.tofile(lensdata_path)
        print("Wrote to:",det)

print("Lens preparation completed successfully")







