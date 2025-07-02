from bilby.gw.source import lal_binary_black_hole
from scipy.constants import c,G
import numpy as np
import h5py
from scipy.interpolate import RectBivariateSpline
from astropy import units as u
Msun = 1.988e+30
constant = (8*np.pi*G/c**3)*Msun

def SIS_DimensionlessFrequency(f,sigma_v,Dl,Ds):

    Dl = Dl* (u.kpc)
    Ds = Ds * (u.kpc)
    Dls= Ds-Dl
    
    Deff = (Dl*Ds/Dls).to(u.km)

    sigma_v = sigma_v* (u.km/u.s)

    c = 3e+5 * (u.km/u.s) # in km/s  
    f = f*(1/u.s)

    w  = 32* 3.14**3  * sigma_v**4 *Deff * f/(c**5)
    return w.value

#load the interpolation file

file = "/work/sreekanth/codes/globular_cluster/ampfac/sis/outdir/interpolation/result_sis1.hdf5"
with h5py.File(file, 'r') as file:
    w_values = file['w_values'][:]
    y_values = file['y_values'][:]
    data = file['results'][:]  # (len(w), len(y), 2)

real_data = data[:, :, 0]
imag_data = data[:, :, 1]

real_spline = RectBivariateSpline(w_values,y_values, real_data, kx=3, ky=3)
imag_spline = RectBivariateSpline(w_values, y_values, imag_data, kx=3, ky=3)


def lensed_lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1,tilt_1,
                                 phi_12,a_2,tilt_2,phi_jl,theta_jn,phase,MLz,y,**kwargs):
    
    ws = constant*MLz*frequency_array
    #F_interpolate = kwargs['F_amp']
    F = real_spline(ws,y)[:,0] + 1j*imag_spline(ws,y)[:,0]
    

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXAS', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    waveform_dict = lal_binary_black_hole(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)
    waveform_dict["plus"] *= F
    waveform_dict["cross"] *= F
    return waveform_dict

def sis_lensed_lal_binary_black_hole(frequency_array, mass_1, mass_2, luminosity_distance, a_1,tilt_1,
                                 phi_12,a_2,tilt_2,phi_jl,theta_jn,phase,sigma_v,Dl,Ds,y,**kwargs):
    
    ws = SIS_DimensionlessFrequency(frequency_array,sigma_v,Dl,Ds)
    #F_interpolate = kwargs['F_amp']
    F = real_spline(ws,y)[:,0] + 1j*imag_spline(ws,y)[:,0]
    

    waveform_kwargs = dict(
        waveform_approximant='IMRPhenomXAS', reference_frequency=50.0,
        minimum_frequency=20.0, maximum_frequency=frequency_array[-1],
        catch_waveform_errors=False, pn_spin_order=-1, pn_tidal_order=-1,
        pn_phase_order=-1, pn_amplitude_order=0)
    waveform_kwargs.update(kwargs)
    waveform_dict = lal_binary_black_hole(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn, phase=phase,
        a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2, phi_12=phi_12,
        phi_jl=phi_jl, **waveform_kwargs)
    waveform_dict["plus"] *= F
    waveform_dict["cross"] *= F
    return waveform_dict
