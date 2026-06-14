from bilby.gw.source import lal_binary_black_hole
from functools import lru_cache
from collections import namedtuple
import numpy as np
import h5py
from gwlens.tools import logutils
import configparser
import os
from scipy.interpolate import RectBivariateSpline
from astropy import units as u
from scipy.constants import c,G
Msun = 1.988e+30
constant = (8*np.pi*G/c**3)*Msun


#### To implement

# Make the lensed waveform model-independent
# The lens specifi dimensionless frequencies should be called from lens folder and the corresponding modules there
# Should allow for arbitrary kwargs as different lenses have different number of parameters.

##  Implement a CLI for inference -> Read the .ini file -> Get the lens_name and lookup_table path -> 
## The CLI entry point could be named as gwlens_pipe_analysis *.ini


LENS_MODELS = {}
 
_Lens = namedtuple("_Lens", ["dimensionless_frequency", "parameters"])
_Splines = namedtuple("_Splines", ["real", "imag"])

@lru_cache(maxsize=None)
def _load_splines(lookup_table):
    """Load and cache the real/imag interpolants of F(w, y) from an HDF5 file.
 
    The file is expected to contain ``w_values`` (1D), ``y_values`` (1D) and
    ``results`` with shape ``(len(w), len(y), 2)`` holding the real and
    imaginary parts of the amplification factor. Cached per path so repeated
    waveform evaluations reuse the same splines.
    """
    with h5py.File(lookup_table, "r") as f:
        w_values = f["w_values"][:]
        y_values = f["y_values"][:]
        data = f["results"][:]
    real = RectBivariateSpline(w_values, y_values, data[:, :, 0], kx=3, ky=3)
    imag = RectBivariateSpline(w_values, y_values, data[:, :, 1], kx=3, ky=3)
    return _Splines(real, imag)



def amplification_factor(frequency_array, lens_model, lookup_table, parameters):
    """Compute the complex wave-optics amplification factor F(f).
 
    The lens parameters are *popped* from ``parameters`` in place, so the
    remaining dictionary is safe to forward to an unlensed bilby source model.
 
    Parameters
    ----------
    frequency_array: array_like
        Frequencies at which to evaluate F, in Hz.
    lens_model: str
        Name of a registered lens model.
    lookup_table: str
        Path to the HDF5 file holding F(w, y) for this lens model.
    parameters: dict
        Mutable parameter dictionary. The keys declared by the lens model are
        removed and used here.
 
    Returns
    -------
    numpy.ndarray
        Complex amplification factor, same length as ``frequency_array``.
    """
    try:
        lens = LENS_MODELS[lens_model]
    except KeyError:
        raise ValueError(
            f"Unknown lens model '{lens_model}'. "
            f"Registered models: {sorted(LENS_MODELS)}"
        )
 
    lens_parameters = {name: parameters.pop(name) for name in lens.parameters}
    y = lens_parameters["y"]
    w = lens.dimensionless_frequency(frequency_array, **lens_parameters)
 
    splines = _load_splines(lookup_table)
    return splines.real(w, y)[:, 0] + 1j * splines.imag(w, y)[:, 0]

"""
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
""" 

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

def lensed_lal_binary_neutron_star(frequency_array, mass_1, mass_2, luminosity_distance, a_1,tilt_1,
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