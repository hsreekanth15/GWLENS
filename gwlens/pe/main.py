"""
GWLENS frequency-domain source models for bilby / bilby_pipe.

Each lens model gets its own explicitly-signed wrapper around
``bilby.gw.source.lal_binary_black_hole``. The explicit signatures matter:
bilby's WaveformGenerator infers the sampled parameters from the function
signature, so the lens parameters (MLz, y / sigma_v, Dl, Ds, y, ...) must be
named arguments, not hidden in **kwargs.

Everything that is *not* a sampled parameter (the lookup-table path, the
approximant, ...) arrives through **kwargs, i.e. through bilby_pipe's
``waveform-arguments-dict``, which gwlens_pipe fills in from the ini file.
"""

import numpy as np
from scipy.constants import G, c

from bilby.gw.source import lal_binary_black_hole

from .lookup import amplification_factor

SOLAR_MASS = 1.98892e30  # kg

# w = 8 * pi * G * MLz[Msun] * f / c^3  (point lens dimensionless frequency)
POINT_LENS_W_CONSTANT = 8 * np.pi * G * SOLAR_MASS / c ** 3

DEFAULT_WAVEFORM_KWARGS = dict(
    waveform_approximant="IMRPhenomXAS",
    reference_frequency=50.0,
    minimum_frequency=20.0,
    catch_waveform_errors=False,
    pn_spin_order=-1,
    pn_tidal_order=-1,
    pn_phase_order=-1,
    pn_amplitude_order=0,
)

# Keys injected by gwlens_pipe that must not be forwarded to lalsimulation
_GWLENS_KWARGS = ("lens_lookup_table", "lens_model")


def _pop_lensing_kwargs(kwargs):
    lookup_table = kwargs.pop("lens_lookup_table", None)
    kwargs.pop("lens_model", None)
    if lookup_table is None:
        raise ValueError(
            "GWLENS source model called without 'lens_lookup_table' in the "
            "waveform arguments. Set 'lens-lookup-table' in your ini file "
            "(or 'lens_lookup_table' in waveform-arguments-dict)."
        )
    return lookup_table


def _unlensed_waveform(frequency_array, mass_1, mass_2, luminosity_distance,
                       a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn,
                       phase, kwargs):
    waveform_kwargs = dict(DEFAULT_WAVEFORM_KWARGS,
                           maximum_frequency=frequency_array[-1])
    waveform_kwargs.update(kwargs)
    return lal_binary_black_hole(
        frequency_array=frequency_array, mass_1=mass_1, mass_2=mass_2,
        luminosity_distance=luminosity_distance, theta_jn=theta_jn,
        phase=phase, a_1=a_1, a_2=a_2, tilt_1=tilt_1, tilt_2=tilt_2,
        phi_12=phi_12, phi_jl=phi_jl, **waveform_kwargs)


def _apply_amplification(waveform_dict, F):
    if waveform_dict is None:  # lal error caught upstream
        return None
    for mode in ("plus", "cross"):
        waveform_dict[mode] = waveform_dict[mode] * F
    return waveform_dict


def point_lensed_lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance,
        a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
        MLz, y, **kwargs):
    """Point-mass lens. Extra sampled parameters: MLz [Msun], y."""
    lookup_table = _pop_lensing_kwargs(kwargs)
    w = POINT_LENS_W_CONSTANT * MLz * frequency_array
    F = amplification_factor(w, y, lookup_table)
    waveform_dict = _unlensed_waveform(
        frequency_array, mass_1, mass_2, luminosity_distance,
        a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, kwargs)
    return _apply_amplification(waveform_dict, F)


def sis_lensed_lal_binary_black_hole(
        frequency_array, mass_1, mass_2, luminosity_distance,
        a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
        sigma_v, Dl, Ds, y, **kwargs):
    """SIS lens. Extra sampled parameters: sigma_v [km/s], Dl, Ds [kpc], y."""
    # Adjust this import/call to match your gwlens.lens.sis API
    from gwlens.lens.sis import SIS
    lookup_table = _pop_lensing_kwargs(kwargs)
    w = SIS.dimensionless_frequency(frequency_array, sigma_v=sigma_v,
                                    Dl=Dl, Ds=Ds)
    F = amplification_factor(w, y, lookup_table)
    waveform_dict = _unlensed_waveform(
        frequency_array, mass_1, mass_2, luminosity_distance,
        a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, kwargs)
    return _apply_amplification(waveform_dict, F)


# Registry: ini-file value of `lens-model`  ->  dotted path of the source model.
# bilby_pipe stores the source model as a string, so we register dotted paths.
LENS_MODELS = {
    "point": "gwlens.source.point_lensed_lal_binary_black_hole",
    "sis": "gwlens.source.sis_lensed_lal_binary_black_hole",
}

# Lens parameters each model expects in the prior file (used for validation)
LENS_MODEL_PARAMETERS = {
    "point": ["MLz", "y"],
    "sis": ["sigma_v", "Dl", "Ds", "y"],
}