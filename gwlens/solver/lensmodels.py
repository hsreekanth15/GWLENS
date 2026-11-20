"""
This lens model registry is specially made for Fitting factot analysis....
There are plans to migrate the lens models in solver1d, solver2d and PE/waveform.py to this file for organizational purpose.
Until the migration is successful please use these models only for fitting factor analysis.
"""
import numpy as np
import inspect
from . import solver1d, solver2d
from pycbc.types import FrequencySeries

TSUN = 4.925490947641267e-06 


def dimensionless_frequency(f, MLz):
    """
    The MLz to dimensionless frequency conversion valid for all lens models,
    where MLz changes from one model to another.
    w = 8 pi G MLz f / c^3, MLz in solar masses (redshifted lens mass), f in Hz.

    Parameters:
    -----------

    
    Returns:
    w
    
    """
    return 8.0 * np.pi * TSUN * MLz * np.asarray(f, dtype=float)


def spectrum_1d(w, y, lens, **kw):
    """
    F at frequencies w for an axisymmetric solver1d lens and source position y.
    
    """
    f = solver1d.amplification
    return f(np.asarray(w, dtype=float), float(y), lens, **kw)


def spectrum_2d(w, y, lens, **kw):
    """
    F at frequencies w for a solver2d lens (any macro image type) and source position y = (y1, y2).
    Type I, II and III images are handled internally by the solver2d with support from macro.py
    """
    f = solver2d.amplification
    return f(np.asarray(w, dtype=float), (float(y[0]), float(y[1])), lens, **kw)


class LensModel:
    """
    A lens model: func(w, **params) -> F(w), with the names of its parameters (besides MLz) and
    optional default values for some of them. 
    Allows unpacking other lens parameters
    A function with an argument named MLz also receives the lens mass (e.g. to convert physical halo
    properties into lengths in units of xi0); MLz is not counted among its lens parameters.
    
    """

    def __init__(self, func, parameters=None, defaults=None, name=None):

        self.func = func
        sig = list(inspect.signature(func).parameters.values())[1:]
        self.needs_MLz = any(p.name == "MLz" for p in sig)        # func(w, MLz, ...) gets the lens mass
        if parameters is None:                    
            parameters = [p.name for p in sig if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                          and p.name != "MLz"]
            defaults = {**{p.name: p.default for p in sig if p.default is not p.empty and p.name != "MLz"},
                        **(defaults or {})}
        
        self.parameters = tuple(p for p in parameters if p != "MLz")
        self.defaults = dict(defaults or {})
        self.name = name or getattr(func, "__name__", "lens")

    @property
    def all_parameters(self):
        return ("MLz",) + self.parameters

    def resolve(self, params):
        """
        The lens parameters of params (a dict, extra keys ignored), defaults filled in.
        """
        out = {}
        for p in self.parameters:
            if p in params:
                out[p] = params[p]
            elif p in self.defaults:
                out[p] = self.defaults[p]
            else:
                raise ValueError(f"lens model '{self.name}' needs '{p}' (parameters: {', '.join(self.all_parameters)})")
        return out

    def F_of_w(self, w, MLz=None, **params):
        """
        F at dimensionless frequencies w Takahashi convention.
        MLz is passed on to the lens function if it takes it.
        """
        kw = self.resolve(params)
        if self.needs_MLz:
            if MLz is None:
                raise ValueError(f"lens model '{self.name}' needs MLz")
            kw["MLz"] = MLz
        return np.asarray(self.func(np.asarray(w, dtype=float), **kw), dtype=complex)

    def amplification(self, f, MLz, f_lower=0.0, f_upper=np.inf, convention="lal", **params):
        """
        F at frequencies f [Hz] for redshifted lens mass MLz [Msun]; F = 1 outside [f_lower, f_upper] and at
        f <= 0. convention='lal': conj(F), the factor that multiplies a LAL / PyCBC / bilby waveform
        h(f) = int h(t) exp(-2 pi i f t) dt; convention='gwlens': F as the solvers return it.
        """

        f = np.asarray(f, dtype=float)
        F = np.ones(f.shape, dtype=complex)
        band = (f > 0) & (f >= f_lower) & (f <= f_upper)
        if band.any():
            F[band] = self.F_of_w(dimensionless_frequency(f[band], MLz), MLz=MLz, **params)
        if convention == "lal":
            return np.conj(F)
        if convention == "default":
            return F
        raise ValueError("convention must be 'lal' or 'default'")

    def __repr__(self):
        return f"LensModel('{self.name}', parameters={self.all_parameters})"


_REGISTRY = {}


def register(name, parameters=None, defaults=None):
    """Decorator: register func(w, **params) as lens model `name` (parameters: from the signature)."""
    def deco(func):
        _REGISTRY[name] = LensModel(func, parameters, defaults, name=name)
        return func
    return deco


def get_lens_model(model, parameters=None, defaults=None):
    """
    A LensModel from a registered name, a LensModel, or a function func(w, **params).
    """

    if isinstance(model, LensModel):
        return model
    
    if isinstance(model, str):
        if model not in _REGISTRY:
            raise ValueError(f"unknown lens model '{model}'; available: {', '.join(sorted(_REGISTRY))}")
        return _REGISTRY[model]
    
    if callable(model):
        return LensModel(model, parameters, defaults)
    raise TypeError("model must be a name, a LensModel or a function func(w, **params)")


def available_lens_models():
    """{name: parameters} of the registered lens models."""
    return {k: v.all_parameters for k, v in sorted(_REGISTRY.items())}


def lens_frequency_series(h, model, params, f_lower=0.0, f_upper=np.inf, convention="lal"):
    """
    A PyCBC FrequencySeries h(f) multiplied by the lens model's factor (see LensModel.amplification);
    params holds MLz and the lens parameters.
    """
    
    m = get_lens_model(model)
    lens = {k: v for k, v in params.items() if k != "MLz"}
    F = m.amplification(h.sample_frequencies.numpy(), params["MLz"], f_lower, f_upper, convention, **lens)
    return FrequencySeries(h.numpy() * F, delta_f=h.delta_f, epoch=h.epoch)


#built-in lens models
@register("point")
def point(w, y):
    return spectrum_1d(w, y, solver1d.point_mass(1.0))


@register("sis")
def sis(w, y):
    return spectrum_1d(w, y, solver1d.sis(1.0))


@register("cis")
def cis(w, y, rc):
    return spectrum_1d(w, y, solver1d.cis(1.0, rc))


@register("nfw", defaults={"k": 1.0})
def nfw(w, y, kappa_s, k=1.0):
    # psi0 = 4 kappa_s / k^2 keeps kappa_s the NFW characteristic convergence for any k = xi0 / r_s
    return spectrum_1d(w, y, solver1d.nfw(k=k, psi0=4.0 * kappa_s / k ** 2))


@register("point_shear", defaults={"gamma2": 0.0})
def point_shear(w, y1, y2, kappa, gamma1, gamma2=0.0):
    return spectrum_2d(w, (y1, y2), solver2d.point_mass(1.0) + solver2d.external_shear(gamma1, gamma2, kappa))


@register("sis_shear", defaults={"gamma2": 0.0})
def sis_shear(w, y1, y2, kappa, gamma1, gamma2=0.0):
    return spectrum_2d(w, (y1, y2), solver2d.sis(1.0) + solver2d.external_shear(gamma1, gamma2, kappa))


@register("nfw_shear", defaults={"gamma2": 0.0, "k": 1.0})
def nfw_shear(w, y1, y2, kappa, gamma1, kappa_s, gamma2=0.0, k=1.0):
    lens = solver2d.nfw(k=k, psi0=4.0 * kappa_s / k ** 2) + solver2d.external_shear(gamma1, gamma2, kappa)
    return spectrum_2d(w, (y1, y2), lens)