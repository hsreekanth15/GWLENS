# GWLENS

GWLENS is a Python package for studying the gravitational lensing of gravitational waves. When a gravitational wave passes near a massive object such as a star, black hole, or galaxy, its path is bent and its amplitude and phase are modified. Because gravitational-wave wavelengths can be comparable to the size of the lens, geometric optics is not always enough, and wave effects like diffraction and interference become important. gwlens handles this by numerically solving the Fresnel–Kirchhoff diffraction integral to compute the frequency-dependent amplification factor, F(f), for a range of lens models. These amplification factors are then applied to unlensed waveforms and used with the Bilby inference library to search detector data for lensing signatures and estimate lens parameters. The package is being extended from transient compact-binary signals to continuous gravitational waves, such as those expected from rotating neutron stars.

## Installation

GWLENS is available in PyPI and can be installed with pip

`pip install gwlens`




```{toctree}
:maxdepth: 2
:caption: Contents:
installation
getting_started
rotated_tail_method
```

```{toctree}
:maxdepth: 2
:caption: Examples
axisymmetric_lenses
non-axisymmetric_lens
creating_your_lens
cbc_lensed_parameter_estimation
cw_datageneration
cw_microlensing
```


```{toctree}
:maxdepth: 2
:caption: API Reference
solver1d
solver2d
```
