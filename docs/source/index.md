# GWLENS

GWLENS is a Python package for studying the gravitational lensing of gravitational waves. When a gravitational wave passes near a massive object such as a star, black hole, or galaxy, its path is bent and its amplitude and phase are modified. Because gravitational-wave wavelengths can be comparable to the size of the lens, geometric optics is not always enough, and wave effects like diffraction and interference become important. gwlens handles this by numerically solving the Fresnel–Kirchhoff diffraction integral to compute the frequency-dependent amplification factor, F(f), for a range of lens models. These amplification factors are then applied to unlensed waveforms and used with the Bilby inference library to search detector data for lensing signatures and estimate lens parameters. The package is being extended from transient compact-binary signals to continuous gravitational waves, such as those expected from rotating neutron stars.

```{image} _static/lensing_light.gif
:alt: Amplification factor of an SIS lens as the source position changes
:align: center
:width: 75%
```

::::{div} sd-text-center sd-my-4
```{button-ref} examples/solver1d_tutorial
:ref-type: doc
:color: primary
:shadow:
Get started with the tutorial
```
```{button-ref} modules
:ref-type: doc
:color: secondary
:outline:
Browse the API reference
```
::::

## Installation

GWLENS is available in PyPI and can be installed with pip

`pip install gwlens`

## Quick start

Compute the amplification factor $|F|$ of a singular isothermal sphere (SIS) for several source positions $y$:

```python
import numpy as np
import matplotlib.pyplot as plt
from gwlens import solver1d as s1d

ws = np.geomspace(0.05, 100, 400)          # dimensionless frequencies
fig, ax = plt.subplots()
for y in (0.2, 0.5, 0.9, 1.5):             # source positions
    F = s1d.amplification_many(ws, y, s1d.sis(1.0))
    ax.semilogx(ws, np.abs(F), label=f"$y={y}$")
ax.set(xlabel="$w$", ylabel="$|F|$", title="SIS ($\\psi_0 = 1$)")
ax.legend(); plt.show()
```

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
examples/solver1d_tutorial
examples/creating_lens
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
parameter_estimation
modules
```
