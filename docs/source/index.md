# GWLENS

```{raw} html
<p class="gw-tagline"><b>G</b>ravitational <b>W</b>ave <b>L</b>ensing <b>E</b>xploration of <b>N</b>on-visible <b>S</b>tructures</p>
```

GWLENS is a Python package for studying the gravitational lensing of gravitational waves. When a gravitational wave passes near a massive object such as a star, black hole, or galaxy, its path is bent and its amplitude and phase are modified. Because gravitational-wave wavelengths can be comparable to the size of the lens, geometric optics is not always enough, and wave effects like diffraction and interference become important. GWLENS handles this by numerically solving the Fresnel–Kirchhoff diffraction integral to compute the frequency-dependent amplification factor, F(f), for a range of lens models. These amplification factors are then applied to unlensed waveforms and used with the Bilby inference library to search detector data for lensing signatures and estimate lens parameters. The package is being extended from transient compact-binary signals to continuous gravitational waves, such as those expected from rotating neutron stars.

```{raw} html
<div class="gw-figures">
  <div class="gw-card">
    <video autoplay muted loop playsinline preload="metadata"
           poster="_static/gwlens_binary_poster.png"
           aria-label="Equal-mass binary point lens amplification">
      <source src="_static/gwlens_binary.mp4" type="video/mp4">
    </video>
  </div>
  <div class="gw-card">
    <video autoplay muted loop playsinline preload="metadata"
           poster="_static/gwlens_plummer_poster.png"
           aria-label="Plummer lens amplification">
      <source src="_static/gwlens_plummer.mp4" type="video/mp4">
    </video>
  </div>
</div>
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

## Quick install

```bash
pip install gwlens
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Contents

installation
getting_started
rotated_tail_method
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Examples

examples/solver1d_tutorial
examples/creating_lens
axisymmetric_lenses
non-axisymmetric_lens
cbc_lensed_parameter_estimation
cw_datageneration
cw_microlensing
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: API Reference

solver1d
solver2d
parameter_estimation
modules
```