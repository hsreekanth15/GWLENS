# GWLENS-  Gravitational Wave Lensing Exploration of Non-visible Structures



The documentation can be found here: https://hsreekanth15.github.io/GWLENS/

GWLENS is a Python package for studying the gravitational lensing of gravitational waves. When a gravitational wave passes near a massive object such as a star, black hole, or galaxy, its path is bent and its amplitude and phase are modified. Because gravitational-wave wavelengths can be comparable to the size of the lens, geometric optics is not always enough, and wave effects like diffraction and interference become important. GWLENS handles this by numerically solving the Fresnel–Kirchhoff diffraction integral to compute the frequency-dependent amplification factor, F(f), for a range of lens models. These amplification factors are then applied to unlensed waveforms and used with the Bilby inference library to search detector data for lensing signatures and estimate lens parameters. The package is being extended from transient compact-binary signals to continuous gravitational waves, such as those expected from rotating neutron stars.


