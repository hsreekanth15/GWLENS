# Lensed Signal Parameter Estimation

:::{admonition} Warning
:class: warning
GWLENS solver1d and solver2d uses `finufft` which requires CPU's that support AVX2/FMA instructions.
So if you submitting jobs to old CPUS your will stumble upon an error saying Illegal Instructution. So for such 
old CPU's we recommend using lookup table or find a better cpu to run!!!
:::


## Waveform

```{eval-rst}
.. autoclass:: gwlens.pe.waveform.LensedWaveformGenerator
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: gwlens.pe.waveform.LensModel
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: gwlens.pe.waveform.LookupTable
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autoclass:: gwlens.pe.waveform.LensingSettings
   :members:
   :show-inheritance:
```

## Main

## utils

## Parser

