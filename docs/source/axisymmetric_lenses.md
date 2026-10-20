# Axisymmetric lenses

Lens potentials $\psi(x)$ available in `solver1d`, with free parameters listed per model. Any analytic $\psi(x)$ can be supplied in addition to these (see {doc}`creating_your_lens`), and potentials can be added with `+` to build composite lenses. These potentials have identical forms in `solver2d`, with $x$ replaced by $\lvert\boldsymbol{x}-\boldsymbol{x}_0\rvert$.

| Model | Parameters | Function call | $\psi(x)$ |
|---|---|---|---|
| Point mass | $m$ | `point_mass` | $m\ln x$ |
| SIS | $\psi_0$ | `sis` | $\psi_0\,x$ |
| CIS | $\psi_0,\,x_c$ | `cis` | $\psi_0\left[r + x_c\ln\dfrac{2x_c}{r+x_c}\right]$ |
| NFW | $\psi_0,\,k$ | `nfw` | $\dfrac{\psi_0}{2}\left[\ln^{2}\dfrac{v}{2}+\arccos^{2}\dfrac{1}{v}\right]$ |
| Mass sheet | $\kappa_{\rm ext}$ | `mass_sheet` | $\dfrac{1}{2}\,\kappa_{\rm ext}\,x^{2}$ |
| Power law | $\psi_0,\,p$ | `power_law` | $\dfrac{\psi_0\,x^{p}}{p}$ |
| Plummer sphere | $\psi_0,\,x_c$ | `plummer` | $\dfrac{1}{2}\,\psi_0\,x_c^{2}\ln\left(1+\dfrac{x^{2}}{x_c^{2}}\right)$ |
| External shear$^{*}$ | $\gamma_1,\,\gamma_2,\,\kappa$ | `external_shear` | $\dfrac{1}{2}\left[\kappa\,(x_1^{2}+x_2^{2}) + \gamma_1\,(x_1^{2}-x_2^{2}) + 2\gamma_2\,x_1x_2\right]$ |

$^{*}$ Two-dimensional lens: $\psi$ depends on $(x_1,x_2)$ and not just on $\lvert x\rvert$.

Here $r=\sqrt{x^2+x_c^2}$ for the CIS and $v=kx$ for the NFW.

## References

- Point mass, SIS, CIS: Keeton (2001)
- NFW: Villarrubia-Rojo et al. (2024)