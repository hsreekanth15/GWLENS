import bilby
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from gwcosmo import priors as p


class PLGaussian(bilby.core.prior.Prior):
    def __init__(self, name=None, latex_label=None, unit=None):
        super().__init__(name=name, latex_label=latex_label, unit=unit)
       
        model = p.BBH_powerlaw_gaussian()
        samples = model.sample(10000)[0]

        self.samples = np.sort(samples)
        self.kde = gaussian_kde(self.samples)
        
        # Empirical CDF calculated from samples
        self.cdf_values = np.linspace(0, 1, len(self.samples))
        self.inverse_cdf = interp1d(self.cdf_values, self.samples, kind='linear', fill_value="extrapolate")
        
        # Minimum and maximum values for support
        self.minimum = np.min(samples)
        self.maximum = np.max(samples)
    
    def prob(self, val):
        # Probability density from KDE
        return self.kde(val)[0]
    
    def cdf(self, val):
        # Compute CDF using the integral of the KDE
        idx = np.searchsorted(self.samples, val, side='right')
        if idx > 0 and idx < len(self.cdf_values):
            return self.cdf_values[idx]
        elif idx == 0:
            return 0.0
        else:
            return 1.0

    def rescale(self, val):
        # Rescale using the inverse CDF
        return self.inverse_cdf(val)



class PLGaussianTest(bilby.core.prior.Prior):
    def __init__(self, 
                 name=None, 
                 latex_label=None, 
                 unit=None,
                 mminbh=4.98, 
                 mmaxbh=112.5, 
                 alpha=3.78, 
                 mu_g=32.27, 
                 sigma_g=3.88, 
                 lambda_peak=0.03, 
                 delta_m=4.8, 
                 beta=0.81,
                 nsamples=10000):
        """
        Bilby Prior for the BBH PowerLaw + Gaussian model.

        Parameters
        ----------
        mminbh, mmaxbh, alpha, mu_g, sigma_g, lambda_g, delta_m, beta : float
            Parameters defining the BBH powerlaw+gaussian model (see 2111.03604).
        nsamples : int
            Number of Monte Carlo samples to approximate the distribution.
        """

        super().__init__(name=name, latex_label=latex_label, unit=unit)

        # Build the astrophysical prior model
        model = p.BBH_powerlaw_gaussian(
            mminbh=mminbh, mmaxbh=mmaxbh, alpha=alpha, mu_g=mu_g, sigma_g=sigma_g,
            lambda_peak=lambda_peak, delta_m=delta_m, beta=beta
        )

        # Draw samples for KDE + empirical CDF
        samples = model.sample(nsamples)[0]   # Assuming model.sample returns (samples, weights)
        self.samples = np.sort(samples)

        # KDE for PDF evaluation
        self.kde = gaussian_kde(self.samples)

        # Build inverse CDF from empirical distribution
        self.cdf_values = np.linspace(0, 1, len(self.samples))
        self.inverse_cdf = interp1d(self.cdf_values, self.samples, kind="linear", fill_value="extrapolate")

        # Support bounds
        self.minimum = np.min(self.samples)
        self.maximum = np.max(self.samples)

    def prob(self, val):
        """Probability density function (PDF)."""
        return self.kde(val)

    def cdf(self, val):
        """Cumulative distribution function (CDF)."""
        val = np.atleast_1d(val)
        idx = np.searchsorted(self.samples, val, side="right")
        result = np.zeros_like(val, dtype=float)
        result[idx > 0] = self.cdf_values[np.clip(idx[idx > 0]-1, 0, len(self.cdf_values)-1)]
        result[idx >= len(self.samples)] = 1.0
        return result if len(result) > 1 else result[0]

    def rescale(self, val):
        """Inverse CDF (maps [0,1] -> mass values)."""
        return self.inverse_cdf(val)
