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
