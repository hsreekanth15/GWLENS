"""
Amplification-factor lookup tables for GWLENS.

The lookup table is an HDF5 file with datasets:
    w_values : (Nw,)  dimensionless frequency grid (strictly increasing)
    y_values : (Ny,)  impact parameter grid (strictly increasing)
    results  : (Nw, Ny, 2)  real and imaginary parts of F(w, y)

The path of the table is *not* hardcoded anywhere. It is passed through the
bilby waveform_arguments (i.e. through the bilby_pipe ini file) and the
splines are built lazily and cached per-path, so the file is read exactly
once per process even though the source model is called at every likelihood
evaluation.
"""

from functools import lru_cache

import h5py
import numpy as np
from scipy.interpolate import RectBivariateSpline


@lru_cache(maxsize=8)
def load_amplification_splines(lookup_table_path):
    """Load an HDF5 lookup table and return (real_spline, imag_spline, bounds).

    Cached on the (absolute) path, so repeated calls are free.
    """
    with h5py.File(lookup_table_path, "r") as f:
        w_values = f["w_values"][:]
        y_values = f["y_values"][:]
        data = f["results"][:]  # shape (Nw, Ny, 2)

    real_spline = RectBivariateSpline(w_values, y_values, data[:, :, 0], kx=3, ky=3)
    imag_spline = RectBivariateSpline(w_values, y_values, data[:, :, 1], kx=3, ky=3)
    bounds = (w_values[0], w_values[-1], y_values[0], y_values[-1])
    return real_spline, imag_spline, bounds


def amplification_factor(w, y, lookup_table_path, clip_to_table=True):
    """Evaluate the complex amplification factor F(w, y) from a lookup table.

    Parameters
    ----------
    w: array_like
        Dimensionless frequency array.
    y: float
        Impact parameter.
    lookup_table_path: str
        Path to the HDF5 lookup table (from the ini file).
    clip_to_table: bool
        If True, clip (w, y) to the tabulated range instead of letting the
        spline extrapolate. Splines extrapolate badly, so this is safer; for
        w below the table minimum F -> 1 (geometric-optics/low-frequency
        limit) is usually the better choice, which clipping approximates if
        your table starts at small enough w.

    Returns
    -------
    complex ndarray with the same shape as ``w``.
    """
    real_spline, imag_spline, (w_min, w_max, y_min, y_max) = load_amplification_splines(lookup_table_path)

    w = np.asarray(w, dtype=float)
    if clip_to_table:
        w_eval = np.clip(w, w_min, w_max)
        y_eval = float(np.clip(y, y_min, y_max))
    else:
        w_eval, y_eval = w, float(y)

    # .ev broadcasts a scalar y against the w array (point-wise evaluation)
    return real_spline.ev(w_eval, y_eval) + 1j * imag_spline.ev(w_eval, y_eval)
