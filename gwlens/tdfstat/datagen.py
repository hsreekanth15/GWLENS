#!/usr/bin/env python3
"""
datagen.py - generate ephemeris data for multiple segments and detectors

Python port of datagen.c + get_barycenter.c (Sreekanth Harikumar).
Adapted from the original C code in TDFstat in particular the ephemeris generation and Gaussian noise generation.

Tasks:
  - reads a config .ini (section [tdfstat], plus [cwsignal] when write_signal = true
    and [lensing] when lensing = true)
  - for each segment / detector:
      * picks the PSD file from psd-dict, interpolates it at the band frequency
      * computes the detector ephemeris (DetSSB.bin, rDet.bin, rSSB.bin)
      * writes Gaussian noise to xdat_<seg>_<band>.bin; if write_signal = true,
        the CW signal (sigen.TimeSeries) is added to the noise before writing
      * if lensing = true, the signal is multiplied by the amplification factor
        |F| and its phase shifted by arg(F) before it goes into the noise
  - advances the GPS time by one sidereal day * nod per segment

Barycentring follows get_barycenter.c exactly: per time sample, XLALBarycenter is
evaluated along the three sky directions x^=(0,0), y^=(pi/2,0), z^=(0,pi/2), and
  DetSSB[3i+k] = emit.deltaT / dt     (detector wrt SSB, k-th component)
  rDet[3i+k]   = emit.erot  / dt      (Earth-rotation/diurnal part)
The detector geocentric XYZ and geographic position are the hardcoded values from
get_barycenter.c / get_position.

Only the sidereal phase `sid` cannot be matched bit-for-bit yet (sid.c not in hand):
LAL's GreenwichMeanSiderealTime is used as a correct stand-in -- see sid() note.

Requires:  numpy, scipy, lalsuite   (pip install numpy scipy lalsuite)
Run:       gwlens_cw_datagen config.ini
"""

import argparse
import ast
import operator
import os
import sys
import math
import errno
import configparser
import h5py
from types import SimpleNamespace
from . import constants
import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline
from gwlens.tools.utilities import spline_eval_chunked
from gwlens import microlensing

import lal
import lalpulsar

try:                        # datagen imported as part of a package
    from .sigen import TimeSeries
except ImportError:         # run directly: python datagen.py config.ini
    from sigen import TimeSeries

# ----------------------------------------------------------------------------
# Constants (mirror the #defines in datagen.c / EphemerisDetector.h)
# ----------------------------------------------------------------------------
EPSILON = constants.EPSILON
SIDEREAL_DAY = constants.SIDEREAL_DAY
C_SI = constants.C_SI
_DEG = constants._DEG

# Detector geocentric XYZ [metres] -- hardcoded exactly as in get_barycenter.c.
_H = (-2161414.92636,   -3834695.17889,    4600350.22664)
_L = (-74276.04472380,  -5496283.71971000, 3224257.01744000)
_V = (4546374.09900000,  842989.69762600,  4378576.96241000)
_SITE_XYZ = {
    "H1": _H, "H2": _H, "LHO": _H,
    "L1": _L, "LLO": _L,
    "V1": _V, "VIRGO": _V,
}

# Detector enum names, in the same order as names[] in get_barycenter.c in the c verison of tdfstat
_NAMES = [
    "none", "GB", "NA", "AO", "HO", "PR", "TD", "PK", "JB", "G3", "G1RAD",
    "G8", "VL", "BO", "MO", "NC", "EF", "FB", "H1", "H2", "LHO", "LLO",
    "L1", "GEO", "G1", "V1", "VIRGO", "TAMA", "T1",
]

# Signal parameters read from the [cwsignal] section (all floats).
_SIGNAL_KEYS = (
    "h0", "frequency", "spin_down", "ra", "dec",
    "wobble_angle", "inclination", "polarization", "reference_phase",
)
_SIGNAL_DEFAULTS = {
    "wobble_angle": math.pi / 2,   # sin^2 = 1, so h0 is the plain amplitude
    "reference_phase": 0.0,
}


# ----------------------------------------------------------------------------
# Detector helpers  (get_detector / get_position)
# ----------------------------------------------------------------------------
def get_detector(name):
    """Match a detector name against the names[] table; return the canonical
    name, or 'none' if unknown (mirrors get_detector returning enum 0)."""
    name = name.strip()
    return name if name in _NAMES and name != "none" else "none"


def get_position(detector):
    """Geographic position of a detector, exactly as in get_position():
        [0] latitude [rad], [1] longitude [rad], [2] height [m], [3] orientation [rad].
    """
    deg = _DEG  # = 180/pi; dividing a degree-valued expression by deg gives radians
    if detector in ("L1", "LLO"):
        ephi = (30 + (33 + 46.4196 / 60.0) / 60.0) / deg
        elam = -(90 + (46 + 27.2654 / 60.0) / 60.0) / deg
        eheight = -6.574
        egam = 242.7165 / deg
    elif detector in ("LHO", "H1", "H2"):
        ephi = (46 + (27 + 18.528 / 60.0) / 60.0) / deg
        elam = -(119 + (24 + 27.5657 / 60.0) / 60.0) / deg
        eheight = 142.554
        egam = 170.9994 / deg
    elif detector in ("VIRGO", "V1"):
        ephi = 0.76151183984
        elam = 0.18333805213
        eheight = 51.884
        x_azi = 0.33916285222
        y_azi = 5.05155183261
        egam = math.pi / 2 - 0.5 * (x_azi + y_azi)
    else:
        sys.stderr.write(f"Detector {detector}: unknown position\n")
        ephi = elam = eheight = egam = 0.0
    return np.array([ephi, elam, eheight, egam])


# ----------------------------------------------------------------------------
# Time helpers  (gps2mjd / sid)
# ----------------------------------------------------------------------------
def gps2mjd(gps_time):
    """GPS seconds -> MJD (GPS epoch = MJD 44244.0, no leap-second handling).
    Kept for parity with the C call site; adjust if your sid.c differs."""
    return 44244.0 + gps_time / 86400.0


def sid(gps_time, elam):
    """Local mean sidereal phase phir [rad].

    NOTE (sid.c not yet ported): this returns LAL's Greenwich Mean Sidereal Time
    plus the detector east-longitude `elam`, wrapped to [0, 2*pi). It is a correct
    sidereal phase, but if your sid.c uses a different epoch/offset convention the
    two trailing scalars in DetSSB.bin (phir, EPSILON) will differ. Paste sid.c
    and this one function can be made bit-for-bit identical.
    """
    gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(gps_time))  # radians
    return float(np.mod(gmst + elam, 2.0 * np.pi))


# ----------------------------------------------------------------------------
# Directory creation  (mkdir_p)
# ----------------------------------------------------------------------------
def mkdir_p(path):
    try:
        os.makedirs(path, mode=0o755, exist_ok=True)
        return 0
    except OSError as e:
        if e.errno != errno.EEXIST:
            sys.stderr.write(f"mkdir: {e}\n")
            return -1
        return 0


# ----------------------------------------------------------------------------
# PSD interpolation  (interpolate_psd)
# ----------------------------------------------------------------------------
def interpolate_psd(filename, freq_query):
    """Cubic-spline interpolate a 2-column (freq, psd) ASCII file at freq_query."""
    try:
        data = np.loadtxt(filename)
    except OSError as e:
        sys.stderr.write(f"Error opening PSD file: {e}\n")
        return -1.0

    if data.ndim != 2 or data.shape[0] < 2:
        sys.stderr.write("Error: need at least 2 points for interpolation.\n")
        return -1.0

    freqs, psds = data[:, 0], data[:, 1]
    if freq_query < freqs[0] or freq_query > freqs[-1]:
        sys.stderr.write(
            f"Warning: frequency {freq_query:.3f} is outside interpolation "
            f"range [{freqs[0]:.3f}, {freqs[-1]:.3f}].\n"
        )
    return float(CubicSpline(freqs, psds)(freq_query))


# ----------------------------------------------------------------------------
# Gaussian data  (generate_gaussian_data)
# ----------------------------------------------------------------------------
def generate_gaussian_data(N, amp, sigma, dataout, maskfile=None, signal=None):
    """Write N float32 samples amp*N(0, sigma) to the file path `dataout`.

    If `signal` is given (a length-N array in the same units as the noise),
    it is added to the noise before masking and writing, so the file holds
    noise + signal. Samples are zeroed where |mask| < 1e-30.
    Returns 0 on success, 1 on failure.
    """
    if N <= 0:
        sys.stderr.write(f"Invalid N: {N}\n")
        return 1

    seed = int.from_bytes(os.urandom(4), "little")   # like /dev/urandom seeding
    rng = np.random.default_rng(seed)
    x = (amp * rng.normal(0.0, sigma, size=N)).astype(np.float32)

    if signal is not None:
        if len(signal) != N:
            sys.stderr.write(
                f"Signal length {len(signal)} does not match N = {N}\n"
            )
            return 1
        x += np.asarray(signal, dtype=np.float32)

    if maskfile is not None:
        try:
            zmask = np.fromfile(maskfile, dtype=np.float32, count=N)
        except OSError as e:
            sys.stderr.write(f"Opening mask file: {e}\n")
            return 1
        if zmask.size != N:
            sys.stderr.write(
                f"Warning: expected {N} floats, read {zmask.size} in mask file.\n"
            )
        else:
            print(f"Mask file read successfully ({zmask.size} floats).")
        m = np.zeros(N, dtype=np.float32)
        m[: zmask.size] = zmask
        x[np.abs(m) < 1e-30] = 0.0

    try:
        x.tofile(dataout)
    except OSError as e:
        sys.stderr.write(f"Opening output file: {e}\n")
        return 1

    kind = "noise + signal" if signal is not None else "noise"
    print(f"Data written to {dataout} ({N} samples, {kind})")
    return 0


# ----------------------------------------------------------------------------
# Reference frequency  (get_fpo)
# ----------------------------------------------------------------------------
def get_fpo(bandnumber, bandwidth, ov):
    """fpo = 10 + (1 - 2^-ov) * bandnumber * bandwidth."""
    return 10.0 + (1.0 - 2.0 ** (-ov)) * bandnumber * bandwidth


def suggest_bands(freq, bandwidth, ov):
    """Find the band numbers whose range [fpo, fpo + bandwidth] contains freq.

    Neighbouring bands overlap, so a frequency can lie in more than one.
    Returns (best, candidates), where best is the band in which freq sits
    closest to the centre, or (None, []) if no band contains freq
    (bands start at 10 Hz).

    A better version than the one used prior to the GWLENS implementation.
    """
    step = (1.0 - 2.0 ** (-ov)) * bandwidth        # fpo spacing between bands
    if step <= 0.0 or freq < 10.0:
        return None, []

    lo = max(0, math.floor((freq - 10.0 - bandwidth) / step))
    hi = math.ceil((freq - 10.0) / step)
    candidates = [
        b for b in range(lo, hi + 1)
        if get_fpo(b, bandwidth, ov) <= freq <= get_fpo(b, bandwidth, ov) + bandwidth
    ]
    if not candidates:
        return None, []

    best = min(
        candidates,
        key=lambda b: abs(get_fpo(b, bandwidth, ov) + 0.5 * bandwidth - freq),
    )
    return best, candidates


# ----------------------------------------------------------------------------
# Barycentring  (get_barycenter)
# ----------------------------------------------------------------------------
def get_barycenter(gps1, site, edat, N, dt):
    """For N samples from gps1 (step dt), evaluate XLALBarycenter along x^, y^, z^.

    Returns
    -------
    DetSSB : (3N,) float64   DetSSB[3i+k] = emit.deltaT / dt
    rDet   : (3N,) float64   rDet[3i+k]   = emit.erot   / dt
    """
    DetSSB = np.empty(3 * N, dtype=np.float64)
    rDet = np.empty(3 * N, dtype=np.float64)

    baryinput = lalpulsar.BarycenterInput()
    earth = lalpulsar.EarthState()
    emit = lalpulsar.EmissionTime()

    xyz = _SITE_XYZ.get(site)
    if xyz is None:
        sys.stderr.write(
            "No coordinates found for detector site, using defaults\n"
        )
        xyz = (0.0, 0.0, 0.0)
    # positions in light seconds
    baryinput.site.location[0] = xyz[0] / C_SI
    baryinput.site.location[1] = xyz[1] / C_SI
    baryinput.site.location[2] = xyz[2] / C_SI

    # The three sky directions probed, matching get_barycenter.c.
    directions = ((0.0, 0.0), (math.pi / 2, 0.0), (0.0, math.pi / 2))

    for i in range(N):
        gps = gps1 + i * dt
        sec = int(math.floor(gps))                 # split into s + ns as in C
        ns = int((gps - sec) * 1.0e9)
        baryinput.tgps = lal.LIGOTimeGPS(sec, ns)
        lalpulsar.BarycenterEarth(earth, baryinput.tgps, edat)

        for k, (alpha, delta) in enumerate(directions):
            baryinput.alpha = alpha
            baryinput.delta = delta
            baryinput.dInv = 0.0
            lalpulsar.Barycenter(emit, baryinput, earth)
            DetSSB[3 * i + k] = emit.deltaT / dt
            rDet[3 * i + k] = emit.erot / dt

    return DetSSB, rDet


def generate_ephemeris_segment(gps_start, ndays, detector_name, output_dir, N, bandwidth, earth_ephem, sun_ephem):
    """Build DetSSB/rDet/rSSB for one segment+detector and write the .bin files.

    Returns (DetSSB, phir) so the signal generator can reuse them, where DetSSB
    is the flat (3N,) array in units of dt. Returns None on failure.
    """
    dt = 1.0 / (2.0 * bandwidth)

    detector = get_detector(detector_name)

    if mkdir_p(output_dir) != 0:
        sys.stderr.write(f"Failed to create directory {output_dir}\n")
        return None

    position = get_position(detector)
    elam = position[1]
    # C: mjd1 = gps2mjd(gps_start); phir = sid(mjd1, elam).
    # See sid() note -- using LAL GMST(gps) as a correct stand-in.
    phir = sid(gps_start, elam)

    edat = lalpulsar.InitBarycenter(earth_ephem,sun_ephem)

    DetSSB, rDet = get_barycenter(gps_start, detector, edat, N, dt)

    # DetSSB.bin holds 3N values plus the two trailing scalars (phir, epsilon).
    detssb_out = np.empty(3 * N + 2, dtype=np.float64)
    detssb_out[: 3 * N] = DetSSB
    detssb_out[3 * N] = phir
    detssb_out[3 * N + 1] = EPSILON

    rSSB = DetSSB - rDet   # rSSB[j] = DetSSB[j] - rDet[j]

    for fname, arr in (
        ("DetSSB.bin", detssb_out),
        ("rDet.bin", rDet),
        ("rSSB.bin", rSSB),
    ):
        path = os.path.join(output_dir, fname)
        try:
            arr.astype(np.float64).tofile(path)
        except OSError:
            sys.stderr.write(f"Error opening file {path}\n")

    return DetSSB, phir


# ----------------------------------------------------------------------------
# Signal generation (bridge to sigen.TimeSeries)
# ----------------------------------------------------------------------------
def build_detector(det_name, segment, band, ov, gps_start, N, bandwidth, fpo,
                   DetSSB, phir, strain=None):
    """Wrap one segment's data in the detector object the original
    sigen.TimeSeries expects.

    Units: time in seconds since gps_start; DetSSB converted back from units of
    dt (as stored in DetSSB.bin) to light-seconds and reshaped to (N, 3).
    With the usual bandwidth = 0.5 Hz, dt = 1 s, so seconds and samples coincide.
    TimeSeries adds its signal into `strain` in place; with the default (None)
    that is a zero array, so afterwards it holds the pure signal.
    """
    if strain is None:
        strain = np.zeros(N, dtype=np.float64)
    dt = 1.0 / (2.0 * bandwidth)
    latitude, _longitude, _height, gamma = get_position(get_detector(det_name))
    t = np.arange(N, dtype=np.float64) * dt

    return SimpleNamespace(
        name=det_name,
        segment=segment,
        band=band,
        overlap=ov,
        start_time=gps_start,
        delta_t=dt,
        duration=N * dt,
        starting_frequency=fpo,
        sampling_frequency=1.0 / dt,
        bandwidth=bandwidth,
        time_array=t,
        time_square_array=t ** 2,
        t_plus_t2_over_2=0.5 * t ** 2,
        ephemeris=SimpleNamespace(DetSSB=DetSSB.reshape(N, 3) * dt, phir=phir),
        # sigen reads this attribute as the site angle in the JKS a(t), b(t)
        # formulas, where the angle is the detector LATITUDE. The attribute name
        # is kept for compatibility; the value passed is the latitude.
        longitude_radians=latitude,
        orientation_radians=gamma,
        time_domain_strain=strain,
    )


# Safe parsing of numbers such as "np.pi/2" in the .ini file (no eval()).
_EXPR_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}
_EXPR_MODULES = ("np", "numpy", "math")
_EXPR_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow,
}
_EXPR_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def parse_number(text):
    """Parse a plain number or a simple arithmetic expression into a float.

    Accepts numbers, the constants pi / e / tau (optionally written as np.pi,
    numpy.pi or math.pi), + - * / ** and parentheses, e.g. "np.pi/2",
    "-0.25*pi", "1e-24". Anything else raises ValueError.
    """
    def _eval(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _EXPR_BINOPS:
            return _EXPR_BINOPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _EXPR_UNARYOPS:
            return _EXPR_UNARYOPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.Name) and node.id in _EXPR_CONSTANTS:
            return _EXPR_CONSTANTS[node.id]
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in _EXPR_MODULES and node.attr in _EXPR_CONSTANTS):
            return _EXPR_CONSTANTS[node.attr]
        raise ValueError("unsupported expression")

    try:
        return float(_eval(ast.parse(text.strip(), mode="eval").body))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as e:
        raise ValueError(f"cannot parse {text!r} as a number ({e})") from None


def read_signal_params(config):
    """Read the [cwsignal] section into a dict of floats, or None on error."""
    if not config.has_section("cwsignal"):
        sys.stderr.write("write_signal = true but there is no [cwsignal] section.\n")
        return None
    sec = config["cwsignal"]
    params = {}
    missing = []
    for key in _SIGNAL_KEYS:
        if key in sec:
            try:
                params[key] = parse_number(sec[key])
            except ValueError as e:
                sys.stderr.write(f"[cwsignal] {key}: {e}\n")
                return None
        elif key in _SIGNAL_DEFAULTS:
            params[key] = _SIGNAL_DEFAULTS[key]
        else:
            missing.append(key)
    if missing:
        sys.stderr.write(f"[cwsignal] section is missing: {', '.join(missing)}\n")
        return None
    return params


# ----------------------------------------------------------------------------
# Lensing: amplification factor F = |F| exp(i arg F) along the microlensing track
# ----------------------------------------------------------------------------
LENS_MODEL_PARAMETERS = {
    "point": ["w_values", "y_values", "results"],
    "sis": ["w_values", "y_values", "results"],
    "nfw": ["w_values", "y_values", "rs_values", "results"],
    "dpbh": ["w_values", "y_values", "f_pbh_values", "m_pbh_values", "results"],
}

# Axes of `results` beyond (w, y), and the [lensing] key selecting a value on each.
_LENS_SELECTION_KEYS = {
    "f_pbh_values": "f_pbh",
    "m_pbh_values": "m_pbh",
    "rs_values": "rs",
}

# The microlensing timescales (Einstein crossing time, impact parameter) are in
# days. Change this if your microlensing module uses a different unit.
_SECONDS_PER_DAY = 86400.0

# w = 8*pi*G*M_Lz*f / c^3, the dimensionless frequency, M_Lz in solar masses.
_W_FACTOR = 8.0 * math.pi * lal.G_SI * lal.MSUN_SI / lal.C_SI ** 3


def _load_hdf5_table(file_path, lens_model):
    """
    Load the HDF5 table for the given lens model and return the parameters as a dictionary.
    """
    if lens_model not in LENS_MODEL_PARAMETERS:
        raise ValueError(f"Unsupported lens model: {lens_model}")

    params = {}
    with h5py.File(file_path, 'r') as f:
        for param in LENS_MODEL_PARAMETERS[lens_model]:
            if param in f:
                params[param] = f[param][:]
            else:
                raise KeyError(f"Parameter '{param}' not found in HDF5 file for lens model '{lens_model}'.")

    return params


class _FixedWSpline:
    """Wrap a 1-D spline so it takes the (w, y, grid=False) call that
    spline_eval_chunked expects, for tables holding a single w value."""

    def __init__(self, spline):
        self._spline = spline

    def __call__(self, w, y, grid=False):
        return self._spline(y)


def build_amplification_splines(lens_data, lens_model, selection):
    """Reduce `results` to a (w, y) grid and spline its real and imaginary parts.

    `results` is indexed [w, y, <model parameters>, 0/1], the last axis being
    real / imaginary. For every model parameter beyond w and y, the value in
    the table nearest to the one requested in `selection` is used.

    Returns (w_values, y_values, real_spline, imag_spline, chosen), where
    `chosen` maps each selection key to the table value actually used.
    """
    w_values = np.asarray(lens_data["w_values"], dtype=np.float64)
    y_values = np.asarray(lens_data["y_values"], dtype=np.float64)
    results = np.asarray(lens_data["results"])

    index = [slice(None), slice(None)]
    chosen = {}
    for param in LENS_MODEL_PARAMETERS[lens_model]:
        if param in ("w_values", "y_values", "results"):
            continue
        grid = np.asarray(lens_data[param], dtype=np.float64)
        key = _LENS_SELECTION_KEYS.get(param, param)
        value = selection.get(key)
        if value is None:
            j = 0
            print(f"  {key}: not given, using {grid[j]:g} (first in the table)")
        else:
            j = int(np.argmin(np.abs(grid - value)))
            print(f"  {key} = {value:g}, using {grid[j]:g} (nearest in the table)")
        chosen[key] = float(grid[j])
        index.append(j)

    real = np.asarray(results[tuple(index) + (0,)], dtype=np.float64)
    imag = np.asarray(results[tuple(index) + (1,)], dtype=np.float64)
    if real.shape != (w_values.size, y_values.size):
        raise ValueError(
            f"results has shape {real.shape} after selecting the model parameters, "
            f"expected (w_values, y_values) = ({w_values.size}, {y_values.size})"
        )

    if w_values.size == 1:
        # Nothing to interpolate over w: spline in y only, at that single w.
        return (w_values, y_values,
                _FixedWSpline(CubicSpline(y_values, real[0])),
                _FixedWSpline(CubicSpline(y_values, imag[0])),
                chosen)

    kx = min(3, w_values.size - 1)
    ky = min(3, y_values.size - 1)
    return (w_values, y_values,
            RectBivariateSpline(w_values, y_values, real, kx=kx, ky=ky),
            RectBivariateSpline(w_values, y_values, imag, kx=kx, ky=ky),
            chosen)


class LensingModel:
    """F(t) along one microlensing track, evaluated segment by segment.

    Time is counted in days from the first sample of the run. The closest
    approach sits at t0, which defaults to the middle of the run. Each sample's
    time comes from its GPS time rather than from a sample count, so the track
    stays aligned with the data even though segments are a whole number of
    samples long while the GPS time advances by nod sidereal days.
    """

    def __init__(self, real_spline, imag_spline, w, TE, t_obs, y0, delta_t,
                 t0=None, y_range=None, chunk_size=500_000):
        self.real_spline = real_spline
        self.imag_spline = imag_spline
        self.w = w
        self.TE = TE
        self.t_obs = t_obs                # length of the run [days]
        self.y0 = y0                      # minimum impact parameter
        self.delta_t = delta_t            # sample spacing [s]
        # Time of closest approach, in days from the first sample of the run.
        self.t0 = 0.5 * t_obs if t0 is None else t0
        self.y_range = y_range
        self.chunk_size = chunk_size

    def amplification(self, offset, count):
        """Complex F for `count` samples starting `offset` seconds into the run."""
        elapsed = offset + np.arange(count, dtype=np.float64) * self.delta_t
        t = elapsed / _SECONDS_PER_DAY    # days from the start of the run
        y = microlensing.MicrolensingImpactparameter(t, self.y0, self.t0, self.TE)

        if self.y_range is not None:
            lo, hi = self.y_range
            if y.min() < lo or y.max() > hi:
                sys.stderr.write(
                    f"Warning: impact parameter y in [{y.min():.4g}, {y.max():.4g}] "
                    f"leaves the table range [{lo:.4g}, {hi:.4g}]; the spline is "
                    f"clamped to the edge value there, so F stops varying.\n"
                )

        return spline_eval_chunked(
            self.real_spline, self.imag_spline, self.w, y, self.chunk_size
        )


def prepare_lensing(config, cw_params, delta_t, run_seconds):
    """Read [lensing], load the lookup table and return a LensingModel, or None.

    delta_t is the sample spacing [s] and run_seconds the length of the whole
    run, from the first sample of the first segment to the last sample of the
    last one. The track is centred on the time of closest approach.
    """
    if not config.has_section("lensing"):
        sys.stderr.write("lensing = true but there is no [lensing] section.\n")
        return None
    sec = config["lensing"]

    lens_model = sec.get("lens_model", fallback=None)
    table_path = sec.get("lens_lookup_table", fallback=None)
    if lens_model is None or table_path is None:
        sys.stderr.write("[lensing] needs lens_model and lens_lookup_table.\n")
        return None
    if lens_model not in LENS_MODEL_PARAMETERS:
        sys.stderr.write(
            f"[lensing] unsupported lens_model {lens_model!r}; known models: "
            f"{', '.join(sorted(LENS_MODEL_PARAMETERS))}\n"
        )
        return None

    missing = [
        key for key in ("distance_to_lens", "distance_to_source", "velocity")
        if key not in sec
    ]
    if missing:
        sys.stderr.write(f"[lensing] section is missing: {', '.join(missing)}\n")
        return None

    Dl = sec.getfloat("distance_to_lens")
    Ds = sec.getfloat("distance_to_source")
    velocity = sec.getfloat("velocity")
    # y0 is the dimensionless minimum impact parameter, not a time.
    # "closest_approach" is still accepted for older config files.
    y0 = sec.getfloat("y0", fallback=None)
    if y0 is None:
        y0 = sec.getfloat("closest_approach", fallback=0.0)
    # Optional: when the closest approach happens, in days from the start of
    # the run. Defaults to the middle of the run.
    t0 = sec.getfloat("time_of_closest_approach", fallback=None)
    selection = {
        key: sec.getfloat(key)
        for key in ("f_pbh", "m_pbh", "rs") if key in sec
    }

    print(f"Lensing enabled: model {lens_model}, table {table_path}")
    try:
        lens_data = _load_hdf5_table(table_path, lens_model)
        w_values, y_values, real_spline, imag_spline, chosen = (
            build_amplification_splines(lens_data, lens_model, selection)
        )
    except (OSError, ValueError, KeyError) as e:
        sys.stderr.write(f"Error loading lens data: {e}\n")
        return None

    # Redshifted lens mass: from [lensing], else the m_pbh taken from the table.
    lens_mass = sec.getfloat("lens_mass", fallback=None)
    if lens_mass is None:
        lens_mass = chosen.get("m_pbh")
    if lens_mass is None:
        sys.stderr.write("[lensing] needs lens_mass (redshifted lens mass, M_sun).\n")
        return None

    # The band is narrow, so a single w for the whole signal is enough.
    w = sec.getfloat("dimensionless_frequency", fallback=None)
    if w is None:
        w = _W_FACTOR * lens_mass * cw_params["frequency"]
    if not w_values[0] <= w <= w_values[-1]:
        sys.stderr.write(
            f"Warning: w = {w:.6g} is outside the table range "
            f"[{w_values[0]:.6g}, {w_values[-1]:.6g}]; the spline is clamped to the "
            f"edge value there.\n"
        )

    t_obs = run_seconds / _SECONDS_PER_DAY
    TE = microlensing.EinsteinCrossingTime(Dl, Ds, velocity, lens_mass)
    print(f"  lens mass = {lens_mass:g} M_sun, w = {w:.6g}, "
          f"minimum impact parameter y0 = {y0:g}")
    print(f"  Einstein crossing time = {TE:.6g} days, "
          f"observation span = {t_obs:.6g} days")
    if not 1e-3 < t_obs / TE < 1e3:
        sys.stderr.write(
            "Warning: the observation span and the Einstein crossing time are "
            "wildly different; check that EinsteinCrossingTime returns days.\n"
        )

    y_lo, y_hi = float(y_values[0]), float(y_values[-1])
    if y0 > y_hi:
        sys.stderr.write(
            f"Error: y0 = {y0:g} is beyond the table's y range [{y_lo:.4g}, "
            f"{y_hi:.4g}], so y never enters the table and F is the same in every "
            f"segment.\n"
            f"  y0 is the dimensionless minimum impact parameter (0 means the "
            f"source passes straight behind the lens), not a time.\n"
            f"  To move the peak in time, use time_of_closest_approach in days "
            f"from the start of the run.\n"
        )
        return None

    model = LensingModel(
        real_spline, imag_spline, w, TE, t_obs, y0, delta_t, t0=t0,
        y_range=(y_lo, y_hi),
    )
    print(f"  closest approach at {model.t0:.6g} days into the run "
          f"(run is {t_obs:.6g} days long)")

    # Probe the whole track before generating anything: a y that never moves, or
    # that sits outside the table, gives the same F in every segment.
    t_probe = np.linspace(0.0, run_seconds, 9) / _SECONDS_PER_DAY
    y_probe = np.asarray(
        microlensing.MicrolensingImpactparameter(t_probe, y0, model.t0, TE),
        dtype=np.float64,
    )
    print(f"  impact parameter over the run: y from {y_probe.min():.4g} "
          f"to {y_probe.max():.4g} (table covers {y_lo:.4g} to {y_hi:.4g})")

    if y_probe.size < 2 or np.ptp(y_probe) <= 1e-6 * max(abs(y_probe).max(), 1e-300):
        sys.stderr.write(
            "Error: the impact parameter does not change over the run, so F would "
            "be identical in every segment. Check that\n"
            "  - microlensing.MicrolensingImpactparameter takes its arguments in the "
            "order (t, y0, t0, TE) as called here, and\n"
            "  - EinsteinCrossingTime returns days, matching _SECONDS_PER_DAY.\n"
        )
        return None

    if y_probe.min() > y_hi or y_probe.max() < y_lo:
        sys.stderr.write(
            f"Error: the whole track lies outside the table's y range "
            f"[{y_lo:.4g}, {y_hi:.4g}], so F would be pinned to one edge value "
            f"in every segment.\n"
            f"  The run spans {t_obs:.4g} days against an Einstein crossing time of "
            f"{TE:.4g} days, giving y up to about {t_obs / (2.0 * TE):.4g}.\n"
            f"  For y to stay in the table, the crossing time needs to be at least "
            f"about {t_obs / (2.0 * y_hi):.4g} days.\n"
        )
        return None

    outside = (y_probe < y_lo) | (y_probe > y_hi)
    if outside.any():
        sys.stderr.write(
            f"Warning: part of the track lies outside the table's y range "
            f"[{y_lo:.4g}, {y_hi:.4g}]; F is clamped to the edge value there.\n"
        )

    return model


def generate_signal(detector, cw_params, ref_gps_time, einstein_factor,
                    amplification=None):
    """Generate the CW signal for one segment/detector with the original
    sigen.TimeSeries and return it (float32) in the same units as the noise.

    [cwsignal] gives the absolute GW frequency [Hz] and spin-down [Hz/s]. They
    are converted to what the original class expects:
      * it has no heterodyne, and the data are stored relative to fpo (the
        band start), so the frequency is taken relative to fpo;
      * it uses cos(2*phase), so it wants half the GW frequency and spin-down;
      * it has no h0, so the amplitude h0 / einstein_factor is applied here.

    If `amplification` is given (complex F, one value per sample), it is
    passed on to sigen, which multiplies the signal by |F| and shifts its
    phase by arg(F).
    """
    # GW frequency at this segment's start: spin-down evolves it from the
    # reference time (the first segment), so all segments stay consistent.
    f_gw = (
        cw_params["frequency"]
        + cw_params["spin_down"] * (detector.start_time - ref_gps_time)
    )

    fpo = detector.starting_frequency
    bandwidth = detector.bandwidth
    if not fpo <= f_gw <= fpo + bandwidth:
        sys.stderr.write(
            f"Warning: signal frequency {f_gw:.6f} Hz is outside "
            f"band {detector.band} [{fpo:.6f}, {fpo + bandwidth:.6f}] Hz.\n"
        )
        best, candidates = suggest_bands(f_gw, bandwidth, detector.overlap)
        if best is None:
            sys.stderr.write(
                f"  No band contains {f_gw:.6f} Hz (bands start at 10 Hz).\n"
            )
        else:
            best_fpo = get_fpo(best, bandwidth, detector.overlap)
            msg = (
                f"  Suggestion: set band = {best} in [tdfstat] "
                f"(covers [{best_fpo:.6f}, {best_fpo + bandwidth:.6f}] Hz)"
            )
            others = [str(b) for b in candidates if b != best]
            if others:
                msg += f"; band {', '.join(others)} also contains it"
            sys.stderr.write(msg + "\n")
        sys.stderr.write(
            f"  Or keep band = {detector.band} and set frequency in [cwsignal] "
            f"between {fpo:.6f} and {fpo + bandwidth:.6f} Hz.\n"
        )

    params = {
        key: cw_params[key]
        for key in ("ra", "dec", "wobble_angle", "inclination",
                    "polarization", "reference_phase")
    }
    params["frequency"] = 0.5 * (f_gw - fpo)
    params["spin_down"] = 0.5 * cw_params["spin_down"]

    if amplification is not None:
        params["amplification"] = amplification

    time_series = TimeSeries(detector)
    signal, _ = time_series.generate_time_domain_signal(**params)

    lensed = ""
    if amplification is not None:
        if time_series.amplification is None:
            raise RuntimeError(
                "sigen.TimeSeries ignored the amplification factor: this needs "
                "the version of sigen.py that accepts 'amplification'."
            )
        absF = np.abs(amplification)
        lensed = f", lensed: |F| in [{absF.min():.4f}, {absF.max():.4f}]"

    # The noise is in units of strain / einstein_factor, so the signal must be too.
    signal = signal * (cw_params["h0"] / einstein_factor)

    print(
        f"Generated signal for {detector.name} segment {detector.segment} "
        f"(f = {f_gw:.6f} Hz{lensed})"
    )
    return signal.astype(np.float32)


# ----------------------------------------------------------------------------
# psd-dict parsing  (get_psd_for_detector)
# ----------------------------------------------------------------------------
def get_psd_for_detector(psd_dict, detector):
    """Pull the PSD path for `detector` from a "{H1:/path, L1:/path}" string in the ini file
    same ini file will be used 
    """
    if not psd_dict or not detector:
        return None
    for token in psd_dict.replace("{", "").replace("}", "").split(","):
        token = token.strip()
        if ":" not in token:
            continue
        det, _, path = token.partition(":")
        if det.strip() == detector:
            return path.strip()
    return None


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main(argv=None):

    parser = argparse.ArgumentParser(description="Generate segments for TDFStat")
    parser.add_argument("config", help="Path to the parameters .ini file containing [tdfstat] section")
    args = parser.parse_args(argv)

    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # inline_comment_prefixes lets values carry comments: "ra = 1.2  # rad"
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not config.read(config_path):
        sys.stderr.write(f"Cannot load configuration file: {config_path}\n")
        return 1

    print(f"Loaded config file: {config_path}")

    tdfstat_config = config["tdfstat"]
    starting_gpstime = tdfstat_config.getfloat("starting_gpstime", 1000000000.0)
    nseg = tdfstat_config.getint("nseg", 1)
    nod = tdfstat_config.getint("nod", 1)
    bandwidth = tdfstat_config.getfloat("bandwidth", 1.0)
    ov = tdfstat_config.getfloat("ov", 5.0)
    detector_list = tdfstat_config.get("detectors", fallback=None)
    psd_dict = tdfstat_config.get("psd-dict", fallback=None)
    bandnumber = tdfstat_config.getint("band", 1234)
    output_dir = tdfstat_config.get("output_dir", fallback="./output")
    earth_ephem = tdfstat_config.get("earth-ephem", fallback=None)
    sun_ephem   = tdfstat_config.get("sun-ephem",   fallback=None)
    write_signal = tdfstat_config.getboolean("write_signal", fallback=False)
    lensing = tdfstat_config.getboolean("lensing", fallback=False)
    maskfile = tdfstat_config.get("maskfile", fallback=None)

    for label, path in (("earth", earth_ephem), ("sun", sun_ephem)):
        if path is None:
            sys.stderr.write(f"Error: no {label}-ephem given in config file.\n")
            return 1
        if not os.path.isfile(path):
            sys.stderr.write(
                f"WARNING: {label} ephemeris file not found: {path}\n"
            )


    #The below code is commented out because the amp and sigma parameters are not used in the current implementation of datagen.py,
    # as the Gaussian data is generated with a fixed amplitude of 1.0 and a variance derived from the interpolated PSD value. 
    # If you want to use custom values for amp and sigma, you can uncomment the code and adjust the generate_gaussian_data call accordingly.
    
    #if tdfstat_config.has_option("amp"):
    #     print("Warning: 'amp' option is ignored in this implementation, using default value 1.0")
    #    _amp = float(tdfstat_config.get("amp", 1.0)) if tdfstat_config.has_option("amp") else 1.0       # parity only
    #    _sigma = float(tdfstat_config.get("sigma", 1.0)) if tdfstat_config.has_option("sigma") else 1.0   # parity only
    #else {}


    if psd_dict is None:
        sys.stderr.write("No psd-dict found in config file.\n")
        return 1
    if detector_list is None:
        sys.stderr.write("No detectors specified in config file.\n")
        return 1

    cw_params = None
    if write_signal:
        cw_params = read_signal_params(config)
        if cw_params is None:
            return 1
        print("Signal generation enabled with parameters:")
        for key, value in cw_params.items():
            print(f"  {key} = {value}")

    detectors = [d.strip() for d in detector_list.split(",") if d.strip()]

    dt = 1.0 / (2.0 * bandwidth)
    N = int(SIDEREAL_DAY * nod / dt)          # samples in one segment

    # The whole run is one microlensing track, centred on the closest approach.
    # Its length is measured on the GPS timeline: segments start nod sidereal
    # days apart and the last one is N samples long.
    lens = None
    if lensing:
        if not write_signal:
            sys.stderr.write(
                "lensing = true needs write_signal = true: the amplification "
                "factor is applied to the CW signal.\n"
            )
            return 1
        run_seconds = (nseg - 1) * nod * SIDEREAL_DAY + N * dt
        lens = prepare_lensing(config, cw_params, dt, run_seconds)
        if lens is None:
            return 1
    
    #The scaling used in CW analysis
    #found by Andrzej Królak in the C code, shared this information by Pawel Ciecialag
    einstein_factor = 1e-20

    print(f"Using the Einstein Factor: {einstein_factor:.6e}")

    gps_time = starting_gpstime
    for seg in range(1, nseg + 1):
        segment_name = f"{seg:03d}"

        # One track for the whole run, so F(t) is the same for every detector.
        lens_amp = None
        if lens is not None:
            lens_amp = lens.amplification(gps_time - starting_gpstime, N)

        for det in detectors:
            fname = get_psd_for_detector(psd_dict, det)
            if not fname:
                sys.stderr.write(
                    f"Error: No PSD found for detector {det} in psd-dict.\n"
                )
                continue
            print(f"Using PSD file: {fname}")
            print(f"Using the Einstein Factor: {einstein_factor:.6e}")
            print(f"band={bandnumber} ")
            print(f"bandwidth={bandwidth:f} ")
            print(f"ov={ov:f} ")

            fpo = get_fpo(bandnumber, bandwidth, ov)
            freq = fpo + bandwidth / 2.0
            psd_val = interpolate_psd(fname, freq)
            asd = math.sqrt(psd_val)
            asd_ef = asd / einstein_factor
            variance = asd_ef * math.sqrt(bandwidth)

            print(f"Frequency : {freq:f} ")
            print(f"Effective ASD: {asd_ef:.6e} ")
            print(f"Noise sigma: {variance:.6e}")

            output_directory = os.path.join(output_dir, segment_name, det)
            print(
                f"Generating segment {segment_name} for detector {det} "
                f"at GPS time {gps_time:.0f}"
            )

            ephem = generate_ephemeris_segment(gps_time, nod, det, output_directory, N, bandwidth, earth_ephem, sun_ephem)
            if ephem is None:
                sys.stderr.write(f"Error generating ephemeris for {segment_name}/{det}\n")
                continue
            DetSSB, phir = ephem

            # Signal first (if requested), so it can go into the noise file.
            signal = None
            if write_signal:
                detector = build_detector(
                    det, segment_name, bandnumber, ov, gps_time, N,
                    bandwidth, fpo, DetSSB, phir,
                )
                signal = generate_signal(
                    detector, cw_params, starting_gpstime, einstein_factor,
                    amplification=lens_amp,
                )

            # e.g. ./output/001/H1/xdat_001_1234.bin
            xdat_file = os.path.join(
                output_directory, f"xdat_{segment_name}_{bandnumber:04d}.bin"
            )
            if generate_gaussian_data(N, 1.0, variance, xdat_file, maskfile, signal) != 0:
                sys.stderr.write(
                    f"Error generating Gaussian data for {segment_name}/{det}\n"
                )

        gps_time += nod * SIDEREAL_DAY

    print("All segments processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())