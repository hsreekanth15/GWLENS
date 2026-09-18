#!/usr/bin/env python3
"""
datagen.py - generate ephemeris data for multiple segments and detectors

Python port of datagen.c + get_barycenter.c (Sreekanth Harikumar).
Adapted from the original C code 

Tasks:
  - reads a config .ini (sections [general] and [gaussian])
  - for each segment / detector:
      * picks the PSD file from psd-dict, interpolates it at the band frequency
      * computes the detector ephemeris (DetSSB.bin, rDet.bin, rSSB.bin)
      * writes a Gaussian xdat_<seg>_<band>.bin time series
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
Run:       python datagen.py config.ini
"""

import argparse
import os
import sys
import math
import errno
import configparser
from types import SimpleNamespace
import numpy as np
from scipy.interpolate import CubicSpline

import lal
import lalpulsar


try:                        # datagen imported as part of a package
    from .sigen import TimeSeries
except ImportError:         # run directly: python datagen.py config.ini
    from sigen import TimeSeries
 
# ----------------------------------------------------------------------------
# Constants (mirror the #defines in datagen.c / EphemerisDetector.h)
# ----------------------------------------------------------------------------
#EFILE = "earth00-40-DE405.dat"
#SFILE = "sun00-40-DE405.dat"
EPSILON = 0.40909280422232891          # obliquity of the ecliptic [rad]
SIDEREAL_DAY = 86164.09890369719       # seconds
C_SI = lal.C_SI                        # 299792458.0 m/s
_DEG = 180.0 / math.pi                 # the C 'deg' factor (radians per ... see usage)

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
    """Build DetSSB/rDet/rSSB for one segment+detector and write the .bin files."""
    dt = 1.0 / (2.0 * bandwidth)

    detector = get_detector(detector_name)

    if mkdir_p(output_dir) != 0:
        sys.stderr.write(f"Failed to create directory {output_dir}\n")
        return

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

    return DetSSB,phir



# ----------------------------------------------------------------------------
# Signal injection (bridge to sigen.TimeSeries)
# ----------------------------------------------------------------------------
def build_detector(det_name, segment, band, ov, gps_start, N, bandwidth, fpo,
                   DetSSB, phir, strain=None):
    """Wrap one segment's data in the detector object sigen.TimeSeries expects.
 
    Units: time in seconds since gps_start; DetSSB converted back from units of
    dt (as stored in DetSSB.bin) to light-seconds and reshaped to (N, 3).
    `strain` is used by reference, so the injection modifies it in place.
    """
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
        longitude_radians=latitude,#check if its latitude or longitude, in the C code it was latitude but in the sigen.TimeSeries it is longitude
        orientation_radians=gamma,
        time_domain_strain=strain,
    )
 
 
def read_signal_params(config):
    """Read the [signal] section into a dict of floats, or None on error."""
    if not config.has_section("cwsignal"):
        sys.stderr.write("write_signal = true but there is no [cwsignal] section.\n")
        return None
    sec = config["cwsignal"]
    params = {}
    missing = []
    for key in _SIGNAL_KEYS:
        if key in sec:
            params[key] = sec.getfloat(key)
        elif key in _SIGNAL_DEFAULTS:
            params[key] = _SIGNAL_DEFAULTS[key]
        else:
            missing.append(key)
    if missing:
        sys.stderr.write(f"[cwsignal] section is missing: {', '.join(missing)}\n")
        return None
    return params
 
 
def inject_signal(detector, cw_params, ref_gps_time, einstein_factor):
    """Add the CW signal to detector.time_domain_strain in place."""
    params = dict(cw_params)
 
    # Frequency at this segment's start: spin-down evolves it from the
    # reference time (the first segment), so all segments stay consistent.
    params["frequency"] = (
        cw_params["frequency"]
        + cw_params["spin_down"] * (detector.start_time - ref_gps_time)
    )
    # The noise is in units of strain / einstein_factor, so the signal must be too.
    params["h0"] = cw_params["h0"] / einstein_factor
 
    fpo = detector.starting_frequency
    if not fpo <= params["frequency"] <= fpo + detector.bandwidth:
        sys.stderr.write(
            f"Warning: signal frequency {params['frequency']:.6f} Hz is outside "
            f"the band [{fpo:.6f}, {fpo + detector.bandwidth:.6f}] Hz.\n"
        )
 
    TimeSeries(detector).generate_time_domain_signal(**params)
    print(
        f"Injected signal into {detector.name} segment {detector.segment} "
        f"(f = {params['frequency']:.6f} Hz)"
    )
 
def generate_signal(detector, cw_params, ref_gps_time, einstein_factor):
    """Generate the CW signal for one segment/detector and return it (float32).
 
    It is scaled to the same units as the Gaussian noise (strain / einstein_factor),
    so it can be added straight to the Gaussian noise.
    """
    params = dict(cw_params)
 
    # Frequency at this segment's start: spin-down evolves it from the
    # reference time (the first segment), so all segments stay consistent.
    params["frequency"] = (
        cw_params["frequency"]
        + cw_params["spin_down"] * (detector.start_time - ref_gps_time)
    )
    # The noise is in units of strain / einstein_factor, so the signal must be too.
    params["h0"] = cw_params["h0"] / einstein_factor
 
    fpo = detector.starting_frequency
    if not fpo <= params["frequency"] <= fpo + detector.bandwidth:
        sys.stderr.write(
            f"Warning: signal frequency {params['frequency']:.6f} Hz is outside "
            f"the band [{fpo:.6f}, {fpo + detector.bandwidth:.6f}] Hz.\n"
        )
 
    signal, _ = TimeSeries(detector).generate_time_domain_signal(**params)
    print(
        f"Generated signal for {detector.name} segment {detector.segment} "
        f"(f = {params['frequency']:.6f} Hz)"
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
    args = parser.parse_args()

    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = configparser.ConfigParser()
    config.read(config_path)

 
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
    write_signal = tdfstat_config.get("write_signal", fallback="false").strip().lower() == 'true'
    

    for label, path in (("earth", earth_ephem), ("sun", sun_ephem)):
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
        print("Signal injection enabled with parameters:")
        for key, value in cw_params.items():
            print(f"  {key} = {value}")

    detectors = [d.strip() for d in detector_list.split(",") if d.strip()]

    dt = 1.0 / (2.0 * bandwidth)
    N = int(SIDEREAL_DAY * nod / dt)
    
    #The scaling used in CW analysis
    #found by Andrzej Królak in the C code, shared this information by Pawel Ciecialag
    einstein_factor = 1e-20

    print(f"Using the Einstein Factor:{einstein_factor:.6e}", end="")

    gps_time = starting_gpstime
    for seg in range(1, nseg + 1):
        segment_name = f"{seg:03d}"
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

            fpo = get_fpo(bandnumber, bandwidth, ov) + bandwidth / 2.0
            freq = fpo + bandwidth / 2.0  # center of the band
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
                sys.stderr.write(
                    f"Error generating ephemeris for {segment_name}/{det}\n"
                )
                continue

            DetSSB, phir = ephem

            signal=None

            if write_signal:
                detector = build_detector(det, segment_name, bandnumber, ov, gps_time, N, bandwidth,fpo , DetSSB,phir,)
                signal = generate_signal(detector, cw_params, starting_gpstime, einstein_factor)
     

            xdat_file = os.path.join(
                output_directory, f"xdat_{segment_name}_{bandnumber:04d}.bin"
            )
            if generate_gaussian_data(N, 1.0, variance, xdat_file, None,signal) != 0:
                sys.stderr.write(
                    f"Error generating Gaussian data for {segment_name}/{det}\n"
                )
                

        gps_time += nod * SIDEREAL_DAY

    print("All segments processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
