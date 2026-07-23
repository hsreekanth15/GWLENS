#!/usr/bin/env python3
"""
datagen.py - generate ephemeris data for multiple segments and detectors

Python port of datagen.c + get_barycenter.c (Sreekanth Harikumar).

Pipeline:
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

import os
import sys
import math
import errno
import configparser
import argparse
import numpy as np
from scipy.interpolate import CubicSpline

import lal
import lalpulsar

# ----------------------------------------------------------------------------
# Constants (mirror the #defines in datagen.c / EphemerisDetector.h)
# ----------------------------------------------------------------------------
EFILE = "earth00-40-DE405.dat"
SFILE = "sun00-40-DE405.dat"
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

# Detector enum names, in the same order as names[] in get_barycenter.c.
_NAMES = [
    "none", "GB", "NA", "AO", "HO", "PR", "TD", "PK", "JB", "G3", "G1RAD",
    "G8", "VL", "BO", "MO", "NC", "EF", "FB", "H1", "H2", "LHO", "LLO",
    "L1", "GEO", "G1", "V1", "VIRGO", "TAMA", "T1",
]


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
def generate_gaussian_data(N, amp, sigma, dataout, maskfile=None):
    """Write N float32 samples amp*N(0, sigma) to dataout; zero where |mask|<1e-30.
    Returns 0 on success, 1 on failure."""
    if N <= 0:
        sys.stderr.write(f"Invalid N: {N}\n")
        return 1

    seed = int.from_bytes(os.urandom(4), "little")   # like /dev/urandom seeding
    rng = np.random.default_rng(seed)
    x = (amp * rng.normal(0.0, sigma, size=N)).astype(np.float32)

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

    print(f"Data written to {dataout} ({N} samples)")
    return 0


# ----------------------------------------------------------------------------
# Reference frequency  (get_fpo)
# ----------------------------------------------------------------------------
def get_fpo(bandnumber, bandwidth, ov):
    """fpo = 10 + (1 - 2^-ov) * bandnumber * bandwidth."""
    return 10.0 + (1.0 - 2.0 ** (-ov)) * bandnumber * bandwidth


# ----------------------------------------------------------------------------
# Barycentring  (get_barycenter) -- faithful to get_barycenter.c
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


def generate_ephemeris_segment(gps_start, ndays, detector_name, output_dir, N, bandwidth):
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

    try:
        edat = lalpulsar.InitBarycenter(EFILE, SFILE)
    except Exception as e:
        sys.stderr.write(
            f"InitBarycenter failed for ({EFILE}, {SFILE}): {e}\n"
            f"  -> put the DE405 files in the cwd or set EFILE/SFILE to full paths.\n"
        )
        return

    print(f"  barycentring {N} samples for {detector_name}...", flush=True)
    DetSSB, rDet = get_barycenter(gps_start, detector, edat, N, dt)
    print(f"  barycentring done for {detector_name}.", flush=True)

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


# ----------------------------------------------------------------------------
# psd-dict parsing  (get_psd_for_detector)
# ----------------------------------------------------------------------------
def get_psd_for_detector(psd_dict, detector):
    """Pull the PSD path for `detector` from a "{H1:/path, L1:/path}" string."""
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
def main(argv):
    # Make output appear immediately even when stdout is a pipe/redirect.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    config_file = argv[1] if len(argv) >= 2 else "config.ini"
    print(f"datagen.py starting (cwd={os.getcwd()})", flush=True)

    if not os.path.isfile(config_file):
        print(
            f"ERROR: config file '{config_file}' not found.\n"
            f"  Pass it explicitly:  python datagen.py /path/to/config.ini",
            flush=True,
        )
        return 1

    ini = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    ini.optionxform = str
    if not ini.read(config_file):
        print(f"ERROR: could not parse configuration file: {config_file}", flush=True)
        return 1

    print(f"Loaded config file: {config_file}", flush=True)

    if "general" not in ini:
        print(
            f"ERROR: no [general] section in {config_file}. "
            f"Sections found: {ini.sections()}",
            flush=True,
        )
        return 1
    g = ini["general"]
    starting_gpstime = g.getfloat("starting_gpstime", 1000000000.0)
    nseg = g.getint("nseg", 1)
    nod = g.getint("nod", 1)
    bandwidth = g.getfloat("bandwidth", 1.0)
    ov = g.getfloat("ov", 5.0)
    detector_list = g.get("detectors", fallback=None)
    psd_dict = g.get("psd-dict", fallback=None)
    bandnumber = g.getint("band", 1234)
    output_dir = g.get("output_dir", fallback="./output")

    gauss = ini["gaussian"] if ini.has_section("gaussian") else {}
    _amp = float(gauss.get("amp", 1.0)) if gauss else 1.0       # parity only
    _sigma = float(gauss.get("sigma", 1.0)) if gauss else 1.0   # parity only

    if psd_dict is None:
        sys.stderr.write("No psd-dict found in config file.\n")
        return 1
    if detector_list is None:
        sys.stderr.write("No detectors specified in config file.\n")
        return 1

    detectors = [d.strip() for d in detector_list.split(",") if d.strip()]

    dt = 1.0 / (2.0 * bandwidth)
    N = int(SIDEREAL_DAY * nod / dt)

    print(
        f"Parsed: detectors={detectors}, nseg={nseg}, nod={nod}, "
        f"bandwidth={bandwidth}, band={bandnumber}, output_dir={output_dir}",
        flush=True,
    )
    print(
        f"Workload: N={N} samples/segment, dt={dt}, "
        f"{3 * N} barycentre evaluations per detector",
        flush=True,
    )
    if not detectors:
        print("ERROR: 'detectors' is empty after parsing.", flush=True)
        return 1
    if not (os.path.isfile(EFILE) and os.path.isfile(SFILE)):
        print(
            f"WARNING: ephemeris files not found in cwd "
            f"({EFILE!r}, {SFILE!r}). InitBarycenter will fail until these "
            f"are present or EFILE/SFILE are set to full paths.",
            flush=True,
        )

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

            freq = get_fpo(bandnumber, bandwidth, ov) + bandwidth / 2.0
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

            generate_ephemeris_segment(
                gps_time, nod, det, output_directory, N, bandwidth
            )

            # NOTE: the C hardcodes amp=1.0, sigma=1.15e-3 here; the computed
            # `variance` above is printed but not used. Preserved verbatim.
            xdat_file = os.path.join(
                output_directory, f"xdat_{segment_name}_{bandnumber:04d}.bin"
            )
            if generate_gaussian_data(N, 1.0, 1.15e-3, xdat_file, None) != 0:
                sys.stderr.write(
                    f"Error generating Gaussian data for {segment_name}/{det}\n"
                )

        gps_time += nod * SIDEREAL_DAY

    print("All segments processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))