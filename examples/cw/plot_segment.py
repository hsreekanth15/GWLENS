#!/usr/bin/env python3
"""
plot_segment.py - one segment, one plot, three separate curves:

    * the Gaussian noise ALONE   (the xdat file with its injected signal removed)
    * the unlensed signal ALONE
    * the lensed signal ALONE

Nothing is plotted twice: the noise curve contains no signal, and each signal
curve contains no noise. The signals are rebuilt from the config file and the
DetSSB.bin written next to the data, so this works whether or not the two runs
used the same random seed.

The three curves have very different sizes (the signal sits far below the noise,
and the lensed signal is |F| times the unlensed one), so each curve has its own
scale factor that you set yourself. Every factor applied is shown in the legend,
so the plot stays honest. The unlensed curve is drawn last, on top of the others.

    --noise-scale      multiplies the noise curve      (default 1)
    --signal-scale     multiplies BOTH signal curves   (default 1)
    --lensed-scale     extra factor, lensed curve only (default 1)
    --unlensed-scale   extra factor, unlensed curve only (default 1)

Keep --signal-scale for making the signals visible next to the noise and leave
the two per-curve factors at 1: scaling the lensed and unlensed curves by
different amounts destroys the |F| ratio you are trying to see.

Usage:
    # true amplitudes, first 300 samples
    python plot_segment.py parameters.ini --segment 3 \
        --lensed cwlens --unlensed cwlens_ul

    # signals blown up 1000x to sit next to the noise
    python plot_segment.py parameters.ini --segment 3 \
        --lensed cwlens --unlensed cwlens_ul --signal-scale 1000

    # three stacked panels, each on its own y axis: no scaling needed
    python plot_segment.py parameters.ini --segment 3 \
        --lensed cwlens --unlensed cwlens_ul --layout stacked
"""

import argparse
import configparser
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gwlens.tools import plotting
plotting.apply_figure_style()

try:
    from gwlens.tdfstat import datagen as dg
except ImportError:                       # run from inside gwlens/tdfstat
    import datagen as dg

EINSTEIN_FACTOR = 1e-20                   # same scaling datagen uses

NOISE_COLOR = "0.65"
UNLENSED_COLOR = "tab:blue"
LENSED_COLOR = "tab:orange"


def quiet(func, *args, **kwargs):
    """Call func without its progress printing."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def read_ephemeris(path, N):
    """DetSSB.bin holds 3N values in units of dt, then phir and epsilon."""
    arr = np.fromfile(path, dtype=np.float64)
    if arr.size < 3 * N + 2:
        raise ValueError(f"{path}: {arr.size} values, expected {3 * N + 2}")
    return arr[: 3 * N], float(arr[3 * N])


def label_with_scale(name, scale):
    return name if scale == 1.0 else f"{name}  (x {scale:g})"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="the .ini used to generate the data")
    parser.add_argument("--lensed", required=True, help="output_dir of the lensed run")
    parser.add_argument("--unlensed", required=True, help="output_dir of the unlensed run")
    parser.add_argument("--segment", type=int, default=1, help="segment number (default 1)")
    parser.add_argument("--detector", default=None,
                        help="detector (default: first in the config)")

    # what part of the segment to show
    parser.add_argument("--samples", type=int, default=300,
                        help="samples to plot, 0 for the whole segment (default 300)")
    parser.add_argument("--start", type=int, default=0,
                        help="first sample to plot (default 0)")

    # scaling, all under your control
    parser.add_argument("--noise-scale", type=float, default=1.0,
                        help="multiply the noise curve by this (default 1)")
    parser.add_argument("--signal-scale", type=float, default=1.0,
                        help="multiply BOTH signal curves by this (default 1)")
    parser.add_argument("--lensed-scale", type=float, default=1.0,
                        help="extra factor for the lensed curve only (default 1)")
    parser.add_argument("--unlensed-scale", type=float, default=1.0,
                        help="extra factor for the unlensed curve only (default 1)")

    parser.add_argument("--noise-from", choices=("lensed", "unlensed", "none"),
                        default="lensed",
                        help="which run's xdat the noise is recovered from, by "
                             "subtracting that run's signal (default lensed)")
    parser.add_argument("--layout", choices=("single", "stacked"), default="single",
                        help="one axes for all three curves (default), or three "
                             "stacked panels each with its own y axis")
    parser.add_argument("--out", default=None,
                        help="figure file (default segment_<NNN>_<det>.png)")
    args = parser.parse_args(argv)

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    if not config.read(args.config):
        sys.stderr.write(f"Cannot read {args.config}\n")
        return 1

    sec = config["tdfstat"]
    starting_gpstime = sec.getfloat("starting_gpstime", 1000000000.0)
    nseg = sec.getint("nseg", 1)
    nod = sec.getint("nod", 1)
    bandwidth = sec.getfloat("bandwidth", 1.0)
    ov = sec.getfloat("ov", 5.0)
    band = sec.getint("band", 1234)
    detectors = [d.strip() for d in sec.get("detectors", "H1").split(",") if d.strip()]
    detector_name = args.detector or detectors[0]

    if not 1 <= args.segment <= nseg:
        sys.stderr.write(f"--segment must be between 1 and {nseg}\n")
        return 1

    dt = 1.0 / (2.0 * bandwidth)
    N = int(dg.SIDEREAL_DAY * nod / dt)
    fpo = dg.get_fpo(band, bandwidth, ov)

    cw_params = dg.read_signal_params(config)
    if cw_params is None:
        return 1

    segment_name = f"{args.segment:03d}"
    gps_time = starting_gpstime + (args.segment - 1) * nod * dg.SIDEREAL_DAY

    # ---- rebuild both signals for this segment ------------------------------
    ephem_path = os.path.join(args.lensed, segment_name, detector_name, "DetSSB.bin")
    if not os.path.isfile(ephem_path):
        ephem_path = os.path.join(args.unlensed, segment_name, detector_name, "DetSSB.bin")
    DetSSB, phir = read_ephemeris(ephem_path, N)

    def make_detector():
        return dg.build_detector(detector_name, segment_name, band, ov, gps_time,
                                 N, bandwidth, fpo, DetSSB, phir)

    unlensed_signal = quiet(dg.generate_signal, make_detector(), cw_params,
                            starting_gpstime, EINSTEIN_FACTOR).astype(np.float64)

    lensed_signal, F = None, None
    if config.has_section("lensing"):
        run_seconds = (nseg - 1) * nod * dg.SIDEREAL_DAY + N * dt
        lens = quiet(dg.prepare_lensing, config, cw_params, dt, run_seconds)
        if lens is not None:
            F = lens.amplification(gps_time - starting_gpstime, N)
            lensed_signal = quiet(dg.generate_signal, make_detector(), cw_params,
                                  starting_gpstime, EINSTEIN_FACTOR,
                                  amplification=F).astype(np.float64)
    if lensed_signal is None:
        sys.stderr.write("No usable [lensing] section: plotting the unlensed signal only.\n")

    # ---- noise alone: the data with its own injected signal removed ---------
    def load_xdat(root):
        path = os.path.join(root, segment_name, detector_name,
                            f"xdat_{segment_name}_{band:04d}.bin")
        x = np.fromfile(path, dtype=np.float32).astype(np.float64)
        if x.size != N:
            raise ValueError(f"{path}: {x.size} samples, expected {N}")
        return x

    noise = None
    if args.noise_from == "lensed":
        if lensed_signal is None:
            sys.stderr.write(
                "No lensed signal to subtract; taking the noise from the unlensed run.\n")
            noise = load_xdat(args.unlensed) - unlensed_signal
        else:
            noise = load_xdat(args.lensed) - lensed_signal
    elif args.noise_from == "unlensed":
        noise = load_xdat(args.unlensed) - unlensed_signal

    # ---- slice --------------------------------------------------------------
    first = max(0, min(args.start, N - 1))
    count = N - first if args.samples <= 0 else min(args.samples, N - first)
    sl = slice(first, first + count)
    t = (first + np.arange(count)) * dt

    s_unlensed = args.signal_scale * args.unlensed_scale
    s_lensed = args.signal_scale * args.lensed_scale

    curves = []                            # (array, scale, label, color, zorder)
    if noise is not None:
        curves.append((noise, args.noise_scale,
                       label_with_scale("Gaussian noise", args.noise_scale),
                       NOISE_COLOR, 1))
    if lensed_signal is not None:
        curves.append((lensed_signal, s_lensed,
                       label_with_scale("lensed signal", s_lensed),
                       LENSED_COLOR, 2))
    # unlensed last, so it is drawn on top of the lensed curve
    curves.append((unlensed_signal, s_unlensed,
                   label_with_scale("unlensed signal", s_unlensed),
                   UNLENSED_COLOR, 3))

    title = (f"Segment {segment_name}, {detector_name}, band {band} "
            )
    if F is not None:
        title += (f"\n$|F|$ = {np.abs(F[sl]).mean():.4f}, "
                  f"arg F = {np.degrees(np.angle(F[sl].mean())):.2f} deg")

    # ---- plot ---------------------------------------------------------------
    if args.layout == "stacked":
        fig, axes = plt.subplots(len(curves), 1, figsize=(11, 2.6 * len(curves)),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (y, scale, label, color, _z) in zip(axes, curves):
            ax.plot(t, scale * y[sl], color=color, lw=1.0)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.3)
        axes[0].set_title(title, fontsize=10)
        axes[-1].set_xlabel("time into the segment [s]")
        fig.supylabel("strain / einstein_factor", fontsize=9)
    else:
        fig, ax = plt.subplots(figsize=(15, 5))
        for y, scale, label, color, z in curves:
            lw = 1.6 if color is UNLENSED_COLOR else (0.8 if color is NOISE_COLOR else 1.4)
            ax.plot(t, scale * y[sl], color=color, lw=lw, label=label, zorder=z)
        ax.set_xlabel("Time into the segment [s]")
        ax.set_ylabel("strain /$10^{-20}$")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out = args.out or f"segment_{segment_name}_{detector_name}.png"
    fig.savefig(out, dpi=130)

    # ---- report -------------------------------------------------------------
    print(f"Segment {segment_name}, {detector_name}: "
          f"samples {first} to {first + count - 1} ({count} of {N})")
    for y, scale, label, _c, _z in curves:
        rms = np.std(y[sl])
        print(f"  {label.split('  (')[0]:<16} rms {rms:.4e}"
              + (f"   drawn at rms {scale * rms:.4e}" if scale != 1.0 else ""))
    if F is not None:
        print(f"  $|F|$ in this window  [{np.abs(F[sl]).min():.4f}, "
              f"{np.abs(F[sl]).max():.4f}]")
    if noise is not None:
        sig = np.std(unlensed_signal[sl])
        if sig > 0:
            print(f"  --signal-scale {float(f'{0.6 * np.std(noise[sl]) / sig:.0e}'):g} "
                  f"would make the signals about 60% of the noise amplitude")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
