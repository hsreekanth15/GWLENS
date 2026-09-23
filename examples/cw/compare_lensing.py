#!/usr/bin/env python3
"""
compare_lensing.py - compare a lensed and an unlensed TDFStat data set.

The injected signal is far below the noise in the time series, so plotting the
two xdat files on top of each other shows nothing but noise (and the noise is
different in the two runs anyway, since each uses its own random seed).

Instead, for every segment this script rebuilds the injected signal from the
config file and the DetSSB.bin written next to the data, then projects both
data sets onto that template and onto a copy shifted by 90 degrees. The two
projections give the amplitude and phase of the signal actually present in the
data:

    unlensed run  ->  amplitude 1, phase 0
    lensed run    ->  amplitude |F(t)|, phase arg F(t)

Those are plotted against the |F| and arg F predicted by the lens model.

Usage:
    python compare_lensing.py parameters.ini --lensed cwlens --unlensed cwlens_ul
    python compare_lensing.py parameters.ini --lensed cwlens --unlensed cwlens_ul \
        --detector L1 --out comparison_L1.png
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

try:
    from gwlens.tdfstat import datagen as dg
except ImportError:                       # run from inside gwlens/tdfstat
    import datagen as dg

EINSTEIN_FACTOR = 1e-20                   # same scaling datagen uses


def read_ephemeris(path, N):
    """DetSSB.bin holds 3N values in units of dt, then phir and epsilon."""
    arr = np.fromfile(path, dtype=np.float64)
    if arr.size < 3 * N + 2:
        raise ValueError(f"{path}: {arr.size} values, expected {3 * N + 2}")
    return arr[: 3 * N], float(arr[3 * N])


def quiet(func, *args, **kwargs):
    """Call func without its progress printing."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def project(x, s, s90):
    """Amplitude and phase of the template s in the data x.

    s90 is s shifted by 90 degrees, so the two projections are the in-phase
    and quadrature parts. Returns (amplitude, phase, amplitude uncertainty).
    """
    ss = np.dot(s, s)
    qq = np.dot(s90, s90)
    c = np.dot(x, s) / ss
    q = np.dot(x, s90) / qq
    sigma = np.std(x)                     # noise dominates, so this is the noise level
    return np.hypot(c, q), np.arctan2(q, c), sigma / np.sqrt(ss)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="the .ini used to generate the data")
    parser.add_argument("--lensed", required=True, help="output_dir of the lensed run")
    parser.add_argument("--unlensed", required=True, help="output_dir of the unlensed run")
    parser.add_argument("--detector", default=None, help="detector to use (default: the first)")
    parser.add_argument("--out", default="lensing_comparison.png", help="figure file")
    parser.add_argument("--zoom", type=int, default=60,
                        help="samples shown in the waveform panel (default 60)")
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

    dt = 1.0 / (2.0 * bandwidth)
    N = int(dg.SIDEREAL_DAY * nod / dt)
    fpo = dg.get_fpo(band, bandwidth, ov)

    cw_params = dg.read_signal_params(config)
    if cw_params is None:
        return 1

    # Lens model, for the |F| and arg F curves to compare against.
    run_seconds = (nseg - 1) * nod * dg.SIDEREAL_DAY + N * dt
    lens = None
    if sec.getboolean("lensing", fallback=False) or config.has_section("lensing"):
        lens = quiet(dg.prepare_lensing, config, cw_params, dt, run_seconds)

    print(f"Detector {detector_name}, {nseg} segments of {N} samples, "
          f"band {band} (fpo = {fpo:.6f} Hz)")

    runs = {"lensed": args.lensed, "unlensed": args.unlensed}
    measured = {name: {"amp": [], "phase": [], "err": []} for name in runs}
    model_abs, model_arg = [], []
    waveforms = None

    for seg in range(1, nseg + 1):
        segment_name = f"{seg:03d}"
        gps_time = starting_gpstime + (seg - 1) * nod * dg.SIDEREAL_DAY

        ephem_path = os.path.join(args.lensed, segment_name, detector_name, "DetSSB.bin")
        DetSSB, phir = read_ephemeris(ephem_path, N)

        make_detector = lambda: dg.build_detector(
            detector_name, segment_name, band, ov, gps_time, N,
            bandwidth, fpo, DetSSB, phir,
        )

        # Template and its 90-degree copy (a constant amplification of i).
        template = quiet(dg.generate_signal, make_detector(), cw_params,
                         starting_gpstime, EINSTEIN_FACTOR).astype(np.float64)
        template90 = quiet(dg.generate_signal, make_detector(), cw_params,
                           starting_gpstime, EINSTEIN_FACTOR,
                           amplification=np.full(N, 1j)).astype(np.float64)

        F = None
        if lens is not None:
            F = lens.amplification(gps_time - starting_gpstime, N)
            model_abs.append(float(np.mean(np.abs(F))))
            model_arg.append(float(np.angle(np.mean(F))))

        for name, root in runs.items():
            path = os.path.join(root, segment_name, detector_name,
                                f"xdat_{segment_name}_{band:04d}.bin")
            x = np.fromfile(path, dtype=np.float32).astype(np.float64)
            if x.size != N:
                raise ValueError(f"{path}: {x.size} samples, expected {N}")
            amp, phase, err = project(x, template, template90)
            measured[name]["amp"].append(amp)
            measured[name]["phase"].append(phase)
            measured[name]["err"].append(err)

        if seg == 1:                      # keep segment 1 for the waveform panel
            lensed_template = template
            if F is not None:
                lensed_template = quiet(
                    dg.generate_signal, make_detector(), cw_params,
                    starting_gpstime, EINSTEIN_FACTOR, amplification=F,
                ).astype(np.float64)
            waveforms = (template, lensed_template)

    segments = np.arange(1, nseg + 1)
    print(f"\n{'seg':>4} {'|F| lensed':>12} {'|F| unlensed':>14} {'model |F|':>11} "
          f"{'phase lensed':>14} {'model arg F':>12}")
    for i, seg in enumerate(segments):
        model_a = model_abs[i] if model_abs else float("nan")
        model_p = np.degrees(model_arg[i]) if model_arg else float("nan")
        print(f"{seg:>4} {measured['lensed']['amp'][i]:>12.4f} "
              f"{measured['unlensed']['amp'][i]:>14.4f} {model_a:>11.4f} "
              f"{np.degrees(measured['lensed']['phase'][i]):>13.1f}d {model_p:>11.1f}d")

    # ---- figure -----------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5))

    ax = axes[0]
    for name, style in (("lensed", "o-"), ("unlensed", "s-")):
        ax.errorbar(segments, measured[name]["amp"], yerr=measured[name]["err"],
                    fmt=style, capsize=3, label=f"{name} data")
    if model_abs:
        ax.plot(segments, model_abs, "k--", label="|F| from the lens model")
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("segment")
    ax.set_ylabel("recovered amplitude / injected")
    ax.set_title(f"Signal amplitude recovered from the data ({detector_name})")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    for name, style in (("lensed", "o-"), ("unlensed", "s-")):
        ax.plot(segments, np.degrees(measured[name]["phase"]), style, label=f"{name} data")
    if model_arg:
        ax.plot(segments, np.degrees(model_arg), "k--", label="arg F from the lens model")
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("segment")
    ax.set_ylabel("recovered phase [deg]")
    ax.set_title("Signal phase recovered from the data")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    n_show = min(args.zoom, N)
    t_show = np.arange(n_show) * dt
    ax.plot(t_show, waveforms[0][:n_show], label="unlensed signal")
    ax.plot(t_show, waveforms[1][:n_show], label="lensed signal", alpha=0.8)
    ax.set_xlabel("time into segment 1 [s]")
    ax.set_ylabel("strain / einstein_factor")
    ax.set_title("Injected signal, first samples of segment 1 (noise not shown)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())