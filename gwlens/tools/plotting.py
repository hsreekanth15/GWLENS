### Plotting modules

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



def get_scaled_fontsizes(fig_width,fig_height, base_figsize=(6, 4),base_fontsizes=None,min_scale=0.75,max_scale=1.5,):
    """
    Calculate font sizes based on figure dimensions.

    Uses geometric scaling to avoid overemphasizing either
    figure dimension.

    Some of this code is adapted from the LVK lensing group figure style guide in GWTC 4.0 paper
    https://zenodo.org/records/18163632 (o4a_lensing_figure_style.py)
    """
    if base_fontsizes is None:
        base_fontsizes = {"label": 14,"title": 14,"legend": 12,"xtick": 12,"ytick": 12,"annotation": 10,}

    width_scale = fig_width / base_figsize[0]
    height_scale = fig_height / base_figsize[1]

    scale = np.sqrt(width_scale * height_scale)
    scale = np.clip(scale, min_scale, max_scale)

    return {
        key: value * scale
        for key, value in base_fontsizes.items()
    }


def apply_figure_style(fig_width=7.2,fig_height=5,fontsize=None,usetex=False,dpi=150,ont_family="STIXGeneral",font_scale=1.0,):
    """
    Apply consistent publication-quality Matplotlib settings.
    Some of this code is adapted from the LVK lensing group figure style guide in GWTC 4.0 paper
    https://zenodo.org/records/18163632 (o4a_lensing_figure_style.py)
    """

    fonts = get_scaled_fontsizes(fig_width, fig_height)

    if fontsize is not None:
        fonts.update(fontsize)

    fonts = {
        key: value * font_scale
        for key, value in fonts.items()
    }

    plt.rcParams.update({

        # Figure
        "figure.figsize": (fig_width, fig_height),
        "figure.dpi": dpi,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Fonts
        "text.usetex": usetex,
        "font.family": font_family,
        "mathtext.fontset": "cm",

        # Font sizes
        "axes.labelsize": fonts["label"],
        "axes.titlesize": fonts["title"],
        "legend.fontsize": fonts["legend"],
        "xtick.labelsize": fonts["xtick"],
        "ytick.labelsize": fonts["ytick"],

        # Axes
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "axes.labelpad": 6,
        "axes.titlepad": 10,

        # Ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1,
        "ytick.major.width": 1,

        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 6,

        # Legend
        "legend.frameon": False,
        "legend.handlelength": 2.0,

        # Export
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    return fonts




def create_figure(
    width=7.2,
    height=5,
    nrows=1,
    ncols=1,
    figsize=None,
    sharex=False,
    sharey=False,
    squeeze=True,
    constrained_layout=True,
):
    """
    Create a consistently styled figure and axes.
    """

    if figsize is None:
        figsize = (width, height)

    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        layout="constrained" if constrained_layout else None,
    )

    return fig, ax




def format_axis(
    ax,
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    ylim=None,
    grid=False,
    minor_ticks=True,
    top_right_spines=False,
):
    """
    Apply consistent formatting to an individual axis.
    """

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=5,
        width=1,
        top=top_right_spines,
        right=top_right_spines,
    )

    ax.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=3,
        width=0.8,
        top=top_right_spines,
        right=top_right_spines,
    )

    if minor_ticks:
        ax.minorticks_on()

    if grid:
        ax.grid(
            True,
            which="major",
            linestyle="--",
            linewidth=0.6,
            alpha=0.35,
        )

    ax.spines["top"].set_visible(top_right_spines)
    ax.spines["right"].set_visible(top_right_spines)

    return ax


def add_legend(
    ax,
    loc="best",
    ncol=1,
    title=None,
    frameon=False,
    fontsize=None,
    **kwargs,
):
    """
    Consistent legend formatting.
    """

    return ax.legend(
        loc=loc,
        ncol=ncol,
        title=title,
        frameon=frameon,
        fontsize=fontsize,
        handlelength=2,
        columnspacing=1.2,
        **kwargs,
    )



def save_figure(
    fig,
    filename,
    formats=None,
    dpi=300,
    transparent=False,
):
    """
    Save a figure in one or multiple formats.
    """

    filename = Path(filename)

    if formats is None:
        formats = [filename.suffix.lstrip(".") or "pdf"]

    filename.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        output = filename.with_suffix(f".{fmt}")

        fig.savefig(
            output,
            dpi=dpi,
            bbox_inches="tight",
            transparent=transparent,
        )

    return True
