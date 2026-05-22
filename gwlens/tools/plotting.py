### Plotting modules

#credit: Eungwang Seo

import matplotlib.pyplot as plt

def get_scaled_fontsizes(fig_width, fig_height, base_figsize=(6, 4), base_fontsizes=None):
    if base_fontsizes is None:
        base_fontsizes = {
            'label': 14,
            'title': 14,
            'legend': 12,
            'xtick': 12,
            'ytick': 12
        }
    width_scale = fig_width / base_figsize[0]
    height_scale = fig_height / base_figsize[1]
    scale = (width_scale + height_scale) / 2.0

    return {k: v * scale for k, v in base_fontsizes.items()}


def apply_figure_style(fig_width=7.2, fig_height=5):
    fonts = get_scaled_fontsizes(fig_width, fig_height)
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'STIXGeneral',
        'mathtext.fontset': 'cm',
        'figure.figsize': (fig_width, fig_height),
        'axes.labelsize': fonts['label'],
        'axes.titlesize': fonts['title'],
        'legend.fontsize': fonts['legend'],
        'xtick.labelsize': fonts['xtick'],
        'ytick.labelsize': fonts['ytick'],
    })
