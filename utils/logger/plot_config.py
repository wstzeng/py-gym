import matplotlib.pyplot as plt

def set_plot_style():
    plt.xkcd()
    # plt.style.use('seaborn-v0_8-darkgrid')
    
    plt.rcParams.update({
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',

        # Axes and ticks
        'axes.titlesize': 20,
        'axes.titleweight': 'bold',
        'axes.labelsize': 16,
        'axes.labelweight': 'bold',
        'axes.grid': False,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,

        # Font fallback: Noto Sans -> DejaVu Sans -> sans-serif
        # 'font.family': 'sans-serif',
        # 'font.sans-serif': ['Noto Sans', 'DejaVu Sans', 'sans-serif'],

        # Line style
        'lines.linewidth': 2,
        'lines.markersize': 6,

        # Legend
        'legend.fontsize': 12,
        'legend.frameon': False,

        'axes.unicode_minus': False,
    })

    suptitle_style = {
        'fontsize': 18,
        'fontweight': 'bold',
        'ha': 'center',
    }

    return {'suptitle': suptitle_style}
