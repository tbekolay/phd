import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def shiftedcmap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    """Offset the 'center' of a colormap.

    Useful for data with a negative min and positive max and you
    want the middle of the colormap's dynamic range to be at zero.

    Parameters
    ----------
    cmap : The matplotlib colormap to be altered
    start : Offset from lowest point in the colormap's range.
            Defaults to 0.0 (no lower ofset). Should be between
            0.0 and `midpoint`.
    midpoint : The new center of the colormap. Defaults to
               0.5 (no shift). Should be between 0.0 and 1.0. In
               general, this should be  1 - vmax/(vmax + abs(vmin))
               For example if your data range from -15.0 to +5.0 and
               you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
    stop : Offset from highets point in the colormap's range.
           Defaults to 1.0 (no upper ofset). Should be between
           `midpoint` and 1.0.

    From http://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib  # noqa
    """
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False),
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)
        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap


def cochleogram(data, time, freqs, cmap=plt.cm.RdBu):
    if data.min() >= 0.0:
        cmap = plt.cm.Blues
    elif not np.allclose(data.max() + data.min(), 0, atol=1e-5):
        midpoint = np.abs(data.min()) / (data.max() - data.min())
        cmap = shiftedcmap(cmap, midpoint=midpoint)

    plt.pcolormesh(time, freqs, data.T, cmap=cmap)
    plt.yscale('log')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (ms)')
    plt.axis('tight')
    sns.despine()
    plt.colorbar()
    plt.tight_layout()