import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
from scipy import stats

from . import analysis
from .experiments import (
    AuditoryFeaturesResult, ProductionResult, RecognitionResult)


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


def cochleogram(data, time, freqs, ax=None, cax=None, cbar=True):
    if data.min() >= 0.0:
        cmap = plt.cm.get_cmap()
    elif np.allclose(data.max() + data.min(), 0, atol=1e-5):
        cmap = plt.cm.RdBu
    else:
        midpoint = np.abs(data.min()) / (data.max() - data.min())
        cmap = shiftedcmap(plt.cm.RdBu, midpoint=midpoint)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    mesh = ax.pcolormesh(time, freqs, data.T,
                         linewidth=0, rasterized=True, cmap=cmap)
    ax.set_yscale('log')
    ax.set_yticks((200, 1000, 2000, 4000, 8000))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylabel('Frequency (Hz)')
    ax.set_ylim(freqs[0], freqs[-1])
    ax.set_xlim(time[0], time[-1])
    if cbar and cax is None:
        fig.colorbar(mesh, pad=0.015, use_gridspec=True)
    elif cbar:
        fig.colorbar(mesh, ticklocation='right', cax=cax)
    sns.despine(ax=ax)


def prep_data(data, columns, x_keys, x_label, y_label,
              relative_to=None, group_by=None, filter_by=None):
    data = data.copy()  # Make a copy, as we modify it

    # Make columns relative to other columns
    relative_to = [] if relative_to is None else relative_to
    for col, rel in zip(columns, relative_to):
        data[col] /= data[rel].mean()

    filter_by = [] if filter_by is None else filter_by

    extra_keys = ['seed']
    extra_keys.extend([key for key, val in filter_by])
    extra_keys.extend(relative_to)
    if group_by is not None:
        extra_keys.append(group_by)

    # Get the requested columns, and the one we're grouping by
    data = pd.concat([data[[c] + extra_keys] for c in columns],
                     keys=x_keys, names=[x_label])
    # Merge all of the columns into one
    data[y_label] = np.nan
    for c in columns:
        data[y_label].fillna(data[c], inplace=True)
        del data[c]

    # Make the index (`x_label`) into a column
    data.reset_index(level=0, inplace=True)
    # Only take what we're filtering by
    for key, val in filter_by:
        data = data[data[key] == val]
    return data


def compare(data, columns, x_keys, x_label, y_label,
            relative_to=None, group_by=None, filter_by=None,
            plot_f=sns.violinplot, **plot_args):
    data = prep_data(data, columns, x_keys, x_label, y_label,
                     relative_to, group_by, filter_by)
    # Go Seaborn!
    plot_f(x=x_label, y=y_label, hue=group_by, data=data, **plot_args)
    sns.despine()


def timeseries(data, columns, x_keys, x_label, y_label,
               relative_to=None, group_by=None, filter_by=None,
               **plot_args):
    data = prep_data(data, columns, x_keys, x_label, y_label,
                     relative_to, group_by, filter_by)
    data[group_by] = data[group_by].apply(float)
    # data.sort_values(by=group_by, inplace=True)

    # plot_args.setdefault("err_style", "ci_bars")
    # plot_args.setdefault("estimator", stats.nanmedian)
    plot_args.setdefault("estimator", stats.nanmean)
    plot_args.setdefault("ci", 95)

    sns.tsplot(data=data,
               time=group_by,
               unit='seed',
               value=y_label,
               condition=x_label,
               **plot_args)
    sns.despine()

root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def savefig(fig, subdir, name, ext='svg'):
    path = os.path.join(root, 'plots', subdir, '%s.%s' % (name, ext))
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fig.savefig(path)


def setup(figsize=None):
    if figsize is not None:
        plt.rc('figure', figsize=figsize)
    sns.set_style('white')
    sns.set_style('ticks')


def plot_traj(traj, zscore=False, ax=None, cbar=True):
    if zscore:
        traj = stats.zscore(traj, axis=0)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    mesh = ax.pcolormesh(traj.T, linewidth=0, rasterized=True)
    if cbar:
        fig.colorbar(mesh, pad=0.015, use_gridspec=True)
    ax.set_ylim(top=traj.shape[1])
    ax.set_xlim(right=traj.shape[0])
    ax.set_yticks(())
    sns.despine(left=True, ax=ax)
    if fig is not None:
        fig.tight_layout()
    return fig, ax


def plot_trajs(traj1, traj2, zscore=(False, False), cbar=True):
    fig = plt.figure(figsize=(6, 4))
    ax1 = plt.subplot(2, 1, 1)
    plot_traj(traj1, zscore[0], ax=ax1, cbar=cbar)
    ax1.set_xticks(())
    sns.despine(bottom=True, left=True, ax=ax1)
    ax2 = plt.subplot(2, 1, 2)
    plot_traj(traj2, zscore[1], ax=ax2, cbar=cbar)
    fig.tight_layout()
    return fig, ax1, ax2


# #####################################
# Model 1: Neural cepstral coefficients
# #####################################

def ncc_accuracy(columns, vary, hue_order, relative=True, filter_by=None):
    df = analysis.load_results(AuditoryFeaturesResult, columns + ['phones'])
    filter_by = [('phones', 'consonants')] if filter_by is None else filter_by
    phones = 'Consonants'
    common_args = {'x_label': 'Data set',
                   'y_label': 'Correctness',
                   'hue_order': hue_order,
                   'filter_by': filter_by,
                   'group_by': vary}

    if len(filter_by) > 0:
        for k, v in filter_by:
            if k == 'phones':
                phones = v
                common_args['y_label'] = ('%s correctness'
                                          % phones[:-1].capitalize())

    figs = []
    for plot_f in [sns.violinplot, sns.barplot]:
        acc = plt.figure()
        if relative:
            compare(df,
                    columns=['ncc_train_acc', 'ncc_test_acc'],
                    relative_to=['mfcc_train_acc', 'mfcc_test_acc'],
                    x_keys=['Training', 'Testing'],
                    plot_f=plot_f,
                    **common_args)
            plt.axhline(1.0, c='k', ls=':')
            plt.ylim(bottom=0.95)
            plt.ylabel('Relative %s correctness' % phones[:-1].lower())
        else:
            compare(df,
                    columns=['mfcc_train_acc', 'mfcc_test_acc',
                             'ncc_train_acc', 'ncc_test_acc'],
                    x_keys=['MFCC training', 'MFCC testing',
                            'NCC training', 'NCC testing'],
                    plot_f=plot_f,
                    **common_args)
            plt.ylabel('%s correctness' % phones[:-1].capitalize())
        plt.xlabel("")

        acc.tight_layout()
        savefig(acc, 'results', 'ncc-%s-%sacc-%s' % (
            vary, 'r' if relative else '', plot_f.__name__[0]))
        figs.append(acc)
    return tuple(figs)


def ncc_tsaccuracy(columns, vary, relative=True, filter_by=None):
    df = analysis.load_results(AuditoryFeaturesResult, columns + ['phones'])
    filter_by = [('phones', 'consonants')] if filter_by is None else filter_by
    phones = 'Consonants'
    common_args = {'x_label': 'Data set',
                   'y_label': 'Correctness',
                   'filter_by': filter_by,
                   'group_by': vary}

    if len(filter_by) > 0:
        for k, v in filter_by:
            if k == 'phones':
                phones = v
                common_args['y_label'] = '%s correctness' % v[:-1].capitalize()

    acc = plt.figure()
    if relative:
        timeseries(df,
                   columns=['ncc_train_acc', 'ncc_test_acc'],
                   relative_to=['mfcc_train_acc', 'mfcc_test_acc'],
                   x_keys=['Training', 'Testing'],
                   **common_args)
        plt.axhline(1.0, c='k', ls=':')
        plt.ylabel('Relative %s correctness' % phones[:-1].lower())
    else:
        timeseries(df,
                   columns=['mfcc_train_acc', 'mfcc_test_acc',
                            'ncc_train_acc', 'ncc_test_acc'],
                   x_keys=['MFCC training', 'MFCC testing',
                           'NCC training', 'NCC testing'],
                   **common_args)
        plt.ylabel('%s correctness' % phones[:-1].capitalize())
    plt.xlabel("")
    acc.tight_layout()
    savefig(acc, 'results', 'ncc-%s-acc-t' % vary)
    return acc


def ncc_time(columns, vary, hue_order, filter_by=None):
    df = analysis.load_results(AuditoryFeaturesResult, columns + ['phones'])
    filter_by = [('phones', 'consonants')] if filter_by is None else filter_by

    time = plt.figure()
    compare(df,
            columns=['mfcc_time', 'ncc_time', 'mfcc_fit_time', 'ncc_fit_time'],
            group_by=vary,
            filter_by=filter_by,
            x_keys=['MFCC', 'NCC', 'SVM for MFCC', 'SVM for NCC'],
            x_label='Scenario',
            y_label='Time (s)',
            plot_f=sns.barplot,
            hue_order=hue_order)
    plt.ylabel("Time (s)")
    plt.xlabel("")
    time.tight_layout()
    savefig(time, 'results', 'ncc-%s-time' % vary)
    return time


# ############################
# Model 2: Syllable production
# ############################

def prod_time(key, x_label):
    df = analysis.load_results(ProductionResult, [key])

    cmp_args = {'group_by': key,
                'x_label': x_label}

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(2, 2, 1)
    timeseries(df,
               columns=['accuracy'],
               x_keys=['Accuracy'],
               y_label='Accuracy',
               ax=ax,
               **cmp_args)
    ax.axhline(analysis.gs_accuracy_baseline, c='k', ls=':')
    ax.set_xlabel("")
    ax.set_xticks(())
    l = ax.legend()
    l.remove()

    ax = plt.subplot(2, 2, 2)
    timeseries(df,
               columns=['n_sub', 'n_del', 'n_ins'],
               x_keys=['Substitutions', 'Insertions', 'Deletions'],
               y_label='Count',
               ax=ax,
               **cmp_args)
    ax.set_xlabel("")
    ax.set_xticks(())
    ax.legend(loc='best', title="")

    ax = plt.subplot(2, 2, 3)
    timeseries(df,
               columns=['timing_mean', 'timing_var'],
               x_keys=['Timing mean', 'Timing variance'],
               y_label='Timing (s)',
               ax=ax,
               **cmp_args)
    ax.legend(loc='best', title="")
    ax.set_xlabel(x_label)

    ax = plt.subplot(2, 2, 4)
    timeseries(df,
               columns=['cooccur'],
               x_keys=['Co-occurrence'],
               y_label='Co-occurrence',
               ax=ax,
               **cmp_args)
    ax.axhline(df["co_chance"].mean(), c='k', ls=":")
    l = ax.legend()
    l.remove()
    ax.set_xlabel(x_label)

    fig.tight_layout()
    savefig(fig, 'results', 'prod-%s' % key)
    return fig


def prod_cmp(key, x_label, hue_order):
    df = analysis.load_results(ProductionResult, [key])

    cmp_args = {'group_by': key,
                'x_label': x_label,
                'hue_order': hue_order}

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(2, 2, 1)
    compare(df,
            columns=['accuracy'],
            x_keys=['Accuracy'],
            y_label='Accuracy',
            plot_f=sns.barplot,
            ax=ax,
            **cmp_args)
    ax.axhline(analysis.gs_accuracy_baseline, c='k', ls=':')
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    l = ax.legend()
    l.remove()

    ax = plt.subplot(2, 2, 2)
    compare(df,
            columns=['n_sub', 'n_del', 'n_ins'],
            x_keys=['Substitutions', 'Insertions', 'Deletions'],
            y_label='Count',
            plot_f=sns.barplot,
            ax=ax,
            **cmp_args)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    l = ax.legend()
    l.remove()

    ax = plt.subplot(2, 2, 3)
    compare(df,
            columns=['timing_mean', 'timing_var'],
            x_keys=['Timing mean', 'Timing variance'],
            y_label='Timing',
            plot_f=sns.barplot,
            ax=ax,
            **cmp_args)
    ax.set_ylabel("Timing")
    ax.set_xlabel("")

    ax = plt.subplot(2, 2, 4)
    compare(df,
            columns=['cooccur'],
            x_keys=['Co-occurrence'],
            y_label='Co-occurrence',
            plot_f=sns.barplot,
            ax=ax,
            **cmp_args)
    ax.axhline(df["co_chance"].mean(), c='k', ls=":")
    ax.set_ylabel("Co-ocurrence")
    ax.set_xlabel("")

    fig.tight_layout()
    savefig(fig, 'results', 'prod-%s' % key)
    return fig


# #############################
# Model 3: Syllable recognition
# #############################

def recog_time(key, x_label, n_syllables=[3., 3.]):
    df = analysis.load_results(RecognitionResult, [key])

    cmp_args = {'group_by': key,
                'x_label': x_label}

    fig = plt.figure(figsize=(8, 6))

    ax = plt.subplot(2, 2, 1)
    timeseries(df,
               columns=['acc'],
               x_keys=['Accuracy'],
               y_label='Accuracy',
               ax=ax,
               **cmp_args)
    ax.set_xlabel("")
    ax.set_xticks(())
    l = ax.legend()
    l.remove()

    ax = plt.subplot(2, 2, 2)
    timeseries(df,
               columns=['n_sub', 'n_ins', 'n_del'],
               x_keys=['Substitutions', 'Insertions', 'Deletions'],
               y_label='Count',
               ax=ax,
               **cmp_args)
    ax.set_xlabel("")
    ax.set_xticks(())
    ax.legend(loc='best', title="")

    ax = plt.subplot(2, 2, 3)
    timeseries(df,
               columns=['tdiff_mean', 'tdiff_var'],
               x_keys=['Timing mean', 'Timing variance'],
               y_label='Timing',
               ax=ax,
               **cmp_args)
    ax.legend(loc='best', title="")
    ax.set_xlabel(x_label)

    ax = plt.subplot(2, 2, 4)
    timeseries(df,
               columns=['memory_acc'],
               x_keys=['Memory accuracy'],
               y_label='Memory accuracy',
               ax=ax,
               **cmp_args)
    l, r = ax.get_xlim()
    s = np.asarray(n_syllables)
    x = np.linspace(l, r, s.size)
    ax.plot(x, 1./s, c='k', ls=':')
    l = ax.legend()
    l.remove()
    ax.set_xlabel(x_label)

    fig.tight_layout()
    savefig(fig, 'results', 'recog-%s' % key)

    return fig


def recog_cmp(key, x_label, hue_order):
    df = analysis.load_results(RecognitionResult, [key])

    cmp_args = {'group_by': key,
                'x_label': x_label,
                'hue_order': hue_order}

    fig = plt.figure(figsize=(8, 6))
    ax = plt.subplot(2, 2, 1)
    compare(df,
            columns=['acc'],
            x_keys=['Accuracy'],
            y_label='Accuracy',
            plot_f=sns.barplot,
            ax=ax,
            **cmp_args)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("")
    l = ax.legend()
    l.remove()

    ax = plt.subplot(2, 2, 2)
    compare(df,
            columns=['n_sub', 'n_ins', 'n_del'],
            x_keys=['Substitutions', 'Insertions', 'Deletions'],
            y_label='Count',
            plot_f=sns.barplot,
            ax=ax,
            **cmp_args)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    l = ax.legend()
    l.remove()

    ax = plt.subplot(2, 2, 3)
    compare(df,
            columns=['tdiff_mean', 'tdiff_var'],
            x_keys=['Timing mean', 'Timing variance'],
            y_label='Timing',
            plot_f=sns.barplot,
            ax=ax,
            **cmp_args)
    ax.set_ylabel("Timing")
    ax.set_xlabel("")

    ax = plt.subplot(2, 2, 4)
    compare(df,
            columns=['memory_acc'],
            x_keys=['Memory accuracy'],
            y_label='Accuracy',
            plot_f=sns.barplot,
            ax=ax,
            **cmp_args)
    ax.axhline(1./3, c='k', ls=':')  # Always 3 syllables
    ax.set_ylabel("Co-ocurrence")
    ax.set_xlabel("")

    fig.tight_layout()
    savefig(fig, 'results', 'recog-%s' % key)
    return fig
