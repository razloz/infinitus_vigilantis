"""Charting routines for the Infinitus Vigilantis application."""
import pickle
import pandas
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from os import path
from time import sleep, time
from numpy import arange, vstack
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2025, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'tres leches'
plt.style.use('dark_background')
verbose = False


def cartography(symbol, features, candles, timestamps, batch_size=5,
                chart_path=None, chart_size=200, forecast=None):
    """Charting for IVy candles."""
    global plt
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    if not chart_path: chart_path = './charts/active.png'
    print('Plotting', chart_path)
    if verbose: print(f'Cartography: creating chart for {symbol}...')
    plot_features = ['open', 'high', 'low', 'close', 'volume', 'trend',
                     'price_wema', 'volume_wema', 'price_mid', 'volume_mid',
                     'price_dh', 'volume_dh', 'price_dl', 'volume_dl',]
    feature_indices = {f: features.index(f) for f in plot_features}
    data_trim = int(candles.shape[0])
    while data_trim % batch_size != 0:
        data_trim -= 1
    candles = candles[-data_trim:, :]
    data_len = len(candles)
    if data_len < data_trim:
        print('Error in cartography: data_len < data_trim')
        return None
    final_date = timestamps[-1]
    candles = candles[-chart_size:]
    timestamps = timestamps[-chart_size:]
    if len(timestamps) != chart_size:
        print('Error in cartography: len(timestamps) != chart_size')
        return None
    data_len = len(timestamps)
    if forecast is not None:
        f_size = chart_size + batch_size
        forecast = forecast.flatten()[-f_size:]
        timestamps += [' ' for _ in range(batch_size)]
        data_len = len(timestamps)
    data_range = range(data_len)
    features_range = range(len(candles))
    ohlc = ['open', 'high', 'low', 'close']
    ohlc = candles[:, [feature_indices[k] for k in ohlc]].flatten()
    fig = plt.figure(figsize=(19.20, 10.80), dpi=100, constrained_layout=False)
    sargs = dict(ncols=1, nrows=3, figure=fig, height_ratios=[6,2,1])
    spec = gridspec.GridSpec(**sargs)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(spec[2, 0], sharex=ax1)
    plt.xticks(ticks=data_range, labels=timestamps,
               rotation=13, fontweight='bold')
    plt.subplots_adjust(left=0.09, bottom=0.09, right=0.91,
                        top=0.91, wspace=0.01, hspace=0.01)
    ax1.grid(True, color=(0.3, 0.3, 0.3))
    ax1.set_ylabel('Price', fontweight='bold')
    ax1.set_xlim(((data_range[0] - 2), (data_range[-1] + 2)))
    ylim_low = ohlc.min()
    ylim_high = ohlc.max()
    ax1.set_ylim((ylim_low * 0.99, ylim_high * 1.01))
    ytick_step = 0.01 * ylim_high
    y_arange = arange(ylim_low, ylim_high, ytick_step, dtype=float)
    yticks_range = [round(float(i), 2) for i in y_arange]
    ax1.set_yticks(yticks_range)
    ax1.set_yticklabels(yticks_range)
    ax1.yaxis.set_major_locator(mticker.AutoLocator())
    ax1.yaxis.set_major_formatter(mticker.EngFormatter())
    ax1.xaxis.set_major_locator(mticker.AutoLocator())
    ax2.grid(True, color=(0.4, 0.4, 0.4))
    ax2.set_ylabel('Volume', fontweight='bold')
    ax2.yaxis.set_major_locator(mticker.AutoLocator())
    ax2.yaxis.set_major_formatter(mticker.EngFormatter())
    ax3.grid(True, color=(0.4, 0.4, 0.4))
    ax3.set_ylabel('Sentiment', fontweight='bold')
    ax3.set_ylim((-1.03, 1.03))
    xticks = ax1.xaxis.get_ticklabels()
    plt.setp(xticks[:], visible=False)
    plt.setp(ax2.xaxis.get_ticklabels()[:], visible=False)
    # Dynamic width stuff
    tbb = ax1.get_tightbbox(fig.canvas.get_renderer()).get_points()
    xb = tbb[1][0] - tbb[0][0]
    wid_base = (xb / data_len) * 0.5
    wid_wick = wid_base * 0.21
    wid_cdls = wid_base * 0.89
    wid_line = wid_base * 0.34
    # Fibonacci retracements
    pkws = {
        'alpha': 0.5,
        'color': '#F3E6D0',
        'label': None,
        'linestyle': 'dotted',
        'linewidth': wid_line,
        }
    fib_x = [data_range[0], data_range[-1]]
    fib_range = ylim_high - ylim_low
    fib_lines = [
        round(float(ylim_high), 2),
        0.118, 0.250, 0.382, 0.500, 0.618, 0.750, 0.882,
        round(float(ylim_low), 2),
        ]
    fib_len = len(fib_lines) - 1
    for i, v in enumerate(fib_lines):
        if i == 0 or i == fib_len:
            continue
        fib_lines[i] = round(float(ylim_high - (fib_range * v)), 2)
    fib_axes = ax1.twinx()
    fib_axes.grid(False)
    fib_axes.set_ylabel('Retracements', fontweight='bold')
    fib_axes.set_yticks(yticks_range)
    fib_labels = [str(n) if n in fib_lines else '' for n in yticks_range]
    fib_axes.set_yticklabels(fib_labels)
    fib_axes.yaxis.set_major_locator(mticker.FixedLocator(fib_lines))
    fib_axes.yaxis.set_major_formatter(mticker.EngFormatter())
    thick_lines = False
    for y in fib_lines:
        if thick_lines:
            pkws['linewidth'] = wid_line
        else:
            pkws['linewidth'] = wid_line * 0.9
        fib_axes.plot(fib_x, [y, y], **pkws)
        thick_lines = not thick_lines
    # Candle stuff
    pkws['alpha'] = 1.0
    cdl_open = candles[:, feature_indices['open']].flatten()
    cdl_high = candles[:, feature_indices['high']].flatten()
    cdl_low = candles[:, feature_indices['low']].flatten()
    cdl_close = candles[:, feature_indices['close']].flatten()
    cdl_vol = candles[:, feature_indices['volume']].flatten()
    y_loc = [ylim_low, ylim_high]
    labels_set = [False, False]
    data_end = features_range[-1]
    for i in features_range:
        x_loc = [i, i]
        # Candles
        wick_data = [cdl_low[i], cdl_high[i]]
        candle_data = [cdl_close[i], cdl_open[i]]
        ax1.plot(x_loc, wick_data, color='white',
                 linestyle='solid', linewidth=wid_wick, alpha=1)
        if cdl_close[i] > cdl_open[i]:
            cdl_color=(0.33, 1, 0.33, 1)
        else:
            cdl_color=(1, 0.33, 0.33, 1)
        ax1.plot(x_loc, candle_data, color=cdl_color,
                 linestyle='solid', linewidth=wid_cdls, alpha=1)
        # Volume
        volume_data = [0, cdl_vol[i]]
        ax2.plot(x_loc, volume_data, color=(0.33, 0.33, 1, 1),
                 linestyle='solid', linewidth=wid_cdls)
    # Per sample plots
    pkws = {'linestyle': 'solid', 'linewidth': wid_line}
    for key in plot_features:
        if key in ['open', 'high', 'low', 'close', 'volume', 'trend']:
            continue
        cdl_data = candles[:, feature_indices[key]].flatten()
        pkws['label'] = f'{key}: {round(float(cdl_data[-1]), 3)}'
        dev_feats = ['price_mid', 'price_dh', 'price_dl',
                     'volume_mid', 'volume_dh', 'volume_dl']
        pkws['label'] = None
        if key == 'price_wema':
            pkws['color'] = (0.4, 0.7, 0.4, 0.8)
            pkws['label'] = f'Money: {round(float(cdl_data[-1]), 3)}'
        elif key in dev_feats:
            pkws['linestyle'] = 'dotted'
            if key in ['price_mid', 'volume_mid']:
                pkws['linewidth'] = wid_line * 0.67
                if key == 'price_mid':
                    pkws['label'] = f'Dev/Mid: {round(float(cdl_data[-1]), 3)}'
            else:
                pkws['linewidth'] = wid_line * 0.87
            pkws['color'] = (0.7, 0.7, 1, 0.7)
        if key not in ['volume_wema', 'volume_mid', 'volume_dh', 'volume_dl']:
            ax1.plot(features_range, cdl_data, **pkws)
        else:
            ax2.plot(features_range, cdl_data, **pkws)
    # Plot Forecast
    if forecast is not None:
        zs = candles[:, features.index('price_zs')].flatten()
        zs[zs > 1] = 1
        zs[zs < -1] = -1
        forecast[forecast > 1] = 1
        forecast[forecast < -1] = -1
        pkws['linestyle'] = 'solid'
        pkws['linewidth'] = wid_line * 1.1
        pkws['alpha'] = 0.5
        pkws['color'] = '#6F00FF'
        pkws['label'] = 'Z-Score'
        ax3.plot(range(len(zs)), zs, **pkws)
        pkws['color'] = '#FFF5AB'
        pkws['label'] = 'Prediction'
        ax3.fill_between(range(len(forecast)), forecast, 0.0, **pkws)
    # Finalize
    rnc = round(float(cdl_close[-1]), 3)
    t = f'[ {rnc} ]   {symbol}  @  {final_date}'
    fig.suptitle(t, fontsize=18)
    plt.savefig(str(chart_path))
    plt.clf()
    plt.close()
    if verbose: print("Cartography: chart's done!")
    return False


def plot_candelabrum(sigil, symbols):
    """Candelabrum forecast in bar chart format."""
    xticks = range(len(symbols))
    plt.clf()
    fig = plt.figure(figsize=(30.00, 10.80))
    ax = fig.add_subplot()
    plt.subplots_adjust(left=0.03, bottom=0.13, right=0.97, top=0.87)
    ax.set_xlabel('Symbol')
    ax.set_ylabel('Prob')
    ax.set_xticks(xticks)
    ax.set_xticklabels(symbols, fontweight='light')
    ax.tick_params(axis='x', which='major', labelsize=7, pad=5, rotation=15)
    ax.grid(True, color=(0.4, 0.4, 0.4))
    width_adj = [0.7, 0.5, 0.3]
    plot_params = dict(linestyle='solid', color=(0.34, 1, 0.34, 1))
    plot_params['label'] = f'Symbol Probability'
    ax.plot(sigil, **plot_params)
    title = 'Candelabrum'
    fig.suptitle(title, fontsize=18)
    fig.legend(ncol=1, fontsize='xx-large', fancybox=True)
    plt.savefig('./resources/candelabrum.png')
    plt.close()
