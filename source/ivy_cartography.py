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
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
plt.style.use('dark_background')
verbose = False


def cartography(symbol, dataframe, chart_path=None, cheese=None,
                chart_size=0, batch_size=34, adj='1D'):
    """Charting for IVy candles."""
    global plt
    if not chart_path: chart_path = './charts/active.png'
    if verbose: print(f'Cartography: creating chart for {symbol}...')
    features = ['open', 'high', 'low', 'close', 'volume',
                'price_wema', 'volume_wema', 'price_mid', 'volume_mid',
                'price_dh', 'volume_dh', 'price_dl', 'volume_dl',]
    features = dataframe[features]
    data_len = len(features.index)
    if chart_size == 0:
        chart_size = data_len - batch_size
    if data_len > chart_size:
        features = features[-chart_size:]
        data_len = len(features.index)
    data_range = range(data_len)
    ts_lbls = features.index.tolist()
    ts_last = ts_lbls[-1]
    moirai_metrics = ''
    blank_space = ''.join(' ' for i in range(40))
    if cheese:
        stack_next = False
        ignore_keys = ['symbol', 'compass', 'trades', 'signals']
        for key in cheese.keys():
            if key not in ignore_keys:
                addendum = f'{key}: {cheese[key]}'
                moirai_metrics += str(addendum + blank_space[len(addendum):])
                moirai_metrics += f'\n'
        moirai_metrics = moirai_metrics[:-2]
        compass = vstack(cheese['compass'])[-chart_size:]
        trades = vstack(cheese['trades'])[-chart_size:]
        signals = vstack(cheese['signals'])[-chart_size:]
    fig = plt.figure(figsize=(19.20, 10.80), dpi=100, constrained_layout=False)
    sargs = dict(ncols=1, nrows=2, figure=fig, height_ratios=[4,1])
    spec = gridspec.GridSpec(**sargs)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0], sharex=ax1)
    plt.xticks(ticks=data_range, labels=ts_lbls, rotation=13, fontweight='bold')
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.92,
                        top=0.92, wspace=0, hspace=0.01)
    ax1.grid(True, color=(0.3, 0.3, 0.3))
    ax1.set_ylabel('Price', fontweight='bold')
    ax1.set_xlim(((data_range[0] - 2), (data_range[-1] + 2)))
    ohlc = features[['open', 'high', 'low', 'close']]
    ylim_low = min(ohlc.min())
    ylim_high = max(ohlc.max())
    ax1.set_ylim((ylim_low * 0.99, ylim_high * 1.01))
    ytick_step = 0.01 * ylim_high
    yticks_range = [round(i,2) for i in arange(ylim_low, ylim_high, ytick_step)]
    ax1.set_yticks(yticks_range)
    ax1.set_yticklabels(yticks_range)
    ax1.yaxis.set_major_locator(mticker.AutoLocator())
    ax1.yaxis.set_major_formatter(mticker.EngFormatter())
    ax1.xaxis.set_major_locator(mticker.AutoLocator())
    ax2.grid(True, color=(0.4, 0.4, 0.4))
    ax2.set_ylabel('Volume', fontweight='bold')
    ax2.yaxis.set_major_locator(mticker.AutoLocator())
    ax2.yaxis.set_major_formatter(mticker.EngFormatter())
    xticks = ax1.xaxis.get_ticklabels()
    plt.setp(xticks[:], visible=False)
    # Dynamic width stuff
    tbb = ax1.get_tightbbox(fig.canvas.get_renderer()).get_points()
    xb = tbb[1][0] - tbb[0][0]
    wid_base = (xb / chart_size) * 0.5
    wid_wick = wid_base * 0.21
    wid_cdls = wid_base * 0.89
    wid_line = wid_base * 0.34
    # Candle stuff
    cdl_open = features.pop('open')
    cdl_high = features.pop('high')
    cdl_low = features.pop('low')
    cdl_close = features.pop('close')
    cdl_vol = features.pop('volume')
    y_loc = [ylim_low, ylim_high]
    labels_set = [False, False]
    cheese_color = dict(cheddar='#FF9600', gouda='#FFE88E')
    data_end = data_range[-1]
    for i in data_range:
        x_loc = [i, i]
        # Candles
        if cdl_open[i] is not None:
            if cheese:
                signal = signals[i]
                if signal == 0:
                    ax1.plot(x_loc, y_loc, color=(0.33, 1, 0.33, 0.5),
                             linestyle='solid', linewidth=wid_cdls)
                elif signal == 2:
                    ax1.plot(x_loc, y_loc, color=(1, 0.33, 0.33, 0.5),
                             linestyle='solid', linewidth=wid_cdls)
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
    for key in features.keys():
        cdl_data = features[key]
        cdl_range = range(len(cdl_data))
        pkws['label'] = f'{key}: {round(cdl_data[-1], 2)}'
        dev_feats = ['price_mid', 'price_dh', 'price_dl',
                     'volume_mid', 'volume_dh', 'volume_dl']
        pkws['label'] = None
        if key == 'price_wema':
            pkws['color'] = (0.4, 0.7, 0.4, 0.8)
            pkws['label'] = f'Money: {round(cdl_data[-1], 3)}'
        elif key in dev_feats:
            pkws['linestyle'] = 'dotted'
            if key in ['price_mid', 'volume_mid']:
                pkws['linewidth'] = wid_line * 0.67
                if key == 'price_mid':
                    pkws['label'] = f'Dev/Mid: {round(cdl_data[-1], 3)}'
            else:
                pkws['linewidth'] = wid_line * 0.87
            pkws['color'] = (0.7, 0.7, 1, 0.7)
        if key not in ['volume_wema', 'volume_mid', 'volume_dh', 'volume_dl']:
            ax1.plot(cdl_range, cdl_data, **pkws)
        else:
            ax2.plot(cdl_range, cdl_data, **pkws)
    # Finalize
    res = adj if adj else 'None'
    rnc = round(cdl_close[-1], 3)
    t = f'[ {rnc} ]   {symbol}  @  {ts_last} (resample: {res})'
    fig.suptitle(t, fontsize=18)
    plt.savefig(str(chart_path))
    plt.clf()
    plt.close()
    if verbose: print("Cartography: chart's done!")
    return False


def plot_candelabrum(sigil, symbols):
    """Candelabrum forecast in bar chart format."""
    xticks = range(sigil.shape[1])
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
    colors = [(0.34, 0.34, 1, 1), (0.34, 1, 0.34, 1), (1, 0.34, 0.34, 1)]
    colors_set = 0
    for day in range(sigil.shape[0]):
        plot_params = dict(linestyle='solid', color=colors[day])
        if colors_set < 3:
            plot_params['label'] = f'Forecast Day {day + 1}'
            colors_set += 1
        ax.plot(sigil[day], **plot_params)
    title = f'Candelabrum probabilities over the next {sigil.shape[0]} days'
    fig.suptitle(title, fontsize=18)
    fig.legend(ncol=1, fontsize='xx-large', fancybox=True)
    plt.savefig('./resources/candelabrum.png')
    plt.close()
