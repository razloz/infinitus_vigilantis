"""Charting routines for the Infinitus Vigilantis application."""

import pickle
import pandas
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from source.ivy_commons import ivy_dispatcher
from source.ivy_candles import composite_index
from source.ivy_candles import Candelabrum
from os import path
from time import sleep
from time import time

plt.style.use('dark_background')


def cartography(symbol, dataframe, cheese=None,
                chart_path='./charts/active.png', *ignore):
    """Charting for IVy candles."""
    global plt
    plt.close('all')
    print(f'Cartography: creating chart for {symbol}...')
    timestamps = dataframe.index.tolist()
    data_range = range(len(timestamps))
    cdl_open = dataframe['open'].tolist()
    cdl_high = dataframe['high'].tolist()
    cdl_low = dataframe['low'].tolist()
    cdl_close = dataframe['close'].tolist()
    cdl_vol = dataframe['volume'].tolist()
    cdl_wema = dataframe['money_wema'].tolist()
    cdl_mid = dataframe['money_mid'].tolist()
    cdl_dh = dataframe['money_dh'].tolist()
    cdl_dl = dataframe['money_dl'].tolist()
    vol_wema = dataframe['volume_wema'].tolist()
    vol_mid = dataframe['volume_mid'].tolist()
    vol_dh = dataframe['volume_dh'].tolist()
    vol_dl = dataframe['volume_dl'].tolist()
    fig = plt.figure(figsize=(13, 8), constrained_layout=False)
    sargs = dict(ncols=1, nrows=2, figure=fig, height_ratios=[4,1])
    spec = gridspec.GridSpec(**sargs)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0], sharex=ax1)
    plt.xticks(data_range, timestamps, rotation=21, fontweight='bold')
    plt.subplots_adjust(left=0.08, bottom=0.13, right=0.92,
                        top=0.90, wspace=0, hspace=0.08)
    ax1.grid(True, color=(0.3, 0.3, 0.3))
    ax1.set_ylabel('Price', fontweight='bold')
    ax1.set_xlim(((data_range[0] - 2), (data_range[-1] + 2)))
    ylim_low = min(cdl_low)
    ylim_high = max(cdl_high)
    ax1.set_ylim((ylim_low * 0.98, ylim_high * 1.02))
    ax1.set_yticks(cdl_close)
    ax1.set_yticklabels(cdl_close)
    ax1.yaxis.set_major_formatter(mticker.EngFormatter())
    ax1.yaxis.set_major_locator(mticker.AutoLocator())
    ax1.xaxis.set_major_formatter(mticker.IndexFormatter(timestamps))
    ax1.xaxis.set_major_locator(mticker.AutoLocator())
    ax2.grid(True, color=(0.4, 0.4, 0.4))
    ax2.set_ylabel('Volume', fontweight='bold')
    ax2.yaxis.set_major_formatter(mticker.EngFormatter())
    ax2.yaxis.set_major_locator(mticker.AutoLocator())
    plt.setp(ax1.xaxis.get_ticklabels()[:], visible=False)

    # Per candle plots
    signal_y = [min(cdl_dl), max(cdl_dh)]
    for i in data_range:
        x_loc = [i, i]
        # Signals
        if cheese:
            cdl_date = timestamps[i].strftime('%Y-%m-%d %H:%M')
            sig_args = dict(linestyle='solid', linewidth=1.5)
            if cdl_date in cheese:
                buy_sig = cheese[cdl_date]['buy']
                sell_sig = cheese[cdl_date]['sell']
                for sig in buy_sig:
                    if sig[0] == symbol:
                        sig_args['color'] = (0, 1, 0, 0.5)
                        ax1.plot(x_loc, signal_y, **sig_args)
                for sig in sell_sig:
                    if sig[0] == symbol:
                        sig_args['color'] = (1, 0, 0, 0.5)
                        ax1.plot(x_loc, signal_y, **sig_args)
        # Candles
        wick_data = [cdl_low[i], cdl_high[i]]
        candle_data = [cdl_close[i], cdl_open[i]]
        ax1.plot(x_loc, wick_data, color='white',
                 linestyle='solid', linewidth=1.5, alpha=1)
        if cdl_close[i] > cdl_open[i]:
            cdl_color=(0.33, 1, 0.33, 1)
        else:
            cdl_color=(1, 0.33, 0.33, 1)
        ax1.plot(x_loc, candle_data, color=cdl_color,
                 linestyle='solid', linewidth=3, alpha=1)
        # Volume
        volume_data = [0, cdl_vol[i]]
        ax2.plot(x_loc, volume_data, color=(0.33, 0.33, 1, 1),
                 linestyle='solid', linewidth=3)

    pkws = {'linestyle': 'solid', 'linewidth': 1}
    pkws['label'] = f'Money: {round(cdl_wema[-1], 2)}'
    pkws['color'] = (0.1, 0.5, 0.1, 0.8)
    ax1.plot(data_range, cdl_wema, **pkws)
    pkws['label'] = None
    ax2.plot(data_range, vol_wema, **pkws)

    pkws['label'] = f'Mid: {round(cdl_mid[-1], 2)}'
    pkws['color'] = (0.8, 0.8, 1, 0.8)
    ax1.plot(data_range, cdl_mid, **pkws)
    pkws['label'] = None
    ax2.plot(data_range, vol_mid, **pkws)

    pkws['linestyle'] = 'dotted'
    pkws['linewidth'] = 0.75
    pkws['label'] = f'DevHigh: {round(cdl_dh[-1], 2)}'
    ax1.plot(data_range, cdl_dh, **pkws)
    pkws['label'] = None
    ax2.plot(data_range, vol_dh, **pkws)

    pkws['label'] = f'DevLow: {round(cdl_dl[-1], 2)}'
    ax1.plot(data_range, cdl_dl, **pkws)
    pkws['label'] = None
    ax2.plot(data_range, vol_dl, **pkws)

    ts = timestamps[-1].strftime('%Y-%m-%d %H:%M')
    t = f'{symbol}: {cdl_close[-1]} @ {ts}'
    fig.legend(title=t, ncol=6, loc='upper center', fontsize='large')
    plt.savefig(str(chart_path))
    plt.close(fig)
    print("Cartography: chart's done!")
    return False


def cartographer():
    """Charting daemon."""
    cdlm = Candelabrum()
    get_candles = cdlm.load_candles
    ivy_ndx = composite_index('./indexes/custom.ndx')
    charting = True
    last_poll = 0
    try:
        print(f'Cartographer: working on {len(ivy_ndx)} symbols.')
        p = path.getmtime
        e = path.exists
        mp = path.abspath('./last.update')
        ac = path.abspath('./all.cheese')
        while charting:
            mouse_poll = int(p(mp))
            if mouse_poll > last_poll:
                print('Cartographer: starting work.')
                try:
                    with open(ac, 'rb') as pkl:
                        mice = pickle.load(pkl)
                    c = mice.signals
                    t = time()
                    for symbol in ivy_ndx:
                        sym = str(symbol).upper()
                        cp = f'./charts/{sym}.png'
                        cdls = get_candles(symbol)
                        cartography(sym, cdls[-480:], cheese=c, chart_path=cp)
                finally:
                    e = time() - t
                    last_poll = mouse_poll
                    with open('./chart.done', 'w') as f:
                        f.write('yigyig')
                    print(f'Cartographer: finished work in {e} seconds.')
                print(f'Cartographer: going to sleep.')
            sleep(0.5)
    except KeyboardInterrupt:
        print('Keyboard Interrupt: Stopping loop.')
        charting = False
    finally:
        print(f'Cartographer retired.')

