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
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2020, Daniel Ward'
__license__ = 'GPL v3'
__version__ = '2020.04'
__codename__ = 'compass'


def cartography(symbol, dataframe, cheese=None, adj=None,
                chart_path='./charts/active.png', *ignore):
    """Charting for IVy candles."""
    global plt
    plt.close('all')
    print(f'Cartography: creating chart for {symbol}...')
    timestamps = dataframe.index.tolist()
    data_len = len(timestamps)
    data_range = range(data_len)
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
    fs = (19.20, 10.80)
    dpi = 100
    fig = plt.figure(figsize=fs, dpi=dpi, constrained_layout=False)
    sargs = dict(ncols=1, nrows=2, figure=fig, height_ratios=[4,1])
    spec = gridspec.GridSpec(**sargs)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0], sharex=ax1)
    plt.xticks(data_range, timestamps, rotation=21, fontweight='bold')
    plt.subplots_adjust(left=0.08, bottom=0.20, right=0.92,
                        top=0.95, wspace=0, hspace=0.08)
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
    xticks = ax1.xaxis.get_ticklabels()
    plt.setp(xticks[:], visible=False)
    # Dynamic width stuff
    tbb = ax1.get_tightbbox(fig.canvas.get_renderer()).get_points()
    xb = tbb[1][0] - tbb[0][0]
    wid_base = (xb / data_len) * 0.5
    wid_wick = wid_base * 0.21
    wid_cdls = wid_base * 0.89
    wid_line = wid_base * 0.34
    # Per candle plots
    signal_y = [min(cdl_dl), max(cdl_dh)]
    for i in data_range:
        x_loc = [i, i]
        # Signals
        if cheese:
            cdl_date = timestamps[i].strftime('%Y-%m-%d %H:%M')
            lw = 1 + data_range[-1]
            sig_args = dict(linestyle='solid', linewidth=wid_wick)
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
    pkws['label'] = f'Money: {round(cdl_wema[-1], 2)}'
    pkws['color'] = (0.4, 0.7, 0.4, 0.8)
    ax1.plot(data_range, cdl_wema, **pkws)
    pkws['label'] = None
    ax2.plot(data_range, vol_wema, **pkws)
    pkws['label'] = f'Mid: {round(cdl_mid[-1], 2)}'
    pkws['color'] = (0.7, 0.7, 1, 0.7)
    ax1.plot(data_range, cdl_mid, **pkws)
    pkws['label'] = None
    ax2.plot(data_range, vol_mid, **pkws)
    pkws['linestyle'] = 'dotted'
    pkws['linewidth'] = wid_line * 0.67
    pkws['label'] = f'DevHigh: {round(cdl_dh[-1], 2)}'
    ax1.plot(data_range, cdl_dh, **pkws)
    pkws['label'] = None
    ax2.plot(data_range, vol_dh, **pkws)
    pkws['label'] = f'DevLow: {round(cdl_dl[-1], 2)}'
    ax1.plot(data_range, cdl_dl, **pkws)
    # Finalize
    ts = timestamps[-1].strftime('%Y-%m-%d %H:%M')
    res = adj if adj else 'None'
    rnc = round(cdl_close[-1], 3)
    t = f'[ {rnc} ]   {symbol}  @  {ts} (resample: {res})'
    fig.suptitle(t, fontsize=18)
    fig.legend(ncol=4, loc='lower center', fontsize='xx-large', fancybox=True)
    plt.savefig(str(chart_path), dpi=dpi)
    plt.close(fig)
    print("Cartography: chart's done!")
    return False


def scaled_chart(symbol, chart_size, scale, signals,
                 candelabrum, start_date, end_date):
    """Omenize and resample data for chart generation."""
    sym = str(symbol).upper()
    cp = f'./charts/{sym}.png'
    s = chart_size if isinstance(chart_size, int) else 100
    cs = s * -1
    get_candles = candelabrum.load_candles
    omenize = candelabrum.apply_indicators
    cdls = omenize(symbol, start_date, end_date)
    if scale: cdls = cdls.resample(scale).mean().copy()
    cdls.dropna(inplace=True)
    if len(cdls) > 0:
        if len(cdls) > s:
            scaled_cdls = cdls[cs:]
        else:
            scaled_cdls = cdls[:]
        kargs = dict(cheese=signals, chart_path=cp)
        if scale: kargs['adj'] = scale
        cartography(sym, scaled_cdls, **kargs)
        with open(f'./configs/{sym}.done', 'w') as f:
            f.write('yigyig')


def cartographer(symbol=None, chart_size=100, adj_time=None,
                 daemon=False, no_signals=False,
                 start_date=None, end_date=None):
    """Charting daemon."""
    do_once = isinstance(symbol, str)
    valid_times = ('5Min', '10Min', '15Min', '30Min', '1H', '3H')
    if adj_time:
        adj = adj_time if adj_time in valid_times else None
        if not adj:
            print(f'Error: adj_time must be one of {valid_times}')
    else:
        adj = None
    cdlm = Candelabrum()
    if not do_once:
        ivy_ndx = composite_index('./indexes/custom.ndx')
        print(f'Cartographer: working on {len(ivy_ndx)} symbols.')
    else:
        mk_msg = 'Cartographer: creating a {} width {} chart for {}.'
        print(mk_msg.format(chart_size, adj, symbol))
    charting = True
    last_poll = 0
    try:
        mp = path.abspath('./configs/last.update')
        ac = path.abspath('./configs/all.cheese')
        while charting:
            mouse_poll = path.getmtime(mp) if path.exists(mp) else 0
            if mouse_poll > last_poll or no_signals:
                print('Cartographer: starting work.')
                try:
                    if not no_signals:
                        with open(ac, 'rb') as pkl:
                            mice = pickle.load(pkl)
                        c = mice.signals
                    else:
                        c = None
                    t = time()
                    a = (chart_size, adj, c, cdlm, start_date, end_date)
                    if not do_once:
                        for symbol in ivy_ndx:
                            scaled_chart(symbol, *a)
                    else:
                        scaled_chart(symbol, *a)
                finally:
                    e = time() - t
                    last_poll = mouse_poll
                    print(f'Cartographer: finished work in {e} seconds.')
                if daemon:
                    print('Cartographer: going to sleep.')
            if not daemon:
                charting = False
            else:
                sleep(0.5)
    except KeyboardInterrupt:
        print('Keyboard Interrupt: Stopping loop.')
        charting = False
    finally:
        print('Cartographer retired.')
