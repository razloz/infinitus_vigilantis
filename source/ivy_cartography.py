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
from numpy import arange
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
plt.style.use('dark_background')
verbose = False


def cartography(symbol, dataframe, chart_path=None, cheese=None,
                chart_size=0, adj='1D'):
    """Charting for IVy candles."""
    global plt
    plt.close('all')
    if not chart_path: chart_path = './charts/active.png'
    if verbose: print(f'Cartography: creating chart for {symbol}...')
    cdl_open = dataframe['open'].tolist()
    cdl_high = dataframe['high'].tolist()
    cdl_low = dataframe['low'].tolist()
    cdl_close = dataframe['close'].tolist()
    cdl_vol = dataframe['volume'].tolist()
    cdl_wema = dataframe['price_wema'].tolist()
    cdl_mid = dataframe['price_mid'].tolist()
    cdl_dh = dataframe['price_dh'].tolist()
    cdl_dl = dataframe['price_dl'].tolist()
    vol_wema = dataframe['volume_wema'].tolist()
    vol_mid = dataframe['volume_mid'].tolist()
    vol_dh = dataframe['volume_dh'].tolist()
    moirai_metrics = ''
    if cheese:
        metrics = ['num_epochs', 'threshold', 'accuracy', 'mae', 'mse',
                   'loss', 'batch_size', 'batch_pred', 'proj_gain',
                   'proj_time_str']
        for metric_label in metrics:
            m_value = cheese[metric_label]
            addendum = f'{metric_label}: {m_value}'
            if metric_label in ['accuracy', 'proj_gain']:
                addendum += '%'
            add_len = len(addendum)
            if add_len < 80:
                for _ in range(80 - add_len - 1):
                    addendum += ' '
            moirai_metrics += f'{addendum}\n'
        moirai_metrics = moirai_metrics[:-2]
        batch_size = cheese['batch_size']
        pad = [0 for _ in range(batch_size)]
        predictions = pad + cheese['prediction']
        targets = dataframe['price_med'].tolist()
    ts_lbls = [x_ts for x_ts in dataframe.index.tolist()]
    if chart_size > 0:
        ts_lbls = ts_lbls[-chart_size:]
        cdl_open = cdl_open[-chart_size:]
        cdl_high = cdl_high[-chart_size:]
        cdl_low = cdl_low[-chart_size:]
        cdl_close = cdl_close[-chart_size:]
        cdl_vol = cdl_vol[-chart_size:]
        cdl_wema = cdl_wema[-chart_size:]
        cdl_mid = cdl_mid[-chart_size:]
        cdl_dh = cdl_dh[-chart_size:]
        cdl_dl = cdl_dl[-chart_size:]
        vol_wema = vol_wema[-chart_size:]
        vol_mid = vol_mid[-chart_size:]
        vol_dh = vol_dh[-chart_size:]
        if cheese:
            targets = targets [-chart_size:]
            chart_size += batch_size
            cheese_range = range(chart_size)
            predictions = predictions[-chart_size:]
            ts_lbls += [f'cheese_{n_pred + 1}' for n_pred in range(batch_size)]
    data_len = len(cdl_close)
    data_range = range(data_len)
    fs = (19.20, 10.80)
    dpi = 100
    fig = plt.figure(figsize=fs, dpi=dpi, constrained_layout=False)
    sargs = dict(ncols=1, nrows=2, figure=fig, height_ratios=[4,1])
    spec = gridspec.GridSpec(**sargs)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = fig.add_subplot(spec[1, 0], sharex=ax1)
    if cheese:
        plt.xticks(ticks=cheese_range, labels=ts_lbls,
                   rotation=21, fontweight='bold')
    else:
        plt.xticks(ticks=data_range, labels=ts_lbls,
                   rotation=21, fontweight='bold')
    plt.subplots_adjust(left=0.08, bottom=0.28, right=0.92,
                        top=0.95, wspace=0, hspace=0.02)
    ax1.grid(True, color=(0.3, 0.3, 0.3))
    ax1.set_ylabel('Price', fontweight='bold')
    if cheese:
        ax1.set_xlim(((cheese_range[0] - 2), (cheese_range[-1] + 2)))
    else:
        ax1.set_xlim(((data_range[0] - 2), (data_range[-1] + 2)))
    ylim_low = min(cdl_low)
    ylim_high = max(cdl_high)
    ax1.set_ylim((ylim_low * 0.97, ylim_high * 1.03))
    yticks_range = [round(i, 2) for i in arange(ylim_low, ylim_high, 0.01)]
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
    wid_base = (xb / data_len) * 0.5
    wid_wick = wid_base * 0.21
    wid_cdls = wid_base * 0.89
    wid_line = wid_base * 0.34
    for i in data_range:
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
    if cheese:
        pkws['color'] = '#FFE88E' # gouda
        pkws['label'] = f'Prediction:'
        pkws['label'] += f' {round(predictions[-1], 2)}'
        ax1.plot(cheese_range, predictions, alpha=0.99, **pkws)
        pkws['color'] = '#FF9600' # cheddar
        pkws['label'] = f'Target:'
        pkws['label'] += f' {round(targets[-1], 2)}'
        ax1.plot(data_range, targets, alpha=0.99, **pkws)
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
    props = dict(boxstyle='round', facecolor='0.03', alpha=0.97)
    plt.gcf().text(0.01, 0.01, moirai_metrics, fontsize=14, bbox=props)
    last_ts = batch_size + 1
    ts = ts_lbls[-last_ts]
    res = adj if adj else 'None'
    rnc = round(cdl_close[-1], 3)
    t = f'[ {rnc} ]   {symbol}  @  {ts} (resample: {res}'
    if cheese:
        t += f', batch_size: {batch_size}'
    t += ')'
    fig.suptitle(t, fontsize=18)
    fig.legend(ncol=2, loc='lower right', fontsize='xx-large', fancybox=True)
    plt.savefig(str(chart_path), dpi=dpi)
    plt.close(fig)
    if verbose: print("Cartography: chart's done!")
    return False


def scaled_chart(symbol, chart_size, scale, signals,
                 candelabrum, start_date, end_date):
    """Omenize and resample data for chart generation."""
    sym = str(symbol).upper()
    cp = f'./charts/{sym}.png'
    s = chart_size if isinstance(chart_size, int) else 100
    cs = s * -1
    get_candles = candelabrum.gather_data
    resample = candelabrum.resample_candles
    omenize = candelabrum.apply_indicators
    cdls = get_candles(symbol, start_date, end_date)
    if scale: cdls = resample(cdls, scale)
    pivots = PivotPoints(
        cdls['open'].tolist(),
        cdls['high'].tolist(),
        cdls['low'].tolist(),
        cdls['close'].tolist())
    cdls = omenize(cdls)
    if len(cdls) > 0:
        if len(cdls) > s:
            scaled_cdls = cdls[cs:]
        else:
            scaled_cdls = cdls[:]
        kargs = dict(pivot_points=pivots, cheese=signals, chart_path=cp)
        if scale: kargs['adj'] = scale
        cartography(sym, scaled_cdls, **kargs)
        with open(f'./configs/{sym}.done', 'w') as f:
            f.write('yigyig')


def cartographer(symbol=None, chart_size=100, adj_time=None,
                 daemon=False, no_signals=False,
                 start_date=None, end_date=None):
    """Charting daemon."""
    from source.ivy_commons import ivy_dispatcher
    from source.ivy_candles import composite_index
    from source.ivy_candles import Candelabrum
    do_once = isinstance(symbol, str)
    valid_times = ('5Min', '10Min', '15Min', '30Min', '1H', '3H')
    if adj_time:
        adj = adj_time if adj_time in valid_times else None
        if not adj:
            if verbose: print(f'Error: adj_time must be one of {valid_times}')
    else:
        adj = None
    cdlm = Candelabrum()
    if not do_once:
        ivy_ndx = composite_index('./indexes/default.ndx')
        if verbose: print(f'Cartographer: working on {len(ivy_ndx)} symbols.')
    else:
        mk_msg = 'Cartographer: creating a {} width {} chart for {}.'
        if verbose: print(mk_msg.format(chart_size, adj, symbol))
    charting = True
    last_poll = 0
    try:
        mp = path.abspath('./configs/last.update')
        ac = path.abspath('./configs/all.cheese')
        while charting:
            mouse_poll = path.getmtime(mp) if path.exists(mp) else 0
            if mouse_poll > last_poll or no_signals:
                if verbose: print('Cartographer: starting work.')
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
                        for symbol_pair in ivy_ndx:
                            scaled_chart(symbol_pair[0], *a)
                    else:
                        scaled_chart(symbol, *a)
                finally:
                    e = time() - t
                    last_poll = mouse_poll
                    if verbose: print(f'Cartographer: finished work in {e} seconds.')
                if daemon:
                    if verbose: print('Cartographer: going to sleep.')
            if not daemon:
                charting = False
            else:
                sleep(0.5)
    except KeyboardInterrupt:
        print('Keyboard Interrupt: Stopping loop.')
        charting = False
    finally:
        if verbose: print('Cartographer retired.')
