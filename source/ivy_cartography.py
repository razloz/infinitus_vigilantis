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
    plt.xticks(ticks=data_range, labels=ts_lbls, rotation=21, fontweight='bold')
    plt.subplots_adjust(left=0.08, bottom=0.08, right=0.77,
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
    props = dict(boxstyle='round', facecolor='0.03', alpha=0.97)
    plt.gcf().text(0.79, 0.77, moirai_metrics, fontsize=14, bbox=props)
    res = adj if adj else 'None'
    rnc = round(cdl_close[-1], 3)
    t = f'[ {rnc} ]   {symbol}  @  {ts_last} (resample: {res})'
    fig.suptitle(t, fontsize=18)
    fig.legend(ncol=1, loc='lower right', fontsize='xx-large', fancybox=True)
    plt.savefig(str(chart_path))
    plt.clf()
    plt.close()
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
