#!./.env/bin/python3
"""Updater for Infinitus Vigilantis"""

import time
import pandas
import numpy
import pickle
import ivy_commons as icy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from os import path
from alpaca_trade_api import REST


__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2020, Daniel Ward'
__license__ = 'MIT'
__version__ = '2020.01'
__codename__ = 'cria'

plt.style.use('dark_background')
API = REST()
SILENCE = icy.silence
DIV = icy.safe_div
PERCENT = icy.percent_change


def composite_index(ndx_path, ci_path='./indexes/composite.index'):
    """Convert space separated list to csv."""
    if path.exists(ci_path):
        ivy_ndx = pandas.read_csv(ci_path, index_col=0)
    else:
        tk = icy.TimeKeeper()
        with open(ndx_path, 'r') as symbols:
            ndx = symbols.read()
        valid_symbols = list()
        for asset in API.list_assets():
            conditions = [
                asset.status == 'active',
                asset.tradable == True,
                asset.exchange == 'NYSE' or 'NASDAQ'
                ] # conditions
            if all(conditions):
                valid_symbols.append(asset.symbol)
        print(f'Valid Assets: {len(valid_symbols)}')
        ivy_ndx = pandas.DataFrame()
        syms = list({s for s in str(ndx).split() if s in valid_symbols})
        ivy_ndx['symbols'] = syms
        ivy_ndx.to_csv(ci_path)
        fin = tk.final
        if fin < 1: time.sleep(1 - fin)
    return ivy_ndx['symbols'].tolist()


class Spanner:
    """Span tracking for zscore array."""
    span = 0
    def expand(self, n):
        """Track span and reset if direction changes."""
        if n > 0 and self.span < 0: self.span = 0
        if n < 0 and self.span > 0: self.span = 0
        self.span += 1
        return self.span


def thunderstruck(s, a, z):
    """Thunderstruck Rating System."""
    t = list()
    for i in range(len(s)):
        span = s[i]
        savg = a[i]
        zscore = z[i]
        r = 0
        if span > savg:
            r = span - savg
        elif span < savg:
            r = savg - span
        rating = PERCENT((savg - r), savg) * -1
        t.append(rating)
    return t


class Candelabrum:
    """Handler for historical price data."""
    def __init__(self, data_path='./candelabrum', verbose=True):
        self._PATH = data_path
        self._VERBOSE = verbose
        icy.SILENT = not verbose
        self._TIMER = icy.TimeKeeper()
        self._SPANNER = Spanner()
        _ptd = pandas.to_datetime
        _tz = 'America/New_York'
        self._CSV_ARGS = dict(
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            date_parser=lambda col: _ptd(col, utc=True).tz_convert(_tz)
            ) # self._CSV_ARGS
        _date_args = dict(
            start='2017-1-1 00:00:00',
            end=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
            tz=_tz,
            freq='15min',
            name='time'
            ) # _date_args
        self._TEMPLATE = pandas.DataFrame(
            index=pandas.date_range(**_date_args),
            columns=['open', 'high', 'low', 'close', 'volume']
            ) # self._TEMPLATE
        if verbose:
            print(f'Candelabrum initialized.')

    @SILENCE
    def __get_path__(self, symbol, timeframe):
        """You no take candle!"""
        _pth = f'{self._PATH}/{symbol.lower()}_{timeframe.upper()}'
        return _pth

    @SILENCE
    def load_candles(self, symbol, timeframe):
        """=^.^="""
        _pth = self.__get_path__(symbol, timeframe)
        with open(f'{_pth}.ivy') as pth:
            _df = pandas.read_csv(pth, **self._CSV_ARGS)
        return _df.copy()

    @SILENCE
    def save_candles(self, symbol, timeframe, dataframe):
        """=^.^="""
        _pth = self.__get_path__(symbol, timeframe)
        dataframe.to_csv(f'{_pth}.ivy')
        return True

    @SILENCE
    def alpaca_candles(self, symbol, timeframe, after=None, limit=1000):
        """Get all candles for timeframe nyaaa~~~"""
        global icy
        valid_times = ['1Min', '5Min', '15Min', '1D']
        if timeframe not in valid_times:
            if self._VERBOSE:
                print(f'*** timeframe must be one of {valid_times}.')
            return None
        ivy = self._TEMPLATE.copy()
        if self._VERBOSE: self._TIMER.reset
        barset = API.get_barset(symbol, timeframe, after=after, limit=limit)
        ivy.update(barset[symbol].df)
        ivy.dropna(inplace=True)
        if ivy.empty: return None
        if len(ivy.index) < 56: return None
        money = icy.get_money(ivy['close'].tolist())
        zs, sdev, wema, dh, dl, mid = zip(*money)
        ivy['zscore'] = zs
        ivy['sdev'] = sdev
        ivy['wema'] = wema
        ivy['dh'] = dh
        ivy['dl'] = dl
        ivy['mid'] = mid
        ivy['trend'] = icy.get_trend(ivy['high'].tolist(), ivy['low'].tolist())
        if self._VERBOSE:
            e = self._TIMER.update[0]
            print(f'Alpaca candles for {symbol} took {e} to fetch.')
        return ivy.copy()

    @SILENCE
    def update_candles(self, symbol, timeframe, new_data, local_data=None):
        """Combine old data with new data."""
        if local_data is None:
            _old = self.load_candles(symbol, timeframe)
        else:
            _old = local_data
        update = _old.combine_first(new_data)
        return update.copy()

    @SILENCE
    def do_update(self, symbols, timeframe):
        """Update historical data from index."""
        verbose = (True if self._VERBOSE else False)
        tk = self._TIMER
        start_time = tk.reset
        if verbose:
            self._VERBOSE = not self._VERBOSE
            print(f'Starting update loop for {len(symbols)} symbols...')
            blank = ''.join(' ' for i in range(21))
            sym_count = 0
            sym_max = len(symbols)
        for symbol in symbols:
            if verbose:
                sym_count += 1
                prefix = f'({sym_count}/{sym_max}) {symbol}:'
                print(f'{prefix} updating...{blank}', end=chr(13))
            after = '2017-01-01T00:00:00-04:00'
            merge = []
            pth = self.__get_path__(symbol, timeframe)
            do_merge = path.exists(f'{pth}.ivy')
            if do_merge:
                merge = self.load_candles(symbol, timeframe)
                after = str(merge.tail(1).index)
                if ' ' in after:
                    after = after.split(' ')[0]
                elif 'T' in after:
                    after = after.split('T')[0]
                after += 'T00:00:00-04:00'
            if len(merge) == 0: merge = None
            since, elapsed = tk.update
            ivy = self.alpaca_candles(symbol, timeframe, after=after)
            if ivy is None:
                if verbose:
                    msg = f'{prefix} alpaca_candles returned None {blank}'
                    print(msg, end=chr(13))
            else:
                if do_merge:
                    uc = self.update_candles
                    ivy = uc(symbol, timeframe, ivy, local_data=merge)
                ivy['strength'] = icy.trend_line(ivy['trend'].tolist())
                self._SPANNER.span = 0
                ivy['span'] = ivy['zscore'].apply(self._SPANNER.expand)
                ivy['avg_span'] = ivy['span'].expanding().mean()
                targs = (
                    ivy['span'].tolist(),
                    ivy['avg_span'].tolist(),
                    ivy['zscore'].tolist()
                    ) # targs
                ivy['thunderstruck'] = thunderstruck(*targs)
                self.save_candles(symbol, timeframe, ivy)
            elapsed = tk.update[1]
            if elapsed < 0.34:
                waiting = 0.34 - elapsed
                if verbose:
                    print(f'{prefix} waiting {waiting}{blank}', end=chr(13))
                time.sleep(waiting)
        if verbose:
            print(f'Finished update in {tk.final} seconds.')
            self._VERBOSE = not self._VERBOSE

    @SILENCE
    def cartography(self, symbol, timeframe, dataframe, status,
                    cheese=None, chart_path='./charts/active.png'):
        """Charting for IVy candles."""
        global plt
        plt.close('all')
        timestamps = dataframe.index.tolist()
        data_range = range(len(timestamps))
        cdl_open = dataframe['open'].tolist()
        cdl_high = dataframe['high'].tolist()
        cdl_low = dataframe['low'].tolist()
        cdl_close = dataframe['close'].tolist()
        cdl_vol = dataframe['volume'].tolist()
        cdl_wema = dataframe['wema'].tolist()
        cdl_mid = dataframe['mid'].tolist()
        cdl_zs = dataframe['zscore'].tolist()
        cdl_dh = dataframe['dh'].tolist()
        cdl_dl = dataframe['dl'].tolist()
        cdl_trend = dataframe['trend'].tolist()
        fig = plt.figure(figsize=(13, 8), constrained_layout=False)
        sargs = dict(ncols=1, nrows=3, figure=fig, height_ratios=[5,1,1])
        spec = gridspec.GridSpec(**sargs)
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[1, 0], sharex=ax1)
        ax3 = fig.add_subplot(spec[2, 0], sharex=ax1)
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
        ax3.grid(True, color=(0.4, 0.4, 0.4))
        ax3.set_ylabel('Trend:ZScore', fontweight='bold')
        plt.setp(ax1.xaxis.get_ticklabels()[:], visible=False)
        plt.setp(ax2.xaxis.get_ticklabels()[:], visible=False)

        pkws = {'linestyle': 'solid', 'linewidth': 2}
        pkws['label'] = f'ZScore: {round(cdl_zs[-1], 2)}'
        pkws['color'] = (0.8, 0.8, 0.8, 1)
        ax3.plot(data_range, cdl_zs, **pkws)

        pkws['label'] = f'Trend: {cdl_trend[-1]}'
        pkws['color'] = (0.3, 0.6, 0.3, 1)
        ax3.plot(data_range, cdl_trend, **pkws)

        pkws['label'] = f'Money: {round(cdl_wema[-1], 2)}'
        pkws['color'] = (0.1, 0.5, 0.1, 1)
        ax1.plot(data_range, cdl_wema, **pkws)

        pkws['label'] = f'Mid: {round(cdl_mid[-1], 2)}'
        pkws['color'] = (0.8, 0.8, 1, 1)
        ax1.plot(data_range, cdl_mid, **pkws)

        pkws['linestyle'] = 'dotted'
        pkws['linewidth'] = 1.5
        pkws['label'] = f'DevHigh: {round(cdl_dh[-1], 2)}'
        ax1.plot(data_range, cdl_dh, **pkws)

        pkws['label'] = f'DevLow: {round(cdl_dl[-1], 2)}'
        ax1.plot(data_range, cdl_dl, **pkws)

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
        ts = timestamps[-1].strftime('%Y-%m-%d %H:%M')
        t = f'[ {symbol} @ {timeframe} ] [ {ts} ]'
        t += f' [ Market Open: {status} ] Close: {cdl_close[-1]}'
        fig.legend(title=t, ncol=6, loc='upper center', fontsize='large')
        plt.savefig(str(chart_path))
        plt.close(fig)


def cheese_wheel(silent=False, chart_size=0, timeframe='1D',
                 max_days=34, do_update=True):
    """Update and test historical data."""
    global icy
    icy.SILENT = silent
    cdlm = Candelabrum()
    get_candles = cdlm.load_candles
    ivy_ndx = composite_index('./indexes/custom.ndx')
    mice = icy.ThreeBlindMice(ivy_ndx, max_days=max_days)
    make_cheese = mice.get_cheese
    mstat = API.get_clock()
    status = (mstat.is_open, mstat.next_open, mstat.next_close)
    if not silent:
        print(f'Market Status: {status[0]}')
    if do_update:
        passed_check = False
        if timeframe == '1D':
            if bool(status[0]) is False:
                passed_check = True
        else:
            passed_check = True
        if passed_check:
            if not silent:
                print('Passed check, doing update...')
            cdlm.do_update(ivy_ndx, timeframe)
        elif not silent:
            print(f'Check failed, skipping update...(timeframe: {timeframe})')
    if not silent:
        l = len(ivy_ndx)
        print(f'Starting quest for the ALL CHEESE using {l} symbols.')
    tk = icy.TimeKeeper()
    for symbol in ivy_ndx:
        try:
            make_cheese(symbol, get_candles(symbol, timeframe))
        finally:
            pass
    mice.validate_trades()
    signals = mice.signals
    if chart_size > 0:
        for symbol in ivy_ndx:
            try:
                cdls = get_candles(symbol, timeframe)
                cl = len(cdls) - chart_size - 1
                cp = f'./charts/{symbol.lower()}_{timeframe.upper()}.png'
                cdlm.cartography(symbol, timeframe, cdls[cl:], status[0],
                                 chart_path=cp, cheese=signals)
            finally:
                pass
    if not silent:
        positions = mice.positions
        sig_ts = list(signals)[-1]
        buy = signals[sig_ts]['buy']
        sell = signals[sig_ts]['sell']
        print('\nStats:')
        percentiles = ['roi', 'benchmark', 'net change']
        for k, v in mice.stats.items():
            print(f'    {k}: {v}' if k not in percentiles else f'    {k}: {v}%')
        print('\nPositions:')
        for sym in positions:
            print(f'    {sym}: {mice.positions[sym]}')
        print(f'\nSell signals for {sig_ts}:\n    {sell}')
        print(f'\nBuy signals for {sig_ts}:\n    {buy}')
        e = tk.update[0]
        print(f'\nAfter {e} the quest for the ALL CHEESE comes to an end.\n')
    return (status, mice)


def spin_wheel():
    """Spin to win."""
    doc_info = '\nLicense: {0}\n{1}\n{2}\nVersion: {3} ({4})\n'
    print(doc_info.format(__license__, __copyright__, __doc__,
                          __version__, __codename__))
    get_schedule = icy.UpdateSchedule
    spinning = True
    total_spins = 0
    today = ''
    schedule = list()
    keeper = icy.TimeKeeper()
    try:
        print('Spinning the cheese wheel...')
        while spinning:
            utc_now = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
            check_day = utc_now.split(' ')[0]
            if check_day != today:
                today = check_day
                schedule = list(get_schedule(today, freq='5min'))
            if utc_now in schedule:
                total_spins += 1
                print(f'Spin: {total_spins}')
                s, mice = cheese_wheel(max_days=89, chart_size=100)
                status = s[0]
                if mice:
                    with open('./all.cheese', 'wb') as f:
                        pickle.dump(mice, f, pickle.HIGHEST_PROTOCOL)
                    with open('./last.update', 'w') as f:
                        f.write('spin-to-win')
                    print('Going to sleep until next scheduled spin.')
                if not status:
                    spin = input('Keep spinning? [y/N]: ')
                    if spin.lower() != 'y':
                        spinning = False
                    else:
                        print('Spin! Spin! Spin!')
            if spinning:
                time.sleep(1)
    except KeyboardInterrupt:
        print('Keyboard Interrupt: Stopping loop.')
        looping = False
    finally:
        e = keeper.update[0]
        print(f'Stopped spinning after {e} with {total_spins} spins.')


if __name__ == '__main__':
    spin_wheel()
