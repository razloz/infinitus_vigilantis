"""Updater for Infinitus Vigilantis"""

import time
import pandas
import numpy
import pickle
import source.ivy_commons as icy
import source.ivy_alpaca as api
from os import path

SILENCE = icy.silence
DIV = icy.safe_div
PERCENT = icy.percent_change


def daemonize():
    """Keep spinning in the background while GUI is running."""
    return icy.ivy_dispatcher(spin_wheel, ftype='process')


def composite_index(ndx_path, ci_path='./indexes/composite.index'):
    """Convert space separated list to csv."""
    if path.exists(ci_path):
        ivy_ndx = pandas.read_csv(ci_path, index_col=0)
    else:
        tk = icy.TimeKeeper()
        with open(ndx_path, 'r') as symbols:
            ndx = symbols.read()
        valid_symbols = list()
        shp = api.AlpacaShepherd()
        for asset in shp.assets():
            conditions = [
                asset['status'] == 'active',
                asset['tradable'] == True,
                asset['exchange'] == 'NYSE' or 'NASDAQ'
                ] # conditions
            if all(conditions):
                valid_symbols.append(str(asset['symbol']))
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
        self.api = api.AlpacaShepherd()
        tz = 'America/New_York'
        self._tz = tz
        p = lambda c: pandas.to_datetime(c, utc=True).tz_convert(tz)
        self._CSV_ARGS = dict(
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            date_parser=p
            ) # self._CSV_ARGS
        self._COL_NAMES = {
            't': 'utc_ts',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
            } # self._COL_NAMES
        if verbose:
            print(f'Candelabrum initialized.')

    @SILENCE
    def __get_path__(self, symbol):
        """You no take candle!"""
        return f'{self._PATH}/{symbol.upper()}'

    @SILENCE
    def load_candles(self, symbol):
        """=^.^="""
        _pth = self.__get_path__(symbol)
        with open(f'{_pth}.ivy') as pth:
            _df = pandas.read_csv(pth, **self._CSV_ARGS)
        return _df.copy()

    @SILENCE
    def save_candles(self, symbol, dataframe):
        """=^.^="""
        _pth = self.__get_path__(symbol)
        dataframe.to_csv(f'{_pth}.ivy')
        return True

    @SILENCE
    def update_candles(self, symbol, new_data, local_data=None):
        """Combine old data with new data."""
        pth = self.__get_path__(symbol)
        do_merge = path.exists(f'{pth}.ivy')
        if do_merge:
            if not local_data:
                local_data = self.load_candles(symbol)
            update = local_data.combine_first(new_data)
            self.save_candles(symbol, update.copy())
        else:
            self.save_candles(symbol, new_data.copy())

    @SILENCE
    def do_update(self, symbols, limit=None):
        """Update historical data from index."""
        global icy
        verbose = (True if self._VERBOSE else False)
        tk = self._TIMER
        start_time = tk.reset
        tz = self._tz
        pts = pandas.Timestamp
        if verbose:
            self._VERBOSE = not self._VERBOSE
            print(f'Starting update loop for {len(symbols)} symbols...')
        candles = self.api.candles(symbols, limit=limit)
        for symbol in candles.keys():
            cdls = pandas.DataFrame(candles[symbol])
            cdls['time'] = [pts(t, unit='s', tz=tz) for t in cdls['t']]
            cdls.set_index('time', inplace=True)
            cdls.rename(columns=self._COL_NAMES, inplace=True)
            money = icy.get_money(cdls['close'].tolist())
            zs, sdev, wema, dh, dl, mid = zip(*money)
            cdls['money_zscore'] = zs
            cdls['money_sdev'] = sdev
            cdls['money_wema'] = wema
            cdls['money_dh'] = dh
            cdls['money_dl'] = dl
            cdls['money_mid'] = mid
            vm = icy.get_money(cdls['volume'].tolist())
            zs_v, sdev_v, wema_v, dh_v, dl_v, mid_v = zip(*vm)
            cdls['volume_zscore'] = zs_v
            cdls['volume_sdev'] = sdev_v
            cdls['volume_wema'] = wema_v
            cdls['volume_dh'] = dh_v
            cdls['volume_dl'] = dl_v
            cdls['volume_mid'] = mid_v
            cdls = self.update_candles(symbol, cdls)
        if verbose:
            print(f'Finished update in {tk.final} seconds.')
            self._VERBOSE = not self._VERBOSE

    @SILENCE
    def analyze(self, ivy):
        """Return historical data with technical indicators."""
        ivy['trend'] = icy.get_trend(ivy['high'].tolist(), ivy['low'].tolist())
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
        return ivy.copy()


def cheese_wheel(silent=True, max_days=34, do_update=True):
    """Update and test historical data."""
    global icy
    icy.SILENT = silent
    cdlm = Candelabrum()
    get_candles = cdlm.load_candles
    ivy_ndx = composite_index('./indexes/custom.ndx')
    mice = icy.ThreeBlindMice(ivy_ndx, max_days=max_days)
    make_cheese = mice.get_cheese
    api_clock = cdlm.api.clock()
    status = bool(api_clock['is_open'])
    if not silent:
        print(f'Market Status: {status}')
    cdlm.do_update(ivy_ndx, limit=1000)
    if not silent:
        l = len(ivy_ndx)
        print(f'Starting quest for the ALL CHEESE using {l} symbols.')
    tk = icy.TimeKeeper()
    for symbol in ivy_ndx:
        try:
            make_cheese(symbol, get_candles(symbol))
        finally:
            pass
    mice.validate_trades()
    signals = mice.signals
    if not silent:
        if not len(signals) > 0:
            print('No signals! No Stats!')
            return (status, mice)
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
    return (api_clock, mice)


def make_utc(time_string):
    """Liberation from the chains of Daylight Savings."""
    fmt = '%Y-%m-%dT%H:%M:%S%z'
    first_annoyance = time.strptime(time_string, fmt)
    second_annoyance = time.mktime(first_annoyance)
    third_annoyance = time.gmtime(second_annoyance)
    return time.mktime(third_annoyance)


def spin_wheel(daemonized=True):
    """Spin to win."""
    get_schedule = icy.UpdateSchedule
    spinning = True
    sleep_until = 0
    sleep_date = ''
    total_spins = 0
    today = ''
    schedule = list()
    keeper = icy.TimeKeeper()
    try:
        print('Spinning the cheese wheel...')
        while spinning:
            utc_tup = time.gmtime()
            utc_ts = time.mktime(utc_tup)
            utc_now = time.strftime('%Y-%m-%d %H:%M:%S', utc_tup)
            check_day = utc_now.split(' ')[0]
            if check_day != today:
                today = check_day
                schedule = list(get_schedule(today, freq='5min'))
            if utc_now in schedule:
                total_spins += 1
                print(f'Spin: {total_spins}')
                s, mice = cheese_wheel(max_days=89)
                status = bool(s['is_open'])
                if mice:
                    with open('./all.cheese', 'wb') as f:
                        pickle.dump(mice, f, pickle.HIGHEST_PROTOCOL)
                    with open('./last.update', 'w') as f:
                        f.write('spin-to-win')
                    print('Going to sleep until next scheduled spin.')
                if status is False:
                    if not daemonized:
                        spin = input('Keep spinning? [y/N]: ')
                        if spin.lower() != 'y':
                            spinning = False
                        else:
                            print('Spin! Spin! Spin!')
                    else:
                        sleep_date = str(s['next_open'])
                        sleep_until = make_utc(sleep_date)
            if spinning:
                if utc_ts >= sleep_until:
                    time.sleep(1)
                else:
                    u = sleep_until - utc_ts
                    print(f'Next spin scheduled for {sleep_date} in {u} second(s).')
                    time.sleep(u)
    except KeyboardInterrupt:
        print('Keyboard Interrupt: Stopping loop.')
        looping = False
    finally:
        e = keeper.update[0]
        print(f'Stopped spinning after {e} with {total_spins} spins.')

