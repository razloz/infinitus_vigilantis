"""Updater for Infinitus Vigilantis"""

import time
import pandas
import numpy
import pickle
import source.ivy_commons as icy
import source.ivy_alpaca as api
from os import path

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2021, Daniel Ward'
__license__ = 'GPL v3'
__version__ = '2021.05'
__codename__ = 'bling'

SILENCE = icy.silence
DIV = icy.safe_div
PERCENT = icy.percent_change


def daemonize():
    """Keep spinning in the background while GUI is running."""
    return icy.ivy_dispatcher(spin_wheel, ftype='process')

def market_calendar(cpath='./indexes/calendar.index'):
    """Fetch the market calendar and store it locally."""
    if path.exists(cpath):
        ivy_calendar = pandas.read_csv(cpath, index_col=0, dtype=str)
    else:
        shepherd = api.AlpacaShepherd()
        c = shepherd.calendar()
        ivy_calendar = pandas.DataFrame(c, dtype=str)
        ivy_calendar.set_index('date', inplace=True)
        ivy_calendar.to_csv(cpath)
    return ivy_calendar.copy()

def composite_index(ndx_path='./indexes/default.ndx',
                    ci_path='./indexes/composite.index'):
    """Convert space separated list to csv."""
    if path.exists(ci_path):
        ivy_ndx = pandas.read_csv(ci_path, index_col=0)
    else:
        tk = icy.TimeKeeper()
        with open(ndx_path, 'r') as symbols:
            ndx = symbols.read()
        valid_symbols = dict()
        shepherd = api.AlpacaShepherd()
        for asset in shepherd.assets():
            exchange = None
            symbol = str(asset['symbol'])
            if symbol in ('QQQ', 'SPY'):
                exchange = symbol
            else:
                if asset['exchange'] == 'NYSE': exchange = 'SPY'
                if asset['exchange'] == 'NASDAQ': exchange = 'QQQ'
            # asset['status'] == 'active'
            if exchange is not None:
                valid_symbols[symbol] = exchange
        print(f'Composite Index: {len(valid_symbols)} valid assets.')
        sym_list = str(ndx).split()
        syms = list()
        exch = list()
        for s, e in valid_symbols.items():
            if s in sym_list:
                syms.append(s)
                exch.append(e)
        ivy_ndx = pandas.DataFrame()
        ivy_ndx['symbols'] = syms
        ivy_ndx['exchanges'] = exch
        ivy_ndx.to_csv(ci_path)
        fin = tk.final
        if fin < 1: time.sleep(1 - fin)
    s = ivy_ndx['symbols'].tolist()
    e = ivy_ndx['exchanges'].tolist()
    if len(s) != len(e): return None
    return [(str(s[i]), str(e[i])) for i in range(len(s))]


class Candelabrum:
    """Handler for historical price data."""
    def __init__(self, data_path='./candelabrum', verbose=True,
                 BENCHMARKS=('QQQ', 'SPY')):
        self._BENCHMARKS = BENCHMARKS
        self._PATH = data_path
        self._VERBOSE = verbose
        icy.SILENT = not verbose
        self._TIMER = icy.TimeKeeper()
        self.api = api.AlpacaShepherd()
        self.benchmarks = dict()
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
            print('Candelabrum: initialized.')

    @SILENCE
    def __get_path__(self, symbol, date_string):
        """You no take candle!"""
        p = f'{self._PATH}/{symbol.upper()}-{date_string}.ivy'
        return path.abspath(p)

    @SILENCE
    def fill_template(self, dataframe):
        """Populate template and interpolate missing data."""
        date_args = dict(name='time')
        date_args['start'] = dataframe.index[0]
        date_args['end'] = dataframe.index[-1]
        date_args['tz'] = self._tz
        date_args['freq'] = '1min'
        c = ['utc_ts', 'open', 'high', 'low', 'close', 'volume']
        i = pandas.date_range(**date_args)
        template = pandas.DataFrame(index=i, columns=c)
        template.update(dataframe)
        template.fillna(method='ffill', inplace=True)
        template.fillna(method='bfill', inplace=True)
        template.dropna(inplace=True)
        template = template.transpose().copy()
        template.replace(to_replace=0, method='ffill', inplace=True)
        template = template.transpose().copy()
        template.dropna(inplace=True)
        return template.copy()

    @SILENCE
    def load_candles(self, symbol, date_string):
        """=^.^="""
        data_path = self.__get_path__(symbol, date_string)
        if not path.exists(data_path):
            return pandas.DataFrame()
        with open(data_path) as pth:
            df = pandas.read_csv(pth, **self._CSV_ARGS)
        return df.copy()

    @SILENCE
    def save_candles(self, symbol, dataframe, date_string):
        """=^.^="""
        dataframe.to_csv(self.__get_path__(symbol, date_string))

    @SILENCE
    def update_candles(self, symbol, new_data, date_string):
        """Combine old data with new data."""
        data_path = self.__get_path__(symbol, date_string)
        do_merge = path.exists(data_path)
        data = pandas.DataFrame()
        new_data.dropna(inplace=True)
        if len(new_data) > 0:
            if do_merge:
                if not local_data:
                    local_data = self.load_candles(symbol, date_string)
                local_data.dropna(inplace=True)
                if len(local_data) > 0:
                    data = local_data.combine_first(new_data)
            else:
                data = new_data
            if len(data) > 0:
                data.dropna(inplace=True)
                if len(data) > 0:
                    self.save_candles(symbol, data.copy(), date_string)

    @SILENCE
    def do_update(self, symbol, limit=None,
                  start_date=None, end_date=None):
        """Update historical price data for a given symbol."""
        date_string = start_date.split('T')[0]
        if path.exists(self.__get_path__(symbol, date_string)):
            return True
        verbose = True if self._VERBOSE else False
        tk = self._TIMER
        start_time = tk.reset
        tz = self._tz
        pts = pandas.Timestamp
        if verbose:
            self._VERBOSE = not self._VERBOSE
            m = 'Candelabrum: updating {} from {} to {}...'
            print(m.format(symbol, start_date, end_date))
        qa = dict(limit=limit, start_date=start_date, end_date=end_date)
        q = self.api.candles(symbol, **qa)
        if len(q) > 0:
            bars = pandas.DataFrame(q['bars'])
            bars['time'] = [pts(t, unit='s', tz=tz) for t in bars['t']]
            bars.set_index('time', inplace=True)
            bars.rename(columns=self._COL_NAMES, inplace=True)
            tmp = self.fill_template(bars.copy())
            self.update_candles(symbol, tmp, date_string)
        if verbose:
            print(f'Candelabrum: finished update in {tk.final} seconds.')
            self._VERBOSE = not self._VERBOSE
        return False

    @SILENCE
    def gather_benchmarks(self, start_date, end_date, timing):
        if len(self._BENCHMARKS) < 1: return False
        for symbol in self._BENCHMARKS:
            try:
                candles = self.gather_data(symbol, start_date, end_date)
                scaled = self.resample_candles(candles, timing)
                self.benchmarks[symbol] = self.apply_indicators(scaled)
            finally:
                continue
        return True

    @SILENCE
    def gather_data(self, symbol, start_date, end_date):
        calendar = market_calendar()
        calendar_dates = calendar.index.tolist()
        from datetime import datetime
        date_obj = lambda t: datetime.strptime(t, '%Y-%m-%d')
        dates = (date_obj(start_date), date_obj(end_date))
        candles = None
        for ts in calendar_dates:
            day_obj = date_obj(ts)
            if day_obj < dates[0]: continue
            if day_obj > dates[1]: break
            try:
                day_data = self.load_candles(symbol, ts)
                if len(day_data) > 0:
                    if candles is None:
                        candles = day_data.copy()
                    else:
                        candles = pandas.concat([candles, day_data])
            finally:
                pass
        return candles.copy()

    @SILENCE
    def apply_indicators(self, candles):
        global icy
        money = icy.get_money(candles['close'].tolist())
        zs, sdev, wema, dh, dl, mid = zip(*money)
        candles['money_zscore'] = zs
        candles['money_sdev'] = sdev
        candles['money_wema'] = wema
        candles['money_dh'] = dh
        candles['money_dl'] = dl
        candles['money_mid'] = mid
        vm = icy.get_money(candles['volume'].tolist())
        zs_v, sdev_v, wema_v, dh_v, dl_v, mid_v = zip(*vm)
        candles['volume_zscore'] = zs_v
        candles['volume_sdev'] = sdev_v
        candles['volume_wema'] = wema_v
        candles['volume_dh'] = dh_v
        candles['volume_dl'] = dl_v
        candles['volume_mid'] = mid_v
        return candles.copy()

    @SILENCE
    def resample_candles(self, candles, scale):
        c = candles.resample(scale).apply(self.candle_maker)
        c.dropna(inplace=True)
        return c.copy()

    @SILENCE
    def candle_maker(self, candles):
        if len(candles) > 0 and type(candles) == pandas.Series:
            name = str(candles.name)
            value = None
            if name == 'utc_ts':
                value = float(make_utc(candles[0]))
            elif name == 'open':
                value = float(candles[0])
            elif name == 'high':
                value = float(candles.max())
            elif name == 'low':
                value = float(candles.min())
            elif name == 'close':
                value = float(candles[-1])
            elif name == 'volume':
                value = float(candles.sum())
            return value
        return None


def cheese_wheel(silent=True, max_days=34, do_update=True,
                 limit=None, market_open=None, market_close=None):
    """Update and test historical data."""
    global icy
    icy.SILENT = silent
    cdlm = Candelabrum()
    ivy_ndx = composite_index()
    api_clock = cdlm.api.clock()
    status = bool(api_clock['is_open'])
    if not silent:
        print(f'Cheese Wheel: market status returned {status}')
    u = dict(limit=limit, start_date=market_open, end_date=market_close)
    cdlm.do_update(ivy_ndx, **u)
    return api_clock


def validate_mice(start_date, end_date, silent=True, max_days=89, timing="1H"):
    """Collect cheese from mice and do tests."""
    global icy
    icy.SILENT = silent
    cdlm = Candelabrum()
    ivy_ndx = composite_index()
    benchmarked = cdlm.gather_benchmarks(start_date, end_date, timing)
    mice = icy.ThreeBlindMice(ivy_ndx, max_days=max_days,
                              BENCHMARKS=dict(cdlm.benchmarks))
    make_cheese = mice.get_cheese
    get_candles = cdlm.gather_data
    resample = cdlm.resample_candles
    omenize = cdlm.apply_indicators
    print('Validate Mice: starting quest for the ALL CHEESE.')
    tk = icy.TimeKeeper()
    for symbol_pair in ivy_ndx:
        try:
            symbol = symbol_pair[0]
            exchange = symbol_pair[1]
            if not silent:
                print(f'Validate Mice: omenizing {symbol}...')
            cdls = get_candles(symbol, start_date, end_date)
            cdls = resample(cdls, timing)
            cdls = omenize(cdls)
            make_cheese(symbol, cdls, exchange)
        finally:
            pass
    print('Validate Mice: performing historical trades...')
    mice.validate_trades()
    if mice:
        with open('./configs/all.cheese', 'wb') as f:
            pickle.dump(mice, f, pickle.HIGHEST_PROTOCOL)
        with open('./configs/last.update', 'w') as f:
            f.write("nya~")
        if not silent:
            m = 'Validate Mice: After {} the quest '
            m += 'for the ALL CHEESE comes to an end.'
            print(m.format(tk.update[0]))
        return mice
    else:
        if not silent:
            print('Validate Mice: quest failed due to no mice.')
        return None


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
    calendar = market_calendar()
    calendar_dates = calendar.index.tolist()
    market_open = None
    market_close = None
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
                if today in calendar_dates:
                    mo = calendar.loc[today]['session_open']
                    mc = calendar.loc[today]['session_close']
                    qt = '{}T{}:{}:00-04:00'
                    pass_check = True
                    if len(mo) != 4:
                        print(f'Error: open has wrong length.\n{mo}\n')
                        pass_check = False
                    if len(mc) != 4:
                        print(f'Error: close has wrong length.\n{mc}\n')
                        pass_check = False
                    if pass_check:
                        market_open = qt.format(today, mo[0:2], mo[2:])
                        market_close = qt.format(today, mc[0:2], mc[2:])
                        print(f'Range: {market_open} to {market_close}.')
                    else:
                        market_open = None
                        market_close = None
                else:
                    print(f"Error: Couldn't find {today} in calendar.")
                schedule = list(get_schedule(today, freq='5min'))
            if utc_now in schedule and all((market_open, market_close)):
                total_spins += 1
                print(f'Spin: {total_spins}')
                wheel_args = dict(max_days=89)
                wheel_args['limit'] = 1000
                wheel_args['market_open'] = market_open
                wheel_args['market_close'] = market_close
                s = cheese_wheel(**wheel_args)
                status = bool(s['is_open'])
                with open('./configs/last.update', 'w') as f:
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
                    if u > 86400:
                        t = '{} day(s)'.format(round(u / 86400, 2))
                    elif u > 3600:
                        t = '{} hour(s)'.format(round(u / 3600, 2))
                    elif u > 60:
                        t = '{} minute(s)'.format(round(u / 60, 2))
                    else:
                        t = '{} second(s)'.format(round(u, 2))
                    print(f'Next spin scheduled for {sleep_date} in {t}')
                    time.sleep(u)
    except KeyboardInterrupt:
        print('Keyboard Interrupt: Stopping loop.')
        looping = False
    finally:
        e = keeper.update[0]
        print(f'Stopped spinning after {e} with {total_spins} spins.')


def build_historical_database(verbose=False):
    """Cycle through calendar and update data accordingly."""
    calendar = market_calendar()
    calendar_dates = calendar.index.tolist()
    ivy_ndx = composite_index()
    market_open = None
    market_close = None
    current_year = False
    current_month = False
    today = time.strftime('%Y-%m-%d', time.localtime())
    local_year = int(today[0:4])
    local_month = int(today[5:7])
    local_day = int(today[8:])
    cdlm = Candelabrum()
    msg = 'Build Historical Database: {}'
    print(msg.format('starting...'))
    uargs = dict(limit=1000)
    query = '{}T{}:{}:00-04:00'
    keeper = icy.TimeKeeper()
    zzz = (60 / (200 / len(ivy_ndx))) + 0.3
    for ts in calendar_dates:
        ts_year = int(ts[0:4])
        if ts_year < 2019: continue
        if ts_year == local_year:
            ts_month = int(ts[5:7])
            if ts_month == local_month:
                ts_day = int(ts[8:])
                if ts_day > local_day:
                    print(msg.format("timestamp in the future, breaking loop."))
                    break
        skippable = [True]
        o = str(calendar.loc[ts]['session_open'])
        c = str(calendar.loc[ts]['session_close'])
        pass_check = True
        if len(o) != 4:
            print(msg.format(f'open has wrong length.\n{o}\n'))
            pass_check = False
        if len(c) != 4:
            print(msg.format(f'close has wrong length.\n{c}\n'))
            pass_check = False
        if pass_check:
            market_open = query.format(ts, o[0:2], o[2:])
            market_close = query.format(ts, c[0:2], c[2:])
        else:
            market_open = None
            market_close = None
        if all((market_open, market_close)):
            if verbose:
                print(msg.format(f'collecting {market_open} to {market_close}.'))
            uargs['start_date'] = market_open
            uargs['end_date'] = market_close
            skippable = [cdlm.do_update(s[0], **uargs) for s in ivy_ndx]
            e = keeper.update[1]
            if e < zzz and not all(skippable):
                z = zzz - e
                print(msg.format(f'going to sleep for {z} seconds.'))
                time.sleep(zzz)
    print(msg.format(f'completed after {keeper.final}.'))
