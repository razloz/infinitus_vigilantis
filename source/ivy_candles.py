"""Updater for Infinitus Vigilantis"""

import time
import pandas
import numpy
import pickle
import source.ivy_commons as icy
import source.ivy_alpaca as api
from os import path

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2020, Daniel Ward'
__license__ = 'GPL v3'
__version__ = '2020.04'
__codename__ = 'compass'
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

def composite_index(ndx_path, ci_path='./indexes/composite.index'):
    """Convert space separated list to csv."""
    if path.exists(ci_path):
        ivy_ndx = pandas.read_csv(ci_path, index_col=0)
    else:
        tk = icy.TimeKeeper()
        with open(ndx_path, 'r') as symbols:
            ndx = symbols.read()
        valid_symbols = list()
        shepherd = api.AlpacaShepherd()
        for asset in shepherd.assets():
            conditions = [
                asset['status'] == 'active',
                asset['tradable'] == True,
                asset['exchange'] == 'NYSE' or 'NASDAQ'
                ] # conditions
            if all(conditions):
                valid_symbols.append(str(asset['symbol']))
        print(f'Composite Index: {len(valid_symbols)} valid assets.')
        ivy_ndx = pandas.DataFrame()
        syms = list({s for s in str(ndx).split() if s in valid_symbols})
        ivy_ndx['symbols'] = syms
        ivy_ndx.to_csv(ci_path)
        fin = tk.final
        if fin < 1: time.sleep(1 - fin)
    return ivy_ndx['symbols'].tolist()


class Candelabrum:
    """Handler for historical price data."""
    def __init__(self, data_path='./candelabrum', verbose=True):
        self._PATH = data_path
        self._VERBOSE = verbose
        icy.SILENT = not verbose
        self._TIMER = icy.TimeKeeper()
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
            print('Candelabrum: initialized.')

    @SILENCE
    def __get_path__(self, symbol):
        """You no take candle!"""
        return f'{self._PATH}/{symbol.upper()}'

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
        template.dropna(inplace=True)
        return template.copy()

    @SILENCE
    def load_candles(self, symbol):
        """=^.^="""
        data_path = f'{self.__get_path__(symbol)}.ivy'
        with open(data_path) as pth:
            df = pandas.read_csv(pth, **self._CSV_ARGS)
        return df.copy()

    @SILENCE
    def save_candles(self, symbol, dataframe):
        """=^.^="""
        data_path = f'{self.__get_path__(symbol)}.ivy'
        dataframe.to_csv(data_path)

    @SILENCE
    def update_candles(self, symbol, new_data, local_data=None):
        """Combine old data with new data."""
        do_merge = path.exists(f'{self.__get_path__(symbol)}.ivy')
        data = pandas.DataFrame()
        new_data.dropna(inplace=True)
        if len(new_data) > 0:
            if do_merge:
                if not local_data:
                    local_data = self.load_candles(symbol)
                local_data.dropna(inplace=True)
                if len(local_data) > 0:
                    data = local_data.combine_first(new_data)
            else:
                data = new_data
            if len(data) > 0:
                data.dropna(inplace=True)
                if len(data) > 0:
                    self.save_candles(symbol, data.copy())

    @SILENCE
    def do_update(self, symbols, limit=None,
                  start_date=None, end_date=None):
        """Update historical data from index."""
        verbose = (True if self._VERBOSE else False)
        tk = self._TIMER
        start_time = tk.reset
        tz = self._tz
        pts = pandas.Timestamp
        if verbose:
            self._VERBOSE = not self._VERBOSE
            print(f'Candelabrum: updating {len(symbols)} symbols...')
        qa = dict(limit=limit, start_date=start_date, end_date=end_date)
        candles = self.api.candles(symbols, **qa)
        for symbol in candles.keys():
            cdls = pandas.DataFrame(candles[symbol])
            if len(cdls) > 0:
                cdls['time'] = [pts(t, unit='s', tz=tz) for t in cdls['t']]
                cdls.set_index('time', inplace=True)
                cdls.rename(columns=self._COL_NAMES, inplace=True)
                self.update_candles(symbol, self.fill_template(cdls.copy()))
        if verbose:
            print(f'Candelabrum: finished update in {tk.final} seconds.')
            self._VERBOSE = not self._VERBOSE

    @SILENCE
    def apply_indicators(self, symbol, candles):
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


def cheese_wheel(silent=True, max_days=34, do_update=True,
                 limit=None, market_open=None, market_close=None):
    """Update and test historical data."""
    global icy
    icy.SILENT = silent
    cdlm = Candelabrum()
    ivy_ndx = composite_index('./indexes/custom.ndx')
    api_clock = cdlm.api.clock()
    status = bool(api_clock['is_open'])
    if not silent:
        print(f'Cheese Wheel: market status returned {status}')
    u = dict(limit=limit, start_date=market_open, end_date=market_close)
    cdlm.do_update(ivy_ndx, **u)
    return api_clock


def validate_mice(silent=True, max_days=34):
    """Collect cheese from mice and do tests."""
    global icy
    icy.SILENT = silent
    cdlm = Candelabrum()
    ivy_ndx = composite_index('./indexes/custom.ndx')
    mice = icy.ThreeBlindMice(ivy_ndx, max_days=max_days)
    make_cheese = mice.get_cheese
    get_candles = cdlm.load_candles
    omenize = cdlm.apply_indicators
    if not silent:
        print('Validate Mice: starting quest for the ALL CHEESE.')
    tk = icy.TimeKeeper()
    for symbol in ivy_ndx:
        try:
            print(f'Validate Mice: omenizing {symbol}...')
            cdls = omenize(symbol, get_candles(symbol))
            make_cheese(symbol, cdls)
        finally:
            pass
    print('Validate Mice: performing historical trades...')
    mice.validate_trades()
    if mice:
        with open('./configs/all.cheese', 'wb') as f:
            pickle.dump(mice, f, pickle.HIGHEST_PROTOCOL)
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


def build_historical_database():
    """Cycle through calendar and update data accordingly."""
    calendar = market_calendar()
    calendar_dates = calendar.index.tolist()
    ivy_ndx = composite_index('./indexes/custom.ndx')
    market_open = None
    market_close = None
    current_year = False
    current_month = False
    local_date = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    today = local_date.split(' ')[0]
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
        ts_month = int(ts[5:7])
        ts_day = int(ts[8:])
        if ts_year == local_year:
            if ts_month == local_month:
                if ts_day > local_day:
                    print(msg.format("timestamp in the future, breaking loop."))
                    break
        if ts_year != 2020: continue
        try:
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
                print(msg.format(f'collecting {market_open} to {market_close}.'))
                uargs['start_date'] = market_open
                uargs['end_date'] = market_close
                cdlm.do_update(ivy_ndx, **uargs)
        finally:
            e = keeper.update[1]
            if e < zzz:
                z = zzz - e
                print(msg.format(f'going to sleep for {z} seconds.'))
                time.sleep(zzz)
    print(msg.format(f'completed after {keeper.final}.'))
