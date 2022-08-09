"""Updater for Infinitus Vigilantis"""

import time
import pandas
import numpy
import pickle
import json
import source.ivy_commons as icy
import source.ivy_alpaca as api
from os import path

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
SILENCE = icy.silence
DIV = icy.safe_div
PERCENT = icy.percent_change
SHEPHERD = api.AlpacaShepherd()


def daemonize():
    """Keep spinning in the background while GUI is running."""
    return icy.ivy_dispatcher(spin_wheel, ftype='process')


def market_calendar(cpath='./indexes/calendar.index'):
    """Fetch the market calendar and store it locally."""
    if path.exists(cpath):
        ivy_calendar = pandas.read_csv(cpath, index_col=0, dtype=str)
    else:
        c = SHEPHERD.calendar()
        ivy_calendar = pandas.DataFrame(c, dtype=str)
        ivy_calendar.set_index('date', inplace=True)
        ivy_calendar.to_csv(cpath)
    return ivy_calendar.copy()


def alpaca_assets(asset_path='./indexes/alpaca.assets'):
    """Get assets, filter them, and store locally."""
    if path.exists(asset_path):
        with open(asset_path, 'r') as f:
            assets = json.loads(f.read())
    else:
        assets = dict()
        for obj in SHEPHERD.assets():
            symbol = obj['symbol']
            qualified = all((
                symbol.isalpha(),
                obj['status'] == 'active',
                obj['class'] == 'us_equity',
                obj['tradable'] == True
                ))
            if qualified:
                assets[symbol] = obj
        with open(asset_path, 'w+') as f:
            r = f.write(json.dumps(assets))
    return assets


def composite_index(ci_path='./indexes/composite.index'):
    """Validates assets and returns an index of symbols."""
    if path.exists(ci_path):
        ivy_ndx = pandas.read_csv(ci_path, index_col=0)
    else:
        valid_symbols = dict()
        assets = alpaca_assets()
        i = 0
        f = len(assets)
        print(f'Composite Index: {f} assets returned.')
        def validate_response(resp):
            """Ensure data exists for the entire range of dates."""
            validated = False
            try:
                k = resp['bars'][0].keys()
                b = ('t', 'o', 'h', 'l', 'c', 'v')
                validated = all([True if s in k else False for s in b])
            finally:
                return validated
        qa = dict(limit=10000)
        for symbol, v in assets.items():
            i += 1
            print(f'Composite Index: Validating {symbol} ({i}/{f})...')
            validated = [False, False]
            qa['start_date'] = '2019-01-02'
            qa['end_date'] = '2019-01-02'
            validated[0] = validate_response(SHEPHERD.candles(symbol, **qa))
            qa['start_date'] = '2022-01-03'
            qa['end_date'] = '2022-01-03'
            validated[1] = validate_response(SHEPHERD.candles(symbol, **qa))
            if all(validated):
                print(f'Composite Index: {symbol} validated!')
                valid_symbols[symbol] = v
        ivy_ndx = pandas.DataFrame()
        ivy_ndx['symbols'] = valid_symbols.keys()
        ivy_ndx['exchanges'] = valid_symbols.values()
        ivy_ndx.to_csv(ci_path)
    s = ivy_ndx['symbols'].tolist()
    e = ivy_ndx['exchanges'].tolist()
    if len(s) != len(e): return None
    print(f'Composite Index: {len(s)} valid assets.')
    return [(s[i], e[i]) for i in range(len(s))]


class Candelabrum:
    """Handler for historical price data."""
    def __init__(self, index=None):
        """Set symbol index from composite index."""
        self.set_index = lambda n: [t[0] for t in n]
        if index:
            self._INDEX = self.set_index(index)
        else:
            self._INDEX = list()
        self._THREADS = list()
        self._MAX_THREADS = 8
        self._THREAD_COUNT = range(self._MAX_THREADS)
        self._QUEUE = icy.Queue()
        self._BENCHMARKS = ('QQQ', 'SPY')
        self._DATA_PATH = './candelabrum'
        self._ERROR_PATH = './errors'
        self._TIMER = icy.TimeKeeper()
        self.benchmarks = dict()
        tz = 'America/New_York'
        self._tz = tz
        p = lambda c: pandas.to_datetime(c, utc=True).tz_convert(tz)
        self._CSV_ARGS = dict(
            index_col=0,
            parse_dates=True,
            infer_datetime_format=True,
            date_parser=p
            )
        self._COL_NAMES = {
            't': 'utc_ts',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
            }
        print(f'Candelabrum: creating {self._MAX_THREADS} threads...')
        for _ in self._THREAD_COUNT:
            self._THREADS.append(icy.ivy_dispatcher(self.__worker_thread__))
        print('Candelabrum: initialized.')

    def __get_path__(self, symbol, date_string):
        """You no take candle!"""
        p = f'{self._DATA_PATH}/{symbol.upper()}-{date_string}.ivy'
        return path.abspath(p)

    def __worker_thread__(self):
        """Get jobs and save data locally."""
        while True:
            job = self._QUEUE.get()
            if job == 'exit':
                break
            try:
                symbol, data, start_date = job[0], job[1], job[2]
                p = f'{self._DATA_PATH}/{symbol.upper()}-{start_date}.ivy'
                data.to_csv(path.abspath(p), mode='w+')
            except Exception as err:
                print(f'Worker Thread: {err}\n{job}')
            finally:
                self._QUEUE.task_done()
        self._QUEUE.task_done()

    def join_threads(self):
        """Block until all jobs are finished."""
        print('Candelabrum: waiting for all jobs to finish...')
        for _ in self._THREAD_COUNT:
            self._QUEUE.put('exit')
        self._QUEUE.join()

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

    def load_candles(self, symbol, date_string):
        """=^.^="""
        data_path = self.__get_path__(symbol, date_string)
        if not path.exists(data_path):
            return pandas.DataFrame()
        with open(data_path) as pth:
            df = pandas.read_csv(pth, **self._CSV_ARGS)
        return df.copy()

    def save_candles(self, symbol, dataframe, date_string):
        """=^.^="""
        dataframe.to_csv(self.__get_path__(symbol, date_string), mode='w+')

    def do_update(self, symbols, **kwargs):
        """Update historical price data for a given symbol."""
        ivy_path = f'{self._DATA_PATH}/{symbol.upper()}-{start_date}.ivy'
        ivy_path = path.abspath(ivy_path)
        if path.exists(ivy_path):
            return True
        err_path = f'{self._ERROR_PATH}/{symbol.upper()}-{start_date}.error'
        err_path = path.abspath(err_path)
        if path.exists(err_path):
            with open(err_path, 'r') as error_file:
                error_message = error_file.read()
            if """'bars': None""" in error_message:
                return True
        tk = self._TIMER
        start_time = tk.reset
        tz = self._tz
        pts = pandas.Timestamp
        q = SHEPHERD.candles(symbols, **kwargs)
        t = type(symbols)
        try:
            bars = pandas.DataFrame(q['bars'])
            bars['time'] = [pts(t, unit='s', tz=tz) for t in bars['t']]
            bars.set_index('time', inplace=True)
            bars.rename(columns=self._COL_NAMES, inplace=True)
            tmp = self.fill_template(bars.copy())
            tmp.dropna(inplace=True)
            if len(tmp) > 0:
                print(f'Candelabrum: {symbol} updated {start_date}.')
                self._QUEUE.put((symbol, tmp.copy(), start_date))
        except Exception as err:
            print(f'Candelabrum: encountered exception {err}')
            print(f'Candelabrum: {symbol} returned {q}, date={start_date}')
            with open(err_path, 'w+') as f:
                f.write(str(q))
            return False
        return True

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

    def resample_candles(self, candles, scale):
        c = candles.resample(scale).apply(self.candle_maker)
        c.dropna(inplace=True)
        return c.copy()

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
    api_clock = SHEPHERD.clock()
    status = bool(api_clock['is_open'])
    if not silent:
        print(f'Cheese Wheel: market status returned {status}')
    u = dict(limit=limit, start_date=market_open, end_date=market_close)
    cdlm.do_update(ivy_ndx, **u)
    return status


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
    today = time.strftime('%Y-%m-%d', time.localtime())
    local_year = int(today[0:4])
    local_month = int(today[5:7])
    local_day = int(today[8:])
    cdlm = Candelabrum()
    uargs = dict(limit=10000)
    keeper = icy.TimeKeeper()
    msg = 'Build Historical Database: {}'
    print(msg.format('starting...'))
    for ts in calendar_dates:
        ts_year = int(ts[0:4])
        if ts_year < 2019:
            continue
        ts_month = int(ts[5:7])
        ts_day = int(ts[8:])
        time_stop = (
            ts_year == local_year,
            ts_month == local_month,
            ts_day >= local_day
            )
        if all(time_stop):
            break
        uargs['start_date'] = str(ts)
        uargs['end_date'] = str(ts)
        for s in ivy_ndx:
            cdlm.do_update(s[0], **uargs)
    cdlm.join_threads()
    print(msg.format(f'completed after {keeper.final}.'))
