"""Common functions used by the Infinitus Vigilantis application"""
import traceback
from time import time
from time import strptime
from time import strftime
from time import mktime
from time import localtime
from time import gmtime
from statistics import stdev
from statistics import mean
from numpy import busday_count, errstate
from datetime import datetime
from threading import Thread, Lock
from queue import Queue
from multiprocessing import Process
from multiprocessing import Queue as mpQueue
from pandas import date_range
from pandas import DataFrame
from dateutil import parser as date_parser
from collections import Counter
from math import isclose
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


SILENT = True
def silence(fn):
    """Wrapper to catch exceptions and silence them."""
    def proxy_fn(*args, **kwargs):
        global SILENT
        try:
            return fn(*args, **kwargs)
        except Exception as details:
            if not SILENT:
                traceback.print_exc()
            return None
    return proxy_fn


THREAD_LOCK = Lock()
def ivy_dispatcher(func, ftype='thread', args=None,
                   kwargs=None, daemon=True):
    """Create a new thread or process."""
    fargs = dict(target=func)
    if args: fargs['args'] = args
    if kwargs: fargs['kwargs'] = kwargs
    if ftype == 'thread':
        f = Thread(**fargs)
    elif ftype == 'process':
        f = Process(**fargs)
    else:
        return None
    f.daemon = daemon
    f.start()
    return f


def safe_div(a, b):
    """Will it divide?"""
    with errstate(divide='ignore', invalid='ignore'):
        try:
            c = a / b
        except Exception as _:
            return b
        return c


def percent_change(current, previous):
    """Will it percentage?"""
    try:
        chg = (current - previous) / previous
    except ZeroDivisionError:
        chg = 0
    finally:
        return float(round(100 * chg, 2))


__weighted__ = lambda c, p, w: c * w + (p * (1 - w))
__ema__ = lambda c, p, l: __weighted__(mean(c), mean(p), 2/l)
_NO_MONEY_ = {'zs': 0, 'sdev': 0, 'wema': 0, 'dh': 0, 'dl': 0, 'mid': 0}
def money_line(points, fast=8, weight=34):
    """Will it cheese?"""
    money = dict(_NO_MONEY_)
    try:
        # flip kwargs for use in reverse list comprehension
        slow = len(points)
        wp = ((fast - 1) * -1, (slow - 1) * -1, (weight - 1) * -1)
        wc = (fast * -1, slow * -1, weight * -1)
        # calculate moving averages
        fast_ema = __ema__(points[wc[0]:], points[wp[0]:], slow)
        slow_ema = __ema__(points[wc[1]:], points[wp[1]:], fast)
        weight_ema = __ema__(points[wc[2]:], points[wp[2]:], weight)
        # calculate weighted exponential average
        wema = ((slow_ema + fast_ema) / 2) * 0.5
        wema += weight_ema * 0.5
        # get standard deviation, zscore
        sdev = stdev(points, xbar=wema)
        cc = points[-1]
        zs = (cc - wema) / sdev
        # get mid point and one deviation above/below current price
        dh = cc + sdev
        dl = cc - sdev
        cl = min(points)
        mid = 0.5 * (max(points) - cl) + cl
        # get the money
        money['zs'] = zs
        money['sdev'] = sdev
        money['wema'] = wema
        money['dh'] = dh
        money['dl'] = dl
        money['mid'] = mid
    finally:
        return money


def pivot_points(*ohlc):
    """Generate a list of pivot points at price."""
    price_points = [p for l in ohlc for p in l]
    return dict(Counter(price_points))


_FIB_LEVELS_ = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]
def fibonacci(points, trend=0, extend=False):
    """Extends/retraces price movement based on the trend's polarity."""
    a = max(points)
    b = min(points)
    c = points[-1]
    s = a - b
    x = c if extend else a
    m = 'fib_extend_' if extend else 'fib_retrace_'
    if trend > 0:
        fibs = {f'{m}{i}': round(x + s * i, 2) for i in _FIB_LEVELS_}
    else:
        fibs = {f'{m}{i}': round(x - s * i, 2) for i in _FIB_LEVELS_}
    return fibs


def gartley(five_point_wave, tolerance = 0.001):
    """Check points for Gartley's harmonic pattern."""
    p = five_point_wave
    t = tolerance
    r1 = fibonacci((p[1], p[0]))
    r2 = fibonacci((p[1], p[2]))
    c1 = isclose(p[2], r1['fib_retrace_0.618'], rel_tol=t)
    c2 = isclose(p[3], r2['fib_retrace_0.382'], rel_tol=t)
    c3 = isclose(p[4], r1['fib_retrace_0.786'], rel_tol=t)
    g = dict(gartley_target=0, gartley_stop_loss=0)
    if all((c1, c2, c3)):
        g['gartley_target'] = p[1] + ((p[1] - p[3]) * 1.618033988749)
        g['gartley_stop_loss'] = p[0]
    return g


def logic_block(candle, exchange):
    """Generate buy and sell signals."""
    cdl_zscore = float(candle.money_zscore)
    cdl_dh = float(candle.money_dh)
    cdl_dl = float(candle.money_dl)
    cdl_median = float(candle.money_mid)
    cdl_money = float(candle.money_wema)
    cdl_price = float(candle.close)
    cdl_open = float(candle.open)
    cdl_volume = float(candle.volume)
    cdl_vwema = float(candle.volume_wema)
    cdl_trend = float(candle.trend[0])
    cdl_signal = str(candle.trend[1])
    exc_zscore = float(exchange.money_zscore)
    exc_median = float(exchange.money_mid)
    exc_money = float(exchange.money_wema)
    exc_price = float(exchange.close)
    exc_dh = float(exchange.money_dh)
    exc_dl = float(exchange.money_dl)
    exc_trend = float(exchange.trend[0])
    exc_signal = str(exchange.trend[1])
    bull_candle = cdl_open < cdl_price
    cdl_bullish = 0 != cdl_price > cdl_money > cdl_median != 0
    cdl_bearish = 0 != cdl_price < cdl_money < cdl_median != 0
    exc_bullish = 0 != exc_price > exc_money > exc_median != 0
    exc_bearish = 0 != exc_price < exc_money < exc_median != 0
    dbl_bull = cdl_signal == 'buy' == exc_signal
    dbl_bear = cdl_signal == 'sell' == exc_signal
    quad_bull = all((dbl_bull, cdl_bullish, exc_bullish))
    quad_bear = all((dbl_bear, cdl_bearish, exc_bearish))
    strong_bull = 0 != cdl_dl >= cdl_median != 0
    strong_bear = 0 != cdl_dh <= cdl_median != 0
    near_money = -0.3 <= cdl_zscore <= 0.3
    far_bull = cdl_zscore >= 3
    far_bear = cdl_zscore <= -3
    buy_logic = all((dbl_bull, strong_bull, near_money))
    sell_logic = all((strong_bear, far_bear, bull_candle))
    if buy_logic:
        return 1
    elif sell_logic:
        return -1
    else:
        return 0


__next_wave__ = lambda i, l: i + 1 if i + 1 < l else 0
def get_indicators(df, index_key='time'):
    """Collects indicators and adds them to the dataframe."""
    trend_strength = 0
    wave_points = [0, 0, 0, 0, 0]
    wave_index = 0
    wave_length = len(wave_points)
    trend = list()
    fibs_extended = list()
    fibs_retraced = list()
    gartley_checks = list()
    sample = 89
    weights = dict(fast=8, weight=34)
    money_p = {f'price_{k}': list() for k in _NO_MONEY_.keys()}
    money_v = {f'volume_{k}': list() for k in _NO_MONEY_.keys()}
    df_range = range(len(df))
    df_last = df_range[-1]
    if sample >= df_last + 1:
        return df.copy()
    # localize dataframe columns
    o = df['open'].tolist()
    h = df['high'].tolist()
    l = df['low'].tolist()
    c = df['close'].tolist()
    v = df['volume'].tolist()
    # start loop
    for i in df_range:
        # trend detection and tracking
        if i >= 2:
            ii = i - 1
            iii = i - 2
            hp1 = h[i]
            hp2 = h[ii]
            hp3 = h[iii]
            lp1 = l[i]
            lp2 = l[ii]
            lp3 = l[iii]
            trending_up = hp1 > hp2 > hp3 and lp1 > lp2 > lp3
            trending_down = hp1 < hp2 < hp3 and lp1 < lp2 < lp3
            if trending_up and trending_down:
                trend_strength = 0
                wave_points[wave_index] = 0
                wave_index = __next_wave__(wave_index, wave_length)
            elif trending_up:
                if trend_strength <= 0:
                    trend_strength = 0
                    wave_points[wave_index] = lp3
                    wave_index = __next_wave__(wave_index, wave_length)
                trend_strength += 1
            elif trending_down:
                if trend_strength >= 0:
                    trend_strength = 0
                    wave_points[wave_index] = hp3
                    wave_index = __next_wave__(wave_index, wave_length)
                trend_strength -= 1
        r = fibonacci(wave_points, trend=trend_strength)
        e = fibonacci(wave_points, trend=trend_strength, extend=True)
        g = gartley(wave_points)
        fibs_retraced.append(r)
        fibs_extended.append(e)
        trend.append(trend_strength)
        gartley_checks.append(g)
        # Collect money line for price and volume
        si = i - sample
        ei = i + 1
        if i == df_last:
            mp = money_line(c[si:], **weights)
            mv = money_line(v[si:], **weights)
        elif i >= sample:
            mp = money_line(c[si:ei], **weights)
            mv = money_line(v[si:ei], **weights)
        else:
            mp = dict(_NO_MONEY_)
            mv = dict(_NO_MONEY_)
        for key, value in mp.items():
            money_p[f'price_{key}'].append(value)
        for key, value in mv.items():
            money_v[f'volume_{key}'].append(value)
    # Add each indicator column to dataframe
    indicators = DataFrame(index=df.index)
    indicators['trend'] = trend
    dfs = [
        DataFrame(fibs_retraced),
        DataFrame(fibs_extended),
        DataFrame(gartley_checks),
        DataFrame(money_p),
        DataFrame(money_v)
        ]
    for dataframe in dfs:
        for c, s in dataframe.iteritems():
            indicators[c] = s.tolist()
    # percent change
    rolling_diff = lambda s: safe_div((s[1:] - s[:-1]), s[:-1])
    dfc = df['close'].values
    dfo = df['open'].values
    dfw = indicators['price_wema'].values
    indicators['chg_cdl'] = safe_div((dfc - dfo), dfo).tolist()
    indicators['chg_close'] = [0] + rolling_diff(dfc).tolist()
    indicators['chg_open'] = [0] + rolling_diff(dfo).tolist()
    indicators['chg_wema'] = [0] + rolling_diff(dfw).tolist()
    indicators.fillna(0, inplace=True)
    return indicators.copy()


class TimeKeeper:
    """Used to invoke elapsed time."""
    def __init__(self):
        self._start_time = time()
        self._timer = self._start_time

    @property
    def final(self):
        """Final elapsed time."""
        return time() - self._start_time

    @property
    def reset(self):
        """Reset start time."""
        self._start_time = time()
        self._timer = self._start_time
        return self._start_time

    @property
    def update(self):
        """Get elapsed time since update was last called."""
        elapsed = time() - self._timer
        if elapsed > 86400:
            since = '{} days'.format(round(elapsed / 86400, 5))
        elif elapsed > 3600:
            since = '{} hours'.format(round(elapsed / 3600, 5))
        elif elapsed > 60:
            since = '{} minutes'.format(round(elapsed / 60, 5))
        else:
            since = '{} seconds'.format(round(elapsed, 5))
        self._timer = time()
        return (since, elapsed)


def posix_from_time(t, f='%Y-%m-%d %H:%M'):
    """Python time struct to POSIX timestamp."""
    ts = mktime(strptime(t, f))
    return float(ts)


class UpdateSchedule(object):
    """Generator for next update time."""
    def __init__(self, date_string, freq='15min'):
        """Set internal variables."""
        _sd = f"{date_string} 00:00:00"
        _ed = f"{date_string} 23:59:59"
        _drng = date_range(start=_sd, end=_ed, freq=freq)
        self._dates = _drng.tolist()
        self._last = len(self._dates) - 1
        self._current = 0

    def __iter__(self):
        """Make iterable."""
        return self

    def __next__(self):
        """Python3 compatibility."""
        return self.next()

    def next(self):
        """Get next POSIX timestamp."""
        if self._current <= self._last:
            ts = self._dates[self._current]
            self._current += 1
            return strftime('%Y-%m-%d %H:%M:%S', gmtime(ts.timestamp()))
        else:
            raise StopIteration()


class FibonacciSequencer():
    def __init__(self):
        self.n = 0
        self.previous = 0
        self.current = 1
    def __seq_gen__(self, steps=1):
        n = self.n + steps
        while self.n <= n:
            next_number = self.current + self.previous
            self.previous = self.current
            self.current = next_number
            self.n += 1
            yield next_number
    def next(self, n=1):
        return [i for i in self.__seq_gen__(steps=n)]
    def skip(self, n):
        for _ in self.__seq_gen__(steps=n):
            pass
