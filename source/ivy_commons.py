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
from numpy import busday_count, errstate, inf, nan
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


def get_indicators(df, index_key='time'):
    """Collects indicators and adds them to the dataframe."""
    sample = 34
    trend = list()
    trend_strength = 0
    weights = dict(fast=3, weight=13)
    money_p = {f'price_{k}': list() for k in _NO_MONEY_.keys()}
    money_v = {f'volume_{k}': list() for k in _NO_MONEY_.keys()}
    df_range = range(len(df))
    df_last = df_range[-1]
    if sample >= df_last + 1:
        return df.copy()
    o = df['open'].tolist()
    h = df['high'].tolist()
    l = df['low'].tolist()
    c = df['close'].tolist()
    v = df['volume'].tolist()
    for i in df_range:
        if i >= 2:
            ii = i - 1
            iii = i - 2
            hp1, hp2, hp3 = h[i], h[ii], h[iii]
            lp1, lp2, lp3 = l[i], l[ii], l[iii]
            trending_up = hp1 > hp2 > hp3 and lp1 > lp2 > lp3
            trending_down = hp1 < hp2 < hp3 and lp1 < lp2 < lp3
            if trending_up and trending_down:
                trend_strength = 0
            elif trending_up:
                if trend_strength <= 0:
                    trend_strength = 0
                trend_strength += 1
            elif trending_down:
                if trend_strength >= 0:
                    trend_strength = 0
                trend_strength -= 1
        trend.append(trend_strength)
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
    indicators = DataFrame(index=df.index)
    indicators['trend'] = trend
    for dataframe in [DataFrame(money_p), DataFrame(money_v)]:
        for key, value in dataframe.items():
            indicators[key] = value.tolist()
    price_sum = df['open'].values + df['high'].values
    price_sum += df['low'].values + df['close'].values
    indicators['price_med'] = (price_sum / 4).tolist()
    indicators['pct_chg'] = indicators['price_med'].pct_change(periods=sample)
    indicators.replace([inf, -inf], nan, inplace=True)
    indicators.fillna(0, inplace=True)
    return indicators.copy()
