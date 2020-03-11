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
from datetime import datetime
from numpy import busday_count
from datetime import datetime
from threading import Thread
from multiprocessing import Process
from pandas import date_range
from dateutil import parser as date_parser


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
    return f


def safe_div(a, b):
    """Will it divide?"""
    try:
        c = a / b
    except ZeroDivisionError:
        c = 0
    finally:
        return c


def percent_change(current, previous):
    """Will it percentage?"""
    try:
        chg = (current - previous) / previous
    except ZeroDivisionError:
        chg = 0
    finally:
        return float(round(100 * chg, 2))


_WEIGHTED = lambda c, p, w: c * w + (p * (1 - w))
_EMA = lambda c, p, l: _WEIGHTED(mean(c), mean(p), 2/l)
def money_line(points, fast=8, weight=34):
    """Will it cheese?"""
    try:
        global _EMA
        # flip kwargs for use in reverse list comprehension
        slow = len(points)
        wp = ((fast - 1) * -1, (slow - 1) * -1, (weight - 1) * -1)
        wc = (fast * -1, slow * -1, weight * -1)
        # calculate moving averages
        fast_ema = _EMA(points[wc[0]:], points[wp[0]:], slow)
        slow_ema = _EMA(points[wc[1]:], points[wp[1]:], fast)
        weight_ema = _EMA(points[wc[2]:], points[wp[2]:], weight)
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
        return (zs, sdev, wema, dh, dl, mid)
    except Exception as details:
        if not SILENT:
            print(f'money_line encountered {details.args}')
            traceback.print_exc()
        return (0, 0, 0, 0, 0, 0)


def get_money(points, sample=89, fast=13, weight=34):
    """Compile money line from points."""
    points_range = range(len(points))
    points_end = points_range[-1]
    money = list()
    sample = int(sample)
    s = sample * -1
    min_size = sample - 1
    weights = dict(fast=int(fast), weight=int(weight))
    for i in points_range:
        if i == points_end:
            money.append(money_line(points[s:], **weights))
        elif i >= min_size:
            si = i + (s + 1)
            ei = i + 1
            money.append(money_line(points[si:ei], **weights))
        else:
            money.append((0, 0, 0, 0, 0, 0))
    return money


def get_trend(highs, lows):
    """Trend detection."""
    high_len = len(highs)
    low_len = len(lows)
    if high_len != low_len:
        if not SILENT:
            print('Lists must be of same length.')
        return None
    trend = list()
    trend_range = range(high_len)
    trend_end = trend_range[-1]
    trend_strength = 0
    for i in trend_range:
        if i >= 2:
            trend_up = (
                highs[i] > highs[i-1] > highs[i-2],
                lows[i] > lows[i-1] > lows[i-2]
                ) # trend_up
            trend_down = (
                highs[i] < highs[i-1] < highs[i-2],
                lows[i] < lows[i-1] < lows[i-2]
                ) # trend_down
            trending_up = all(trend_up)
            trending_down = all(trend_down)
            trend_strength = 0
            if trending_up != trending_down:
                if trending_up:
                    trend_strength = 1
                elif trending_down:
                    trend_strength = -1
        trend.append(trend_strength)
    return trend


def trend_line(trend):
    """Get trend strength."""
    strength = list()
    t = 0
    for v in trend:
        if v > 0:
            if t < 0:
                t = 0
        elif v < 0:
            if t > 0:
                t = 0
        t += v
        strength.append(t)
    return strength


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


@silence
def logic_block(candle):
    """Generate buy and sell signals."""
    zscore = float(candle.money_zscore)
    dh = float(candle.money_dh)
    dl = float(candle.money_dl)
    median = float(candle.money_mid)
    bullish = 0 != dl > median != 0
    bearish = 0 != dh < median != 0
    buy_logic = (bullish, zscore <= 0.3819661)
    sell_logic = (zscore >= 3, bearish, zscore <= -3)
    if all(buy_logic):
        return 1
    elif any(sell_logic):
        return -1
    else:
        return 0


class ThreeBlindMice:
    """
        Backtest handler for Infinitus Vigilantis.
            Properties:
                stats, benchmark, symbols,
                pending, positions, ledger
    """
    def __init__(self, symbols, cash=5e5, risk=0.0038196,
                 max_days=34, day_trade=False):
        """Set local variables."""
        self._symbols = list(symbols)
        self._init_cash = float(cash)
        self._cash = float(cash)
        self._risk = float(risk)
        self._buyin = round(safe_div(self._init_cash, len(self._symbols)))
        self._total_trades = 0
        self._roi = 0
        self._gain = 0
        self._loss = 0
        self._max_days = int(max_days)
        self._day_trade = bool(day_trade)
        self._benchmark = dict()
        self._positions = dict()
        self._signals = dict()
        self._pending = dict()

    @silence
    def __get_day__(self, t):
        """Drop extra time info from string."""
        ts = (t.split(" ")[0] if " " in t else t)
        return str(ts)

    @silence
    def __sorted_append__(self, key, obj):
        """Get sorted obj after adding key."""
        if key not in obj:
            obj[key] = dict(buy=[], sell=[], neutral=[])
            sorted_keys = sorted(obj, reverse=False)
            return {k: obj[k] for k in sorted_keys}
        else:
            return obj

    @silence
    def __validate_trade__(self, timestamp, symbol, signal,
                           price, stop_loss, target):
        """Manage trade signals."""
        _SAPP = self.__sorted_append__
        _SARGS = (symbol, price, stop_loss, target)
        if symbol not in self._positions:
            valid = self._day_trade
            ts_day = timestamp.split(' ')[0]
            if not valid:
                valid = True
                for ts in self._signals.keys():
                    if ts_day in ts:
                        s = self._signals[ts]['sell']
                        if len(s) > 0:
                            for cheese in s:
                                if cheese[0] == symbol:
                                    valid = False
                                    break
            if valid and signal == 1:
                shares = int(safe_div(self._buyin, price))
                cost = price * shares
                if cost > self._cash:
                    shares = int(safe_div(self._cash, price))
                    cost = price * shares
                if self._cash - cost >= 0 and cost > 0 < self._cash:
                    self._cash -= cost
                    risk_adj = price * self._risk
                    new_position = (
                        timestamp,
                        cost,
                        shares,
                        stop_loss - risk_adj,
                        target + risk_adj
                        ) # new_position
                    self._positions[symbol] = new_position
                    self._signals = _SAPP(timestamp, self._signals)
                    self._signals[timestamp]['buy'].append(_SARGS)
        else:
            ts, entry, shares, adj_stop, adj_target = self._positions[symbol]
            equity = price * shares
            entry_day = self.__get_day__(ts)
            curr_day = self.__get_day__(timestamp)
            days = busday_count(entry_day, curr_day)
            trade_stops = any((
                price >= adj_target,
                price <= adj_stop,
                days >= self._max_days,
                signal == -1 and equity > entry
                )) # trade_stops
            time_check = True if self._day_trade else days > 1
            if time_check and trade_stops:
                self._cash += equity
                roi = percent_change(equity, entry)
                if roi > 0:
                    self._gain += 1
                elif roi < 0:
                    self._loss += 1
                self._roi += roi
                self._total_trades += 1
                self._signals = _SAPP(timestamp, self._signals)
                self._signals[timestamp]['sell'].append(_SARGS)
                del self._positions[symbol]

    @silence
    def get_cheese(self, symbol, dataframe):
        """Get signals and queue orders."""
        closes = dataframe['close'].tolist()
        self._benchmark[symbol] = (closes[0], closes[-1])
        for candle in dataframe.itertuples():
            ts = candle[0].strftime('%Y-%m-%d %H:%M')
            signal = logic_block(candle)
            if not signal: signal = 0
            self._pending = self.__sorted_append__(ts, self._pending)
            candle_close = float(candle.close)
            pargs = (
                str(symbol),
                candle_close,
                float(candle.money_dl),
                candle_close + float(candle.money_sdev) * 1.6180339
                ) # pargs
            if signal == 1:
                self._pending[ts]['buy'].append(pargs)
            elif signal == -1:
                self._pending[ts]['sell'].append(pargs)
            else:
                self._pending[ts]['neutral'].append(pargs)

    @silence
    def validate_trades(self):
        """Check pending signals."""
        validate = self.__validate_trade__
        for ts, cheese in self._pending.items():
            for s in cheese['buy']:
                validate(ts, s[0], 1, s[1], s[2], s[3])
            for s in cheese['sell']:
                validate(ts, s[0], -1, s[1], s[2], s[3])
            for s in cheese['neutral']:
                validate(ts, s[0], 0, s[1], s[2], s[3])

    @property
    def positions(self):
        """Get positions."""
        return dict(self._positions)

    @property
    def signals(self):
        """Get signals."""
        return dict(self._signals)

    @property
    def pending(self):
        """Get pending."""
        return dict(self._pending)

    @property
    def symbols(self):
        """Get symbols."""
        return list(self._symbols)

    @property
    def benchmark(self):
        """Get benchmark."""
        return dict(self._benchmark)

    @property
    def stats(self):
        """Get full backtest results."""
        b = self._benchmark
        p = self._positions
        r = self._buyin
        i = self._init_cash
        sd = safe_div
        mo = sum(list(map(lambda s: b[s][0] * int(sd(r, b[s][0])), b)))
        mc = sum(list(map(lambda s: b[s][1] * int(sd(r, b[s][1])), b)))
        mark_close = (i - mo) + mc
        equity = sum(list(map(lambda s: b[s][1] * p[s][2], p)))
        net = self._cash + equity
        t = list(self._signals.keys())
        gains = int(self._gain)
        losses = int(self._loss)
        profit_loss = (1, 1)
        if gains != 0 != losses:
            if gains > losses:
                profit_loss = (round(sd(gains, losses), 2), 1)
            elif gains < losses:
                profit_loss = (1, round(sd(losses, gains), 2))
        stats = {
            'sample entries': len(t),
            'initial cash': float(i),
            'current cash': round(self._cash, 2),
            'equity': round(equity, 2),
            'net balance': round(net, 2),
            'net change': percent_change(net, i),
            'benchmark': percent_change(mark_close, i),
            'risk threshhold': float(self._risk),
            'buy limit': float(r),
            'intra day': bool(self._day_trade),
            'max days': int(self._max_days),
            'trades': int(self._total_trades),
            'positions': len(p),
            'roi': round(sd(self._roi, self._total_trades), 2),
            'gains': gains,
            'losses': losses,
            'profit_loss': profit_loss
            } # stats
        return stats


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
