#!./.env/bin/python3
"""Scan through the candelabrum and remove files that cause an exception."""
import pandas as pd
from os import path, listdir, remove
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
_CANDLE_KEYS_ = ('utc_ts','open','high','low','close',
                 'volume','num_trades','vol_wma_price')


def clean_candelabrum():
    """Try to get the min/max price and volume from each candle."""
    tz = 'America/New_York'
    p = lambda c: pd.to_datetime(c, utc=True).tz_convert(tz)
    candle_args = dict(
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
        date_parser=p
        )
    candelabrum = path.abspath('./candelabrum')
    candelabrum_files = listdir(candelabrum)
    total_candles = len(candelabrum_files)
    exceptions = dict()
    min_price = 1e30
    max_price = 0
    min_volume = 1e30
    max_volume = 0
    scanned = 0
    scanned_msg = '****    Progress: {} / ' + f'{total_candles}    ****'
    for candle_name in candelabrum_files:
        if candle_name[-4:] == '.ivy':
            try:
                candle_path = f'{candelabrum}/{candle_name}'
                with open(candle_path) as candle_file:
                    candle = pd.read_csv(candle_file, **candle_args)
                features = candle.keys()
                match = all([k in features for k in _CANDLE_KEYS_])
                if not match:
                    raise Exception(f'{candle_path} missing features.')
                candle_max = max([
                    float(max(candle['open'])),
                    float(max(candle['high'])),
                    float(max(candle['low'])),
                    float(max(candle['close']))
                    ])
                candle_min = min([
                    float(min(candle['open'])),
                    float(min(candle['high'])),
                    float(min(candle['low'])),
                    float(min(candle['close']))
                    ])
                greatest_volume = float(max(candle['volume']))
                if candle_max > max_price:
                    max_price = float(candle_max)
                if candle_min < min_price:
                    min_price = float(candle_min)
                if greatest_volume > max_volume:
                    max_volume = float(greatest_volume)
                scanned += 1
                print(scanned_msg.format(scanned))
            except Exception as details:
                print(details)
                exceptions[candle_path] = details.args
                continue
    return min_price, max_price, max_volume, exceptions

if __name__ == '__main__':
    min_price, max_price, max_volume, exceptions = clean_candelabrum()
    print('Largest price in the Candelabrum:', max_price)
    print('Smallest price in the Candelabrum:', min_price)
    print('Largest volume in the Candelabrum:', max_volume)
    print(f'****    Removing {len(exceptions)} corrupted candles    ****')
    for corrupted in exceptions.keys():
        try:
            print('Removing', corrupted)
            remove(corrupted)
        except Exception as details:
            print(details.args)
            continue
