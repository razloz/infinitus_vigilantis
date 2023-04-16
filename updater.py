#!./.env/bin/python3
"""Launcher for the IVy Updater."""
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'muenster'


if __name__ == '__main__':
    print(f'\n{__doc__}\nVersion: {__version__}')
    with open('./license/GPLv3.txt', 'r') as f:
        LICENSE = f.read()
    with open('./license/Disclaimer.txt', 'r') as f:
        DISCLAIMER = f.read()
    print(f'\n{LICENSE}\n{DISCLAIMER}\n')
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--build', action='store_true',
                   help='Build historical database.')
    p.add_argument('--clean', action='store_true',
                   help='Remove all corruption from the candelabrum.')
    p.add_argument('--tlc', action='store_true',
                   help='Feed and care for the mice. <333')
    p.add_argument('--indicators', action='store_true',
                   help='Build historical indicators.')
    p.add_argument('--jazz', action='store_true',
                   help='Alpaca jazz recital.')
    p.add_argument('--pick', action='store_true',
                   help='Select best candles for trading.')
    p.add_argument('--study', action='store_true',
                   help='Quick study session.')
    p.add_argument('--start_date', help='Defaults to 2019-01-01.')
    p.add_argument('--end_date', help='Defaults to today.')
    vt = ('5Min', '10Min', '15Min', '30Min', '1H', '3H')
    p.add_argument('--timing', help=f'Valid times are {vt}. Defaults to 1H.')
    args = p.parse_args()
    if args:
        print('Loading IVy Updater...')
        import source.ivy_candles as updater
        if not args.build:
            candelabrum = updater.Candelabrum()
    else:
        raise('Missing argument.')
    if args.build:
        updater.build_historical_database()
    elif args.clean:
        candelabrum.clean_candelabrum()
    elif args.tlc:
        candelabrum.make_offering(None, cook_time=900, epochs=3)
    elif args.indicators:
        candelabrum.apply_indicators()
    elif args.jazz:
        candelabrum.alpaca_jazz(attendants=137, cook_time=1, epochs=1)
    elif args.pick:
        candelabrum.pick_candles(25)
    elif args.study:
        candelabrum.make_offering(None, cook_time=10800, epochs=-1)
