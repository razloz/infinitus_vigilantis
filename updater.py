#!./.env/bin/python3
"""Launcher for the IVy Updater."""
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
__version__ = '2022.10'
__codename__ = 'moirai'


if __name__ == '__main__':
    print(f'\n{__doc__}\nVersion: {__version__} ({__codename__})')
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
    p.add_argument('--indicators', action='store_true',
                   help='Build historical indicators.')
    p.add_argument('--research', action='store_true',
                   help='Visit the three blind mice.')
    p.add_argument('--start_date', help='Defaults to 2019-01-01.')
    p.add_argument('--end_date', help='Defaults to today.')
    vt = ('5Min', '10Min', '15Min', '30Min', '1H', '3H')
    p.add_argument('--timing', help=f'Valid times are {vt}. Defaults to 1H.')
    args = p.parse_args()
    print('Loading IVy Updater...')
    import source.ivy_candles as updater
    if args.build:
        print('Starting historical update loop...')
        updater.build_historical_database()
    elif args.clean:
        updater.Candelabrum().clean_candelabrum()
    elif args.research:
        while True:
            updater.Candelabrum().research_candles()
    elif args.indicators:
        print('Applying indicators...')
        updater.Candelabrum().apply_indicators()
        print("""Job's done!""")
    else:
        print('Missing argument.')
