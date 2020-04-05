#!./.env/bin/python3
"""Launcher for the IVy Updater."""

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2020, Daniel Ward'
__license__ = 'GPL v3'
__version__ = '2020.04'
__codename__ = 'compass'


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
    p.add_argument('--validate', action='store_true',
                   help='Begin the quest for the ALL CHEESE.')
    p.add_argument('--start_date', help='Defaults to 2015-01-01.')
    p.add_argument('--end_date', help='Defaults to today.')
    args = p.parse_args()
    print('Loading IVy Updater...')
    import source.ivy_candles as updater
    if args.validate:
        print('Starting validation routine...')
        import time
        today = time.strftime('%Y-%m-%d', time.localtime())
        s = str(args.start_date) if args.start_date else '2015-01-01'
        e = str(args.end_date) if args.end_date else today
        updater.validate_mice(s, e, silent=False, max_days=89)
    elif args.build:
        print('Starting historical update loop...')
        updater.build_historical_database()
    else:
        print('Starting IVy Updater...')
        updater.spin_wheel()
