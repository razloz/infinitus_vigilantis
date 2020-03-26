#!./.env/bin/python3
"""Launcher for the IVy Cartographer."""

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2020, Daniel Ward'
__license__ = 'MIT'
__version__ = '2020.03'
__codename__ = 'compass'


if __name__ == '__main__':
    print(f'\n{__doc__}\nVersion: {__version__}')
    with open('./license/MIT.txt', 'r') as f:
        LICENSE = f.read()
    with open('./license/Disclaimer.txt', 'r') as f:
        DISCLAIMER = f.read()
    print(f'\n{LICENSE}\n{DISCLAIMER}\n')
    print('Loading IVy Cartographer...')
    import source.ivy_cartography as charts
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", help='Symbol for historical data.')
    p.add_argument("--size", help='Chart size after resampling.')
    valid_times = ('5Min', '10Min', '15Min', '30Min', '1H', '3H')
    p.add_argument("--timing", help=f'Valid times are {valid_times}.')
    args = p.parse_args()
    print('Starting IVy Cartographer...')
    if args.symbol:
        sym = str(args.symbol)
        cs = int(args.size) if args.size else 610
        adj = str(args.timing) if args.timing else '5Min'
        charts.cartographer(symbol=sym, chart_size=cs, adj_time=adj)
    else:
        print('No arguments passed...daemonizing.')
        charts.cartographer()
