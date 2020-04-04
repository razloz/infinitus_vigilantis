#!./.env/bin/python3
"""Launcher for the IVy Cartographer."""

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
    print('Loading IVy Cartographer...')
    import source.ivy_cartography as charts
    import argparse
    vt = ('5Min', '10Min', '15Min', '30Min', '1H', '3H')
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', help='Symbol for historical data.')
    p.add_argument('--size', help='Chart size after resampling.')
    p.add_argument('--timing', help=f'Valid times are {vt}.')
    p.add_argument('--daemonize', action='store_true',
                   help='Enable daemonized loop.')
    args = p.parse_args()
    chart_args = dict()
    if args.symbol:
        chart_args['symbol'] = str(args.symbol)
    if args.size:
        chart_args['chart_size'] = int(args.size)
    if args.timing:
        if args.timing in vt:
            chart_args['adj_time'] = str(args.timing)
    if args.daemonize:
        chart_args['daemon'] = True
    if len(chart_args) > 0:
        print(f'Starting argumentative cartographer.')
        charts.cartographer(**chart_args)
    else:
        print('Starting default cartographer.')
        charts.cartographer()
