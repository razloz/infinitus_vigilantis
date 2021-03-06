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
    import time
    vt = ('5Min', '10Min', '15Min', '30Min', '1H', '3H')
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', help='Symbol for historical data.')
    p.add_argument('--size', help='Chart size after resampling. Defaults to 540.')
    p.add_argument('--timing', help=f'Valid times are {vt}. Defaults to 15Min.')
    p.add_argument('--daemonize', action='store_true',
                   help='Enable daemonized loop.')
    p.add_argument('--start_date', help='Defaults to 2020-06-01.')
    p.add_argument('--end_date', help='Defaults to today.')
    p.add_argument('--cheeseless', action='store_true',
                   help='Skip signal generation from the ALL CHEESE.')
    p.add_argument('--all', action='store_true',
                   help='Generate charts from index.')
    args = p.parse_args()
    today = time.strftime('%Y-%m-%d', time.localtime())
    s = str(args.start_date) if args.start_date else '2020-06-01'
    e = str(args.end_date) if args.end_date else today
    chart_args = dict(start_date=s, end_date=e)
    chart_args['chart_size'] = int(args.size) if args.size else 540
    if args.daemonize: chart_args['daemon'] = True
    if args.cheeseless: chart_args['no_signals'] = True
    if args.timing:
        if args.timing in vt:
            chart_args['adj_time'] = str(args.timing)
        else:
            chart_args['adj_time'] = '15Min'
    else:
        chart_args['adj_time'] = '15Min'
    if args.symbol:
        chart_args['symbol'] = str(args.symbol)
        print('Starting argumentative cartographer.')
        charts.cartographer(**chart_args)
    elif args.all:
        import pandas
        ci_path = './indexes/composite.index'
        ivy_ndx = pandas.read_csv(ci_path, index_col=0)
        symbols = ivy_ndx['symbols'].tolist()
        for symbol in symbols:
            chart_args['symbol'] = str(symbol)
            charts.cartographer(**chart_args)
    else:
        print('Starting default cartographer.')
        charts.cartographer()
