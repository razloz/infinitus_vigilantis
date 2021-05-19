#!./.env/bin/python3
"""Launcher for the IVy Cartographer."""

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2021, Daniel Ward'
__license__ = 'GPL v3'
__version__ = '2021.05'
__codename__ = 'bling'


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
    p.add_argument('--size', help='Chart size after resampling. Defaults to 200.')
    p.add_argument('--timing', help=f'Valid times are {vt}. Defaults to 1H.')
    p.add_argument('--daemonize', action='store_true',
                   help='Enable daemonized loop.')
    p.add_argument('--start_date', help='Defaults to 2021-01-01.')
    p.add_argument('--end_date', help='Defaults to today.')
    p.add_argument('--cheeseless', action='store_true',
                   help='Skip signal generation from the ALL CHEESE.')
    p.add_argument('--all', action='store_true',
                   help='Generate charts from index.')
    args = p.parse_args()
    today = time.strftime('%Y-%m-%d', time.localtime())
    s = str(args.start_date) if args.start_date else '2021-01-01'
    e = str(args.end_date) if args.end_date else today
    chart_args = dict(start_date=s, end_date=e)
    chart_args['chart_size'] = int(args.size) if args.size else 200
    if args.daemonize: chart_args['daemon'] = True
    if args.cheeseless: chart_args['no_signals'] = True
    if args.timing:
        if args.timing in vt:
            chart_args['adj_time'] = str(args.timing)
        else:
            chart_args['adj_time'] = '1H'
    else:
        chart_args['adj_time'] = '1H'
    if args.symbol:
        chart_args['symbol'] = str(args.symbol)
        print('Starting argumentative cartographer.')
        charts.cartographer(**chart_args)
    elif args.all:
        import pandas
        from source.ivy_candles import composite_index
        ivy_ndx = composite_index()
        for symbol_pair in ivy_ndx:
            symbol = symbol_pair[0]
            chart_args['symbol'] = str(symbol)
            charts.cartographer(**chart_args)
    else:
        print('Starting default cartographer.')
        charts.cartographer()
