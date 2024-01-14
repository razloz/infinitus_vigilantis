#!.\.env\Scripts\python.exe
"""Launcher for the Infinitus Vigilantis application."""
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2024, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'gardneri'


if __name__ == '__main__':
    print(f'\n{__doc__}\nVersion: {__version__}')
    with open('.\license_docs\GPLv3.txt', 'r') as f:
        LICENSE = f.read()
    with open('.\license_docs\Disclaimer.txt', 'r') as f:
        DISCLAIMER = f.read()
    print(f'\n{LICENSE}\n{DISCLAIMER}\n')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--build',
        action='store_true',
        help='Build the Candelabrum historical database.',
    )
    parser.add_argument(
        '--create_website',
        action='store_true',
        help='Validate network, plot charts, and build HTML documents.',
    )
    parser.add_argument(
        '--merge_states',
        action='store_true',
        help='Merge client states with server.',
    )
    parser.add_argument(
        '--skip_charts',
        action='store_true',
        help='Start learning.',
    )
    parser.add_argument(
        '--start_learning',
        action='store_true',
        help='Start learning.',
    )
    parser.add_argument(
        '--start_serving',
        action='store_true',
        help='Start the server.',
    )
    parser.add_argument(
        '--study',
        action='store_true',
        help='Study the candelabrum.',
    )
    args = parser.parse_args()
    if args:
        if args.start_learning:
            import source.ivy_mouse as ivy
            mice = ivy.ThreeBlindMice()
            mice.start_learning()
        else:
            if args.build:
                import source.ivy_candles as ivy_candles
                ivy_candles.build_historical_database()
            if args.merge_states or args.start_serving or args.create_website:
                import source.ivy_mouse as ivy
                mice = ivy.ThreeBlindMice()
                if args.merge_states:
                    mice.merge_states()
                if args.create_website:
                    mice.build_https(skip_charts=args.skip_charts)
                if args.start_serving:
                    mice.start_serving()
            elif args.study:
                import source.ivy_cauldron as ivy_cauldron
                cauldron = ivy_cauldron.Cauldron(verbosity=2)
                while True:
                    cauldron.train_network(validate=False)
    else:
        raise(Exception('Missing argument.'))
