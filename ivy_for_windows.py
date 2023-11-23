#!.\.env\Scripts\python.exe
"""Launcher for the Infinitus Vigilantis application."""
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'muenster'


if __name__ == '__main__':
    print(f'\n{__doc__}\nVersion: {__version__}')
    with open('.\license\GPLv3.txt', 'r') as f:
        LICENSE = f.read()
    with open('.\license\Disclaimer.txt', 'r') as f:
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
        '--skip_validation',
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
    parser.add_argument(
        '--cook',
        action='store_true',
        help='Initial network burn-in period.',
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
                    if args.skip_validation:
                        mice.build_https(skip_validation=True)
                    else:
                        mice.build_https(skip_validation=False)
                if args.start_serving:
                    mice.start_serving()
            elif args.study:
                import source.ivy_cauldron as ivy_cauldron
                cauldron = ivy_cauldron.Cauldron(verbosity=1)
                depth = 1 if args.cook else 9
                hours = 168 if args.cook else 9999
                checkpoint = 500 if args.cook else 90
                use_mask = True if args.cook else False
                while True:
                    cauldron.train_network(
                        depth=depth,
                        hours=hours,
                        checkpoint=checkpoint,
                        use_mask=use_mask,
                    )
    else:
        raise(Exception('Missing argument.'))
