#!./.env/bin/python3
"""Launcher for the Infinitus Vigilantis application."""
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
    import source.ivy_mouse as ivy
    mice = ivy.ThreeBlindMice()
    import argparse
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    if args.start_learning:
        mice.start_learning()
    elif args.start_serving:
        mice.start_serving()
