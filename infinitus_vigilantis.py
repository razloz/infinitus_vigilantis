#!./.env/bin/python3
"""Launcher for the Infinitus Vigilantis application."""

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'
__version__ = '2022.09'
__codename__ = 'moirai'


if __name__ == '__main__':
    print(f'\n{__doc__}\nVersion: {__version__} ({__codename__})')
    with open('./license/GPLv3.txt', 'r') as f:
        LICENSE = f.read()
    with open('./license/Disclaimer.txt', 'r') as f:
        DISCLAIMER = f.read()
    print(f'\n{LICENSE}\n{DISCLAIMER}\n')
    print('Loading GUI...')
    import source.ivy_gui as ivy
    print('Starting IVy GUI...')
    ivy.start_app()
