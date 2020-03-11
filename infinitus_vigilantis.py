#!./.env/bin/python3
"""Launcher for the Infinitus Vigilantis application."""

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
    print('Loading GUI...')
    import source.ivy_gui_tk as ivy
    print('Starting IVy GUI...')
    ivy.start_gui()
