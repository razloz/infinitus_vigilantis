#!./.env/bin/python3
"""Launcher for the Infinitus Vigilantis application."""
__version__ = '1.1.1'

if __name__ == '__main__':
	print(f'\n{__doc__}\nVersion: {__version__}')
	with open('./license/MIT.txt', 'r') as f:
		LICENSE = f.read()
	with open('./license/Disclaimer.txt', 'r') as f:
		DISCLAIMER = f.read()
	print(f'\n{LICENSE}\n{DISCLAIMER}\n')
	import ivy_gui_tk
	ivy_gui_tk.start_gui()
