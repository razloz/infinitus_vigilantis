#!./.env/bin/python3
"""Graphical User Interface for the Infinitus Vigilantis application."""

import pickle
from tkinter import Menu, Frame, Tk, Canvas, PhotoImage, Text
from tkinter.ttk import Treeview
from PIL import Image, ImageTk
from os.path import abspath
from os.path import exists
from os.path import getmtime
from ivy_updater import composite_index
from time import sleep

__version__ = '1.1.1'
__codename__ = 'cria'
def ohhai():
	with open(abspath('./license/MIT.txt'), 'r') as f:
		LICENSE = f.read()
	with open(abspath('./license/Disclaimer.txt'), 'r') as f:
		DISCLAIMER = f.read()
	msg = f'{__doc__}\nVersion: {__version__}\nCodename: {__codename__}\n'
	msg += f'\n{LICENSE}\n{DISCLAIMER}\n'
	return msg


POLL_PATH = abspath('./last.update')
last_update = 0
brewing = False
def tea_pot(*ignore):
	"""Polling daemon that checks update file for changes."""
	global brewing, last_update, POLL_PATH
	print('Brewing a pot of tea.')
	brewing = True
	while brewing:
		if exists(POLL_PATH):
			fp = getmtime(POLL_PATH)
			if fp > last_update:
				last_update = fp
		sleep(0.5)
	return False


class IVyApp(Tk):
	"""Infinitus Vigilantis for Python main window."""
	def __init__(self, *args, **kwargs):
		"""Set screen size and create frames."""
		super().__init__(*args, **kwargs)
		icon_path = abspath('./resources/icon.png')
		self.iconphoto(self, PhotoImage(icon_path))
		self.title('Infinitus Vigilantis')
		self.bind('<Configure>', self.__resize_event__)
		self.old_size = '800x600'
		if exists('window.size'):
			with open('window.size', 'r') as cfg:
				pre_split = cfg.read()
			if 'x' in pre_split:
				res_args = pre_split.split('x')
				self.old_size = f'{int(res_args[0])}x{int(res_args[1])}'
		self.vista = None
		self.chart_path = None
		self.geometry(f'{self.old_size}+0+0')
		self.__create_frames__()
		self.__create_menu__()

	def __create_frames__(self):
		"""Setup vistas."""
		bg_img = abspath('./resources/background.png')
		ph_img = abspath('./resources/placeholder.png')
		t = self.old_size.split('x')
		init_size = (int(t[0]), int(t[1]))
		self.frames = {
			'main': CandelabrumFrame(self, bg_img, init_size),
			'chart': CartographyFrame(self, ph_img, init_size)
		} # self.frames

	def __create_menu__(self):
		"""Setup menu entries."""
		m = lambda: self.set_vista('main')
		c = lambda: self.set_vista('chart')
		self.menu_view = Menu(self, tearoff=0)
		self.menu_view.add_command(label='Candelabrum', command=m)
		self.menu_view.add_command(label='Cartography', command=c)
		self.menu_root = Menu(self)
		self.menu_root.add_cascade(label='View', menu=self.menu_view)
		self.menu_root.add_command(label='Exit', command=self.destroy)
		self.config(menu=self.menu_root)

	def __resize_event__(self, event):
		"""Save current window size."""
		new_width = self.winfo_width()
		new_height = self.winfo_height()
		new_size = f'{new_width}x{new_height}'
		if self.old_size != new_size:
			self.old_size = new_size
			with open('window.size', 'w') as cfg:
				cfg.write(new_size)

	def set_vista(self, vista_key):
		"""Switch between frames."""
		if self.vista:
			self.vista.pack_forget()
		self.vista = self.frames[vista_key]
		self.vista.pack(fill='both', expand=1)


class TemplateFrame(Frame):
	"""Super class for vistas."""
	def __init__(self, parent, bg_path=None, init_size=None):
		"""Create background image on canvas."""
		Frame.__init__(self, parent)
		self.init_size = init_size
		self.parent = parent
		self._last_refresh = 0
		self.columnconfigure(0, weight=1)
		self.rowconfigure(0, weight=1)
		self._raw_image = Image.open(str(bg_path))
		fitted = self._raw_image.resize(init_size, Image.ANTIALIAS)
		self._display_image = ImageTk.PhotoImage(fitted)
		self._background = Canvas(self, bd=0, highlightthickness=0)
		self._bg_args = dict(anchor='nw', tags='bg_img')
		self._bg_args['image'] = self._display_image
		self._background.create_image(0, 0, **self._bg_args)
		self._background.pack(fill='both', expand=True)
		self.bind('<Configure>', self.__resize_event__)

	def __resize_event__(self, event):
		"""Refresh background image."""
		new_size = (event.width, event.height)
		fitted = self._raw_image.resize(new_size, Image.ANTIALIAS)
		self._display_image = ImageTk.PhotoImage(fitted)
		self._bg_args['image'] = self._display_image
		self._background.delete('bg_img')
		self._background.create_image(0, 0, **self._bg_args)


class CandelabrumFrame(TemplateFrame):
	"""Candelabrum vista."""
	def __init__(self, parent, bg_path=None, init_size=None):
		TemplateFrame.__init__(self, parent, bg_path=bg_path,
							   init_size=init_size)
		self.__create_objects__()
		self.__mouse_sentry__()

	def __create_objects__(self):
		"""Setup interface."""
		self.ivy_ndx = composite_index(abspath('./indexes/current.ndx'))
		self.cheese_wheel = Treeview(self._background, show='tree')
		for s in self.ivy_ndx: self.cheese_wheel.insert('', 'end', text=s)
		self.cheese_wheel.bind('<<TreeviewSelect>>', self.__set_symbol__)
		self.cheese_wheel.pack(side='left', fill='y', padx=20, pady=20)
		self.signal_frame = Frame(self._background)
		self.signal_frame.pack(side='right', fill='both', padx=20, pady=20)
		self.buy_signals = Text(self.signal_frame)
		self.buy_signals.pack(anchor='n')
		self.sell_signals = Text(self.signal_frame)
		self.sell_signals.pack(anchor='s')
		self.status_frame = Frame(self._background)
		self.status_frame.pack(side='right', fill='y', pady=20)
		self.stats = Text(self.status_frame)
		self.stats.pack(anchor='n')
		self.positions = Text(self.status_frame)
		self.positions.pack(anchor='s')

	def __set_symbol__(self, event):
		w = event.widget
		selected = w.item(w.focus(), option='text')
		if selected in self.ivy_ndx:
			sym = str(selected).lower()
			chart_path = abspath(f'./charts/{sym}_15MIN.png')
			self.parent.frames['chart'].chart_path = chart_path
			self.parent.set_vista('chart')

	def __mouse_sentry__(self):
		global last_update
		if last_update > self._last_refresh:
			self._last_refresh = last_update
			with open(abspath(f'./all.cheese'), 'rb') as pkl:
				mice = pickle.load(pkl)
			signals = mice.signals
			positions = mice.positions
			s = list(signals)
			t = (str(s[-1]) if len(s) > 0 else None)
			d = ((t.split(' ')[0] if ' ' in t else t) if t else '')
			buy_string = ''
			sell_string = ''
			for ts in signals:
				if d in ts:
					c = signals[ts]
					buy_signals = c['buy']
					sell_signals = c['sell']
					if len(buy_signals) > 0:
						for signal in buy_signals:
							buy_string += f'{ts}: {signal}\n'
					if len(sell_signals) > 0:
						for signal in sell_signals:
							sell_string += f'{ts}: {signal}\n'
			stats_string = ''
			positions_string = ''
			percentiles = ['roi', 'benchmark', 'net change']
			for k, v in mice.stats.items():
				if k not in percentiles:
					stats_string += f'{k}: {v}\n'
				else:
					stats_string += f'{k}: {v}%\n'
			for sym in positions:
				positions_string += f'{sym}: {mice.positions[sym][0]}\n'
			stats_string = 'The ALL CHEESE:\n' + stats_string
			positions_string = 'Positions:\n' + positions_string
			buy_string = 'Buy Signals:\n' + buy_string
			sell_string = 'Sell Signals:\n' + sell_string
			self.stats.delete(1.0, 'end')
			self.stats.insert('end', stats_string)
			self.positions.delete(1.0, 'end')
			self.positions.insert('end', positions_string)
			self.buy_signals.delete(1.0, 'end')
			self.buy_signals.insert('end', buy_string)
			self.sell_signals.delete(1.0, 'end')
			self.sell_signals.insert('end', sell_string)
		self.parent.after(500, self.__mouse_sentry__)


class CartographyFrame(TemplateFrame):
	"""Cartography vista."""
	def __init__(self, parent, bg_path=None, init_size=None):
		TemplateFrame.__init__(self, parent, bg_path=bg_path,
							   init_size=init_size)
		self._last_path = bg_path
		self.chart_path = bg_path
		self.__refresh_chart__()

	def __refresh_chart__(self):
		"""Keep chart current."""
		global last_update
		choices = (
			last_update > self._last_refresh,
			self.chart_path != self._last_path
		) # choices
		if any(choices):
			self._last_refresh = last_update
			self._last_path = self.chart_path
			self._raw_image = Image.open(self._last_path)
			fitted = self._raw_image.resize(self.init_size,
											Image.ANTIALIAS)
			self._display_image = ImageTk.PhotoImage(fitted)
			self._bg_args['image'] = self._display_image
			self._background.delete('bg_img')
			self._background.create_image(0, 0, **self._bg_args)
		self.parent.after(500, self.__refresh_chart__)


def start_gui(*ignore):
	"""Woof."""
	from threading import Thread
	teapot = Thread(target=tea_pot, daemon=True)
	teapot.start()
	root = IVyApp()
	root.set_vista('main')
	root.mainloop()
	return False


if __name__== '__main__':
	print(ohhai())
	start_gui()
