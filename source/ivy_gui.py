"""Graphical User Interface for the Infinitus Vigilantis application."""

import gi
import source.ivy_cartography as charting
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GdkPixbuf
from gi.repository import GLib
from source.ivy_candles import composite_index
from os.path import abspath
from os.path import exists
from os.path import getmtime

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2020, Daniel Ward'
__license__ = 'GPL v3'
__version__ = '2020.04'
__codename__ = 'compass'


class MainWindow(Gtk.Window):
    def __init__(self):
        """Create GUI objects and layout."""
        Gtk.Window.__init__(self, title='Infinitus Vigilantis')
        VERTICAL = Gtk.Orientation.VERTICAL
        HORIZONTAL = Gtk.Orientation.HORIZONTAL
        self.load_image = GdkPixbuf.Pixbuf.new_from_file_at_size
        self.ivy_ndx = composite_index(abspath('./indexes/custom.ndx'))
        self.placeholder = abspath('./resources/placeholder.png')
        self.poll_mice = abspath('./configs/last.update')
        self.poll_chart = abspath('./configs/{}.done')
        self.active_symbol = None
        self.watching = False
        self.resample = None
        self.chart_size = None
        self.make_once = False
        self.last_job = dict(symbol='', last_poll=0, last_chart=0)
        self.main_width = 1280
        self.main_height = 720
        self.set_default_size(self.main_width, self.main_height)
        self.layout = Gtk.Box(orientation=VERTICAL, margin=3)
        self.carto_box = Gtk.Box(orientation=HORIZONTAL, margin=3)
        self.chart_orig = self.load_image(self.placeholder, 1280, 720)
        self.chart = Gtk.Image(xalign=0, yalign=0, xpad=3, ypad=3)
        self.chart.set_from_pixbuf(self.chart_orig)
        self.chart_enum = GdkPixbuf.InterpType.BILINEAR
        self.carto_frame = Gtk.Frame(label='Cartography', margin=3)
        self.carto_lst = Gtk.ComboBoxText(margin=3)
        for s in self.ivy_ndx:
            self.carto_lst.append_text(s)
        self.carto_lst.connect('changed', self.on_symbol_change)
        self.carto_btn = Gtk.Button(label='Make Chart', margin=3)
        self.carto_btn.connect('clicked', self.on_carto_press)
        self.carto_wth = Gtk.Button(label='Watch Symbol', margin=3)
        self.carto_wth.connect('clicked', self.on_watch_press)
        self.carto_rst = Gtk.ComboBoxText(margin=3)
        for t in ('1Min', '5Min', '10Min', '15Min', '30Min', '1H', '3H'):
            self.carto_rst.append_text(t)
        self.carto_rst.connect('changed', self.on_resample_press)
        szr_adj = Gtk.Adjustment(100, 10, 1000, 10, 100, 0)
        self.carto_szr = Gtk.Scale(adjustment=szr_adj, margin=3)
        self.carto_szr.set_digits(0)
        self.carto_szr.set_hexpand(True)
        self.carto_szr.connect('value-changed', self.on_scale_adjust)
        self.carto_box.add(self.carto_lst)
        self.carto_box.add(self.carto_rst)
        self.carto_box.add(Gtk.Label(label='Chart Size:', margin=3))
        self.carto_box.add(self.carto_szr)
        self.carto_box.add(self.carto_btn)
        self.carto_box.add(self.carto_wth)
        self.carto_frame.add(self.carto_box)
        self.connect('size-allocate', self.on_resize_event)
        self.connect('destroy', Gtk.main_quit)
        self.layout.add(self.chart)
        self.layout.add(self.carto_frame)
        print('Brewing a pot of tea.')
        p = GLib.PRIORITY_DEFAULT_IDLE
        Gdk.threads_add_timeout(p, 500, self.teapot)
        self.add(self.layout)

    def on_scale_adjust(self, *ignore):
        v = self.carto_szr.get_value()
        if self.chart_size != v:
            self.chart_size = v

    def on_resample_press(self, *ignore):
        """Set resample timeframe for cartographer."""
        resample = self.carto_rst.get_active_text()
        if self.resample != resample:
            self.resample = resample

    def on_watch_press(self, widget):
        """Throw symbol into the pot."""
        if self.active_symbol:
            label = widget.get_label()
            if label == 'Watch Symbol':
                widget.set_label('Stop Watching')
                self.watching = True
            else:
                widget.set_label('Watch Symbol')
                self.watching = False

    def on_carto_press(self, widget):
        """Deploy cartographer."""
        symbol = self.active_symbol
        if symbol:
            self.last_job['symbol'] = None
            self.last_job['last_poll'] = 0
            self.last_job['last_chart'] = 0
            self.make_once = True

    def on_resize_event(self, *ignore):
        """Store resolution for image scaling."""
        a = self.get_allocation()
        if self.main_width != a.width or self.main_height != a.height:
            self.main_width = a.width
            self.main_height = a.height

    def on_symbol_change(self, *ignore):
        """Store selected symbol for the cartographer."""
        selected = self.carto_lst.get_active_text()
        if self.active_symbol != selected:
            self.active_symbol = selected
            self.last_job['symbol'] = None
            self.last_job['last_poll'] = 0
            self.last_job['last_chart'] = 0
            self.make_once = False

    def teapot(self, *ignore):
        """Polling daemon for active chart."""
        symbol = self.active_symbol
        poll_mice = self.poll_mice
        if symbol and exists(poll_mice):
            poll = getmtime(poll_mice)
            last_poll = self.last_job['last_poll']
            if last_poll != poll:
                last_symbol = self.last_job['symbol']
                make_once = self.make_once and last_symbol != symbol
                if self.watching or make_once:
                    cargs = dict(symbol=symbol, daemon=False)
                    chart_size = self.chart_size
                    if chart_size:
                        cargs['chart_size'] = chart_size
                    resample = self.resample
                    if resample:
                        cargs['adj_time'] = resample
                    charting.cartographer(**cargs)
                    self.last_job['symbol'] = symbol
                    self.last_job['last_poll'] = poll
                    if make_once:
                        self.make_once = False
                    print('made it out of cartography...')
            else:
                print('made it to chart poll...')
                poll_chart = self.poll_chart.format(symbol)
                last_chart = self.last_job['last_chart']
                chart_path = f'./charts/{symbol}.png'
                if poll_chart != last_chart and exists(chart_path):
                    w = self.main_width * 0.5
                    h = self.main_height * 0.5
                    self.chart_orig = self.load_image(chart_path, w, h)
                    self.chart.set_from_pixbuf(self.chart_orig)
                    self.last_job['last_chart'] = poll_chart
        return True


def start_app():
    """Thread the teapot and start the application."""
    w = MainWindow()
    w.show_all()
    Gtk.main()
