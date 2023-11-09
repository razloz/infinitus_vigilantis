"""Graphical User Interface for the Infinitus Vigilantis application."""
import asyncio
import websockets
import json
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk
from gi.repository import Gdk
from gi.repository import GdkPixbuf
from gi.repository import GLib
from os.path import abspath
from os.path import exists
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2023, Daniel Ward'
__license__ = 'GPL v3'


class MainWindow(Gtk.Window):
    def __init__(self):
        """Create GUI objects and layout."""
        Gtk.Window.__init__(self, title='Infinitus Vigilantis')
        VERTICAL = Gtk.Orientation.VERTICAL
        HORIZONTAL = Gtk.Orientation.HORIZONTAL
        self.load_image = GdkPixbuf.Pixbuf.new_from_file_at_size
        self.placeholder = abspath('./resources/placeholder.png')
        self.main_width = 1280
        self.main_height = 720
        self.accepting_data = False
        self.set_default_size(self.main_width, self.main_height)
        self.layout = Gtk.Box(orientation=VERTICAL, margin=3)
        self.background_box = Gtk.Box(orientation=HORIZONTAL, margin=3)
        self.background = Gtk.Image(xalign=0, yalign=0, xpad=3, ypad=3)
        self.background.set_from_pixbuf(
            self.load_image(
                self.placeholder,
                self.main_width,
                self.main_height,
                ),
            )
        self.status_frame = Gtk.Frame(label='Status', margin=3)
        self.toggle_server = Gtk.Button(label='Start Server', margin=3)
        self.toggle_server.connect('clicked', self.on_toggle_server)
        self.merge_data = Gtk.Button(label='Merge Learning Data', margin=3)
        self.merge_data.connect('clicked', self.on_merge_data)
        self.layout.add(self.background)
        self.layout.add(self.status_frame)
        print('Brewing a pot of tea.')
        Gdk.threads_add_timeout(GLib.PRIORITY_DEFAULT_IDLE, 500, self.teapot)
        self.add(self.layout)

    def on_toggle_server(self, widget):
        """Toggle the server on and off."""
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

    def teapot(self, *ignore):
        """Server host for managing module weights and biases."""
        asyncio.get_event_loop().run_until_complete(
            websockets.serve(wss, '0.0.0.0', 411),
            )
        asyncio.get_event_loop().run_forever()

    async def wss(self, websocket, path):
        """WebSocket Server for distributed learning."""
        print(path)
        while True:
            while self.accepting_data:
                try:
                    print('Accepting new data.')
                    rx = await s.recv()
                    self.received_params += 1
                    print(rx)
                except Exception as details:
                    print(*details)

def start_app():
    """Thread the teapot and start the application."""
    w = MainWindow()
    w.show_all()
    Gtk.main()
