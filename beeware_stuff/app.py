"""
Market Forecasting
"""
import toga
import source.ivy_commons as icy
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from os.path import abspath
THREAD_LOCK = icy.THREAD_LOCK
dispatcher = icy.ivy_dispatcher


class InfinitusVigilantis(toga.App):

    def startup(self):
        """
        Construct and show the Toga application.

        Usually, you would add your application to a main content box.
        We then create a main window (with a name matching the app), and
        show the main window.
        """
        Box = toga.Box
        Divider = toga.Divider
        HORIZONTAL = Divider.HORIZONTAL
        VERTICAL = Divider.VERTICAL
        main_box = Box(
            style=Pack(
                direction=COLUMN,
                flex=1,
                padding=5,
            ),
        )
        self.toggle_client_btn = toga.Button(
            'Start Client',
            on_press=self.toggle_client,
        )
        self.auto_client = toga.Switch('auto-start')
        self.toggle_server_btn = toga.Button(
            'Start Server',
            on_press=self.toggle_server,
        )
        self.auto_server = toga.Switch('auto-start')
        self.message_log = toga.MultilineTextInput(
            id='message_log',
            style=Pack(
                direction=COLUMN,
                flex=1,
                padding=5,
            ),
        )
        self.status_text = toga.Label(
            'Status: idle.',
            style=Pack(
                direction=ROW,
                flex=1,
                padding=5,
            ),
        )
        button_box = Box(
            children=[
                Box(
                    children=[
                        self.toggle_server_btn,
                        Divider(
                            direction=HORIZONTAL,
                            style=Pack(padding=15),
                        ),
                        self.auto_server,
                    ],
                    style=Pack(
                        direction=COLUMN,
                        flex=1,
                        padding=5,
                    ),
                ),
                Divider(
                    direction=VERTICAL,
                    style=Pack(padding=15),
                ),
                Box(
                    children=[
                        self.toggle_client_btn,
                        Divider(
                            direction=HORIZONTAL,
                            style=Pack(padding=15),
                        ),
                        self.auto_client,
                    ],
                    style=Pack(
                        direction=COLUMN,
                        flex=1,
                        padding=5,
                    ),
                ),
            ],
            style=Pack(
                direction=ROW,
                padding=15,
            ),
        )
        main_box.add(button_box)
        main_box.add(self.message_log)
        main_box.add(self.status_text)
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.content = main_box
        self.main_window.show()

    def toggle_server(self, button):
        if button.text == 'Start Server':
            self.toggle_client_btn.enabled = False
            button.text = 'Stop Server'
        else:
            self.toggle_server_btn.enabled = True
            self.toggle_client_btn.enabled = True
            button.text = 'Start Server'

    def toggle_client(self, button):
        if button.text == 'Start Client':
            self.toggle_server_btn.enabled = False
            button.text = 'Stop Client'
        else:
            self.toggle_server_btn.enabled = True
            self.toggle_client_btn.enabled = True
            button.text = 'Start Client'


class Teapot():
    def __init__(self):
        """Local distribution network."""
        self.sending_data = False
        self.accepting_data = False
        self.host_ip = '192.168.1.227'

    async def start(selef, *_):
        """Manage module weights and biases over Web Socket."""
        asyncio.get_event_loop().run_until_complete(
            websockets.serve(wss, '0.0.0.0', 411),
            )
        asyncio.get_event_loop().run_forever()

    async def javafy(self, data, load_data=False):
        if load_data:
            return json.loads(data)
        else:
            return json.dumps(data)

    async def wss(self, websocket, path):
        """Web Socket Server for distributed learning."""
        print(path)
        cauldron = self.cauldron
        javafy = self.javafy
        keep_alive = True
        while True:
            while keep_alive:
                try:
                    print('Accepting new data.')
                    rx = await s.recv()
                    self.received_params += 1
                    print(rx)
                    rx = javafy(rx, load_data=True)
                    if rx['type'] == 'request':
                        pass
                    elif rx['type'] == 'submit':
                        cauldron.get_state_dicts()
                except Exception as details:
                    print(*details)
                with THREAD_LOCK:
                    if not self.accepting_data:
                        keep_alive = False

    async def wsc(self, websocket, path):
        """Web Socket Client for distributed learning."""
        print(path)
        cauldron = self.cauldron
        javafy = self.javafy
        keep_alive = True
        while True:
            while keep_alive:
                try:
                    tx = javafy({'type': 'request'}, load_data=False)
                    tx = await s.send(tx)
                    print(tx)
                except Exception as details:
                    print(*details)
                with THREAD_LOCK:
                    if not self.sending_data:
                        keep_alive = False


def main():
    ivy_dispatcher
    return InfinitusVigilantis()
