from toga import Box, Button, Divider, Label, TextInput
from toga.style import Pack
from toga.style.pack import COLUMN, ROW


class ConfigScreen(Box):
    def __init__(self, root_app):
        self.root_app = root_app
        HORIZONTAL = Divider.HORIZONTAL
        VERTICAL = Divider.VERTICAL
        self.root_app = root_app
        self.address_input = TextInput(
            style=Pack(
                direction=ROW,
                flex=1,
                padding=5,
            ),
        )
        self.address_input.value = root_app.settings['host.addr']
        self.port_input = TextInput(
            style=Pack(
                direction=ROW,
                flex=1,
                padding=5,
            ),
        )
        self.port_input.value = root_app.settings['host.port']
        self.host_address = Box(
            children=[
                Label('Host Address:'),
                self.address_input,
            ],
            style=Pack(
                direction=ROW,
                flex=1,
                padding=5,
            ),
        )
        self.host_port = Box(
            children=[
                Label('Host Port:'),
                self.port_input,
            ],
            style=Pack(
                direction=ROW,
                flex=1,
                padding=5,
            ),
        )
        self.save_button = Button(
            'Save Config',
            on_press=self.__save_config__,
        )
        super().__init__(
            id='config',
            children=[
                self.host_address,
                self.host_port,
                Divider(direction=HORIZONTAL, style=Pack(padding=5)),
                self.save_button,
            ],
            style=Pack(direction=COLUMN),
        )
    async def __save_config__(self, *args, **kwargs):
        self.root_app.update_param('host.addr', str(self.address_input.value))
        self.root_app.update_param('host.port', int(self.port_input.value))
