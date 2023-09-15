from toga import Box
from toga.style import Pack
from toga.style.pack import COLUMN, ROW


class ConfigScreen(Box):
    def __init__(self, root_app):
        self.root_app = root_app
        super().__init__(
            id='config',
            children=[],
            style=Pack(direction=COLUMN),
        )
