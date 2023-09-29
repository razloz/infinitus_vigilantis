from toga import Box, Button, Divider, Label
from toga.style import Pack
from toga.style.pack import COLUMN, ROW


class HomeScreen(Box):
    def __init__(self, root_app):
        HORIZONTAL = Divider.HORIZONTAL
        VERTICAL = Divider.VERTICAL
        self.root_app = root_app
        if root_app.settings['run.study'] == 1:
            state_text = 'Stop Client'
        else:
            state_text = 'Start Client'
        self.client_state = Button(
            state_text,
            on_press=self.__start_client__,
        )
        if root_app.settings['run.server'] == 1:
            state_text = 'Stop Server'
        else:
            state_text = 'Start Server'
        self.server_state = Button(
            state_text,
            on_press=self.__start_server__,
        )
        self.status_text = Label(
            'Welcome to the Infinitus Vigilantis application.',
            style=Pack(
                direction=ROW,
                flex=1,
                padding=5,
            ),
        )
        super().__init__(
            id='home',
            children=[
                self.client_state,
                self.server_state,
                Divider(direction=HORIZONTAL, style=Pack(padding=5)),
                self.status_text,
            ],
            style=Pack(direction=COLUMN),
        )

    async def __start_client__(self, button):
        """
        Start learning loop.
        """
        if button.text == 'Start Client':
            self.root_app.update_param('run.study', 1)
            button.text = 'Stop Client'
            self.status_text.text = 'Started study loop.'
        elif button.text == 'Stop Client':
            self.root_app.update_param('run.study', 0)
            button.text = 'Start Client'
            self.status_text.text = 'Stopped study loop.'
        else:
            button.text = '*beep*boop*beep*'

    async def __start_server__(self, button):
        """
        Start server loop.
        """
        if button.text == 'Start Server':
            self.root_app.update_param('run.server', 1)
            button.text = 'Stop Server'
            self.status_text.text = 'Started server loop.'
        elif button.text == 'Stop Server':
            self.root_app.update_param('run.server', 0)
            button.text = 'Start Server'
            self.status_text.text = 'Stopped server loop.'
        else:
            button.text = '*beep*boop*beep*'
