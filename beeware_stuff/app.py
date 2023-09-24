"""
Market Forecasting
"""
import asyncio
import logging
import secrets
import time
import toga
import torch
import socket
from io import BytesIO
from os import path
from os.path import abspath, dirname, getmtime, realpath
from toga import Icon
from toga.command import Command, Group
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
import infinitus_vigilantis.source.ivy_commons as icy
import infinitus_vigilantis.source.ivy_cauldron as ivy_cauldron
from infinitus_vigilantis.screens.about import AboutScreen
from infinitus_vigilantis.screens.config import ConfigScreen
from infinitus_vigilantis.screens.home import HomeScreen
ROOT_PATH = dirname(realpath(__file__))
SETTINGS_PATH = abspath(path.join(ROOT_PATH, 'resources', 'ivy.settings'))
LOG_PATH = abspath(path.join(ROOT_PATH, 'logs', f'{time.time()}.log'))
logging.getLogger('asyncio').setLevel(logging.DEBUG)
logging.basicConfig(
    filename=LOG_PATH,
    encoding='utf-8',
    level=logging.DEBUG,
)
javafy = icy.Javafy()


def __pathing__(*args, **kwargs):
    """
    Returns real file path for source root and settings.
    """
    ROOT_PATH = dirname(realpath(__file__))
    SETTINGS_PATH = abspath(path.join(ROOT_PATH, 'resources', 'ivy.settings'))
    return (ROOT_PATH, SETTINGS_PATH)


async def __teapot__(*args, **kwargs):
    """
    Brew a pot of tea.
    """
    dispatcher = icy.ivy_dispatcher
    last_change = 0
    host_proc = None
    study_proc = None
    root_path, settings_path = __pathing__()
    async def brew(last_change):
        change = getmtime(settings_path)
        if change > last_change + 1:
            settings = javafy.load(file_path=settings_path)
            return (settings, change)
        return None
    while True:
        tea = await brew(last_change)
        teapot_sleep = 1
        if tea is not None:
            settings, last_change = tea
            keys = list(settings.keys())
            if 'run.server' in keys:
                if settings['run.server'] == 1:
                    if host_proc is None:
                        logging.warning('dispatching server process...')
                        host_proc = dispatcher(__run_server__, ftype='process')
                else:
                    if host_proc is not None:
                        logging.warning('killing server process...')
                        host_proc.kill()
                    host_proc = None
            if 'run.study' in keys:
                if settings['run.study'] == 1:
                    if study_proc is None:
                        logging.warning('dispatching study process...')
                        study_proc = dispatcher(__study__, ftype='process')
                else:
                    if study_proc is not None:
                        logging.warning('killing study process...')
                        study_proc.kill()
                    study_proc = None
            if 'teapot.sleep' in keys:
                teapot_sleep = settings['teapot.sleep']
        await asyncio.sleep(teapot_sleep)


async def __start_server__(*args, **kwargs):
    """
    TCP/IP Server for distributed learning.
    """
    address = ('localhost', 33333)
    root_path, settings_path = __pathing__()
    state_path = abspath(path.join(root_path, 'cauldron', '{}.{}.state'))
    def __open_connection__(client_socket, address):
        logging.info(f'{address}: connection open')
        with client_socket as connection:
            connection.send(b'From which old one were you spawned?')
            while True:
                try:
                    data = connection.recv(1024)
                    if not data or len(data) < 9:
                        break
                    data_header = data[:8]
                    if data_header == b'00010001':
                        nbytes = int(data[8:].decode())
                        token = secrets.token_urlsafe(8)
                        real_path = state_path.format(token, time.time())
                        consumed_bytes = 0
                        state_parts = list()
                        connection.send(b'Feed Me!')
                        while consumed_bytes < nbytes:
                            chunk = connection.recv(4096)
                            if not chunk:
                                break
                            state_parts.append(chunk)
                            consumed_bytes += len(chunk)
                        state = b''.join(state_parts)
                        if len(state) == nbytes:
                            with open(real_path, 'wb+') as state_file:
                                state_file.write(state)
                            logging.info(f'{address}: file transfer complete.')
                            connection.send(b'My power levels are over 9000!')
                        break
                except Exception as details:
                    break
        return False
    server_socket = socket.create_server(
        address,
        family=socket.AF_INET,
        backlog=1000,
        reuse_port=True,
    )
    dispatcher = icy.ivy_dispatcher
    with server_socket as server:
        while True:
            connection, address = server.accept()
            dispatcher(__open_connection__, args=[connection, address])


def __run_server__(*args, **kwargs):
    """
    Create the server thread.
    """
    logging.info('*** starting __start_server__')
    asyncio.run(__start_server__(), debug=True)
    return False


def __study__(*args, **kwargs):
    """
    Cauldronic machine learning.
    """
    #getsize = path.getsize
    cauldron = ivy_cauldron.Cauldron()
    state_path = cauldron.state_path
    n_depth = 9
    hours = 1e-5
    checkpoint = 1
    address = ('localhost', 33333)
    msg = 'Starting study with n_depth of {}, '
    msg += 'for {} hours, and a checkpoint every {} iterations.'
    msg = msg.format(n_depth, hours, checkpoint)
    logging.debug(msg)
    loops = 0
    while True:
        loops += 1
        logging.info(f'Starting loop #{loops}')
        try:
            cauldron.train_network(
                n_depth=n_depth,
                hours=hours,
                checkpoint=checkpoint,
            )
            with open(state_path, 'rb') as state_file:
                state = state_file.read()
            nbytes = bytes(str(len(state)), 'utf-8')
            with socket.create_connection(address) as connection:
                while True:
                    message = connection.recv(1024)
                    if not message:
                        break
                    elif message == b'From which old one were you spawned?':
                        connection.send(b'00010001' + nbytes)
                    elif message == b'Feed Me!':
                        connection.sendall(state)
                    elif message == b'My power levels are over 9000!':
                        connection.shutdown(socket.SHUT_RDWR)
                        break
        except Exception as details:
            logging.error(details)


class InfinitusVigilantis(toga.App):
    """
    Graphical interface for the Infinitus Vigilantis Python application.
    """

    def __goto_about__(self, *ignore):
        """
        Switch view to the about screen.
        """
        self.main_window.content = self.viewports['about']

    def __goto_config__(self, *ignore):
        """
        Switch view to the config screen.
        """
        self.main_window.content = self.viewports['config']

    def __goto_home__(self, *ignore):
        """
        Switch view to the home screen.
        """
        self.main_window.content = self.viewports['home']

    async def __check_log__(self, *args, **kwargs):
        log_path = self.log_path
        while True:
            last_change = self.log_change
            change = getmtime(log_path)
            if change > last_change:
                with open(log_path, 'r') as log_file:
                    log_data = log_file.read()
                self.log_change = change
                self.viewports['home'].messages.value = log_data
                self.viewports['home'].messages.value += '\n'
                self.viewports['home'].messages.scroll_to_bottom()
            await asyncio.sleep(0.5)

    def startup(self):
        """
        Construct and show the Toga application.
        """
        Box = toga.Box
        Divider = toga.Divider
        HORIZONTAL = Divider.HORIZONTAL
        VERTICAL = Divider.VERTICAL
        WINDOW = Group.WINDOW
        self.log_path = LOG_PATH
        self.log_change = 0
        root_path, settings_path = __pathing__()
        if path.exists(settings_path):
            self.settings = javafy.load(file_path=settings_path)
        else:
            self.settings = {
                'client.host.addr': 'localhost',
                'client.host.port': '3333',
                'server.host.addr': 'localhost',
                'server.host.port': '3333',
                'run.server': 0,
                'run.study': 0,
                'teapot.sleep': 0.0333,
            }
            javafy.save(data=self.settings, file_path=settings_path)
        self.viewports = {
            'about': AboutScreen(self),
            'config': ConfigScreen(self),
            'home': HomeScreen(self),
        }
        self._action_about_ = Command(
            self.__goto_about__,
            'About',
            tooltip='About the application.',
            icon='resources/icons/about.png',
            group=WINDOW,
            section=0,
            order=2,
        )
        self._action_config_ = Command(
            self.__goto_config__,
            'Config',
            tooltip='Configuration for the application.',
            icon='resources/icons/config.png',
            group=WINDOW,
            section=0,
            order=1,
        )
        self._action_home_ = Command(
            self.__goto_home__,
            'IVy',
            tooltip='Main page of the application.',
            icon='resources/icons/ivy.png',
            group=WINDOW,
            section=0,
            order=0,
        )
        self.main_window = toga.MainWindow(title=self.formal_name)
        self.main_window.toolbar.add(
            self._action_about_,
            self._action_config_,
            self._action_home_,
        )
        self.main_window.content = self.viewports['home']
        self.root_path = root_path
        self.settings_path = settings_path
        self.main_window.show()
        self.add_background_task(__teapot__)
        self.add_background_task(self.__check_log__)

    def update_param(self, param, value):
        """
        Set param to value in settings and save.
        """
        self.settings[param] = value
        javafy.save(data=self.settings, file_path=self.settings_path)


def main():
    """
    Launch the Toga application.
    """
    return InfinitusVigilantis()
