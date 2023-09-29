"""
Market Forecasting
"""
import asyncio
import hashlib
import logging
import secrets
import socket
import time
import toga
import torch
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
LOG_NAME = f'{time.time()}.log'
LOG_PATH = abspath(path.join(ROOT_PATH, 'logs', LOG_NAME))
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
            if 'host.addr' in keys:
                host_addr = settings['host.addr']
            else:
                host_addr = 'localhost'
            if 'host.port' in keys:
                host_port = settings['host.port']
            else:
                host_port = '33333'
            address = (host_addr, host_port)
            if 'run.server' in keys:
                if settings['run.server'] == 1:
                    if host_proc is None:
                        logging.warning('dispatching server process...')
                        host_proc = dispatcher(
                            __run_server__,
                            args=(address,),
                            ftype='process',
                        )
                else:
                    if host_proc is not None:
                        logging.warning('killing server process...')
                        host_proc.kill()
                    host_proc = None
            if 'run.study' in keys:
                if settings['run.study'] == 1:
                    if study_proc is None:
                        logging.warning('dispatching study process...')
                        study_proc = dispatcher(
                            __study__,
                            args=(address,),
                            ftype='process',
                        )
                else:
                    if study_proc is not None:
                        logging.warning('killing study process...')
                        study_proc.kill()
                    study_proc = None
            if 'teapot.sleep' in keys:
                teapot_sleep = settings['teapot.sleep']
        await asyncio.sleep(teapot_sleep)


async def __start_server__(address, *args, **kwargs):
    """
    TCP/IP Server for distributed learning.
    """
    _join = path.join
    root_path = __pathing__()[0]
    state_path = abspath(_join(root_path, 'cauldron', '{}.{}.state'))
    PATHING = (
        abspath(_join(root_path,'cauldron','cauldron.state')),
        abspath(_join(root_path,'candelabrum','candelabrum.candles')),
        abspath(_join(root_path,'candelabrum','candelabrum.features')),
        abspath(_join(root_path,'candelabrum','candelabrum.symbols')),
    )
    REQUEST_HEADERS = (
        b'10001000',
        b'10000100',
        b'10000010',
        b'10000001',
    )
    def __open_connection__(client_socket, address):
        logging.info(f'{address}: connection open')
        with client_socket as connection:
            connection.sendall(b'11111111')
            while True:
                try:
                    data = connection.recv(4096)
                    if not data or len(data) < 9:
                        break
                    data_header = data[:8]
                    if data_header == b'00010001':
                        nbytes = int(data[8:].decode())
                        consumed_bytes = 0
                        state_parts = list()
                        connection.sendall(b'00010010')
                        while consumed_bytes < nbytes:
                            chunk = connection.recv(4096)
                            if not chunk:
                                break
                            state_parts.append(chunk)
                            consumed_bytes += len(chunk)
                        state = b''.join(state_parts)
                        if len(state) == nbytes:
                            connection.sendall(b'00010100')
                        else:
                            break
                    elif data_header == b'00011000':
                        client_hash = data[8:].decode()
                        server_hash = hashlib.sha512(state).digest()
                        if client_hash == server_hash:
                            token = secrets.token_urlsafe(8)
                            real_path = state_path.format(token, time.time())
                            with open(real_path, 'wb+') as state_file:
                                state_file.write(state)
                            connection.sendall(b'00000000')
                            logging.info(f'{address}: saved {real_path}')
                        break
                    elif data_header in REQUEST_HEADERS:
                        header_index = REQUEST_HEADERS.index(data_header)
                        real_path = PATHING[header_index]
                        with open(real_path, 'rb') as requested_file:
                            binary_file = requested_file.read()
                        nbytes = bytes(str(len(binary_file)), 'utf-8')
                        connection.sendall(b'00010001' + nbytes)
                        while True:
                            message = connection.recv(4096)
                            if not message:
                                break
                            if message == b'00010010':
                                connection.sendall(binary_file)
                            elif message == b'00010100':
                                file_hash = hashlib.sha512(binary_file).digest()
                                connection.sendall(b'00011000' + file_hash)
                            elif message == b'00000000':
                                nbytes = 0
                                binary_file = None
                                logging.info(f'{address}: got {real_path}')
                                break
                except Exception as details:
                    logging.error(f'{address}: {repr(details)}')
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


def __run_server__(address, *args, **kwargs):
    """
    Create the server thread.
    """
    asyncio.run(__start_server__(address), debug=True)
    return False


def __update_network__(address, *args, **kwargs):
    """
    Get current cauldron state and candelabrum files from server.
    """
    _join = path.join
    root_path = __pathing__()[0]
    PATHING = (
        abspath(_join(root_path,'cauldron','cauldron.state')),
        abspath(_join(root_path,'candelabrum','candelabrum.candles')),
        abspath(_join(root_path,'candelabrum','candelabrum.features')),
        abspath(_join(root_path,'candelabrum','candelabrum.symbols')),
    )
    REQUEST_HEADERS = (
        b'10001000',
        b'10000100',
        b'10000010',
        b'10000001',
    )
    file_path = ''
    with socket.create_connection(address) as connection:
        for request_header in REQUEST_HEADERS:
            connection.sendall(request_header)
            while True:
                message = connection.recv(4096)
                if not message:
                    break
                if len(message) < 9:
                    break
                data_header = data[:8]
                if data_header == b'00010001':
                    file_path = PATHING[REQUEST_HEADERS.index(request_header)]
                    nbytes = int(data[8:].decode())
                    consumed_bytes = 0
                    file_parts = list()
                    connection.sendall(b'00010010')
                    while consumed_bytes < nbytes:
                        chunk = connection.recv(4096)
                        if not chunk:
                            break
                        file_parts.append(chunk)
                        consumed_bytes += len(chunk)
                    binary_file = b''.join(file_parts)
                    if len(binary_file) == nbytes:
                        connection.sendall(b'00010100')
                    else:
                        break
                elif data_header == b'00011000':
                    server_hash = data[8:].decode()
                    client_hash = hashlib.sha512(binary_file).digest()
                    if client_hash == server_hash:
                        with open(file_path, 'wb+') as local_file:
                            local_file.write(binary_file)
                        connection.sendall(b'00000000')
                    break


def __study__(address, *args, **kwargs):
    """
    Cauldronic machine learning.
    """
    __update_network__(address)
    cauldron = ivy_cauldron.Cauldron()
    state_path = cauldron.state_path
    n_depth = 9
    hours = 0.5
    checkpoint = 1000
    loops = 0
    while True:
        loops += 1
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
                    message = connection.recv(4096)
                    if not message:
                        break
                    elif message == b'11111111':
                        connection.sendall(b'00010001' + nbytes)
                    elif message == b'00010010':
                        connection.sendall(state)
                    elif message == b'00010100':
                        file_hash = hashlib.sha512(state).digest()
                        connection.sendall(b'00011000' + file_hash)
                    elif message == b'00000000':
                        state = None
                        break
        except Exception as details:
            logging.error(repr(details))


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

    def startup(self):
        """
        Construct and show the Toga application.
        """
        Box = toga.Box
        Divider = toga.Divider
        HORIZONTAL = Divider.HORIZONTAL
        VERTICAL = Divider.VERTICAL
        WINDOW = Group.WINDOW
        self.log_name = LOG_NAME
        self.log_path = LOG_PATH
        self.log_change = 0
        root_path, settings_path = __pathing__()
        if path.exists(settings_path):
            self.settings = javafy.load(file_path=settings_path)
        else:
            self.settings = {
                'host.addr': 'localhost',
                'host.port': '33333',
                'run.server': 0,
                'run.study': 0,
                'teapot.sleep': 0.05,
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
