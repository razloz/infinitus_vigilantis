"""
Market Forecasting
"""
import asyncio
import logging
import time
import toga
import torch
from io import BytesIO
from os import path
from os.path import abspath, dirname, getmtime, realpath
from toga import Icon
from toga.command import Command, Group
from toga.style import Pack
from toga.style.pack import COLUMN, ROW
from websockets.server import serve
from websockets.sync.client import connect
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
                        host_proc = dispatcher(__run_serve__, ftype='process')
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


async def __wss__(websocket, *args, **kwargs):
    """
    Web Socket Server for distributed learning.
    """
    #_javafy = icy.Javafy()
    #buffer_io = BytesIO()
    #cauldron.save_state(None, to_buffer=True, buffer_io=buffer_io)
    torch_save = torch.save
    cauldron = ivy_cauldron.Cauldron()
    root_path, settings_path = __pathing__()
    state_path = abspath(path.join(root_path, 'cauldron', '{}.state_dict'))
    ts = time.time()
    async for rx in websocket:
        ts = time.time()
        logging.debug(f'{ts} rx type: {type(rx)}')
        if not type(rx) == str:
            logging.debug('received non-string data from client')
            continue
        if len(rx) < 10:
            logging.debug('received malformed data from client')
            continue
        logging.debug(f'rx len: {len(rx)}')
        message_type = rx[0:10]
        logging.debug(f'message_header: {message_header}')
        if message_type == b'^∙^,01,^∙^':
            message_data = rx[2:]
            logging.debug('got state dicts push request from client')
            logging.debug(f'type: {type(message_data)}')
            torch_save(message_data, state_path)
            logging.debug('saved torch state')
        elif message_type == b'^∙^,02,^∙^':
            logging.debug('got state dicts pull request from client')
        elif message_type == b'^∙^,03,^∙^':
            logging.debug('got candelabrum pull request from client')


async def __serve__(*args, serve_addr='localhost', serve_port='3333', **kwargs):
    """
    Server for collecting state_dict data.
    """
    logging.debug('Starting __wss__')
    async with serve(__wss__, serve_addr, serve_port):
        await asyncio.Future()


def __run_serve__(*args, **kwargs):
    """
    Create the server thread.
    """
    logging.debug('Starting __serve__')
    asyncio.run(__serve__(), debug=True)
    return False


def __push_state_dicts__(cauldron, host_addr='localhost', host_port='3333'):
    """
    Send state_dict to the server.
    """
    buffer_io = BytesIO()
    cauldron.save_state(None, to_buffer=True, buffer_io=buffer_io)
    state_dict = b'^∙^,01,^∙^' + buffer_io.getvalue()
    uri = f'ws://{host_addr}:{host_port}'
    try:
        with connect(uri, max_size=1e13) as websocket:
            logging.debug('pushing state dicts to server')
            websocket.send(state_dict)
    except Exception as details:
        logging.error(details)
    finally:
        return False


def __pull_state_dicts__(host_addr='localhost', host_port='3333'):
    """
    Send state_dict to the server.
    """
    try:
        with connect(f'ws://{host_addr}:{host_port}') as websocket:
            logging.debug('pulling state dicts from server')
            websocket.send(b'^∙^,02,^∙^')
            state_dicts = websocket.recv()
            logging.debug(f'state_dict type: {type(state_dicts)}')
    except Exception as details:
        logging.error(details)
    finally:
        return False


def __pull_candelabrum__(host_addr='localhost', host_port='3333'):
    """
    Get candelabrum from the server.
    """
    try:
        logging.info('Pulling candelabrum from server.')
        with connect(f'ws://{host_addr}:{host_port}') as websocket:
            logging.debug(f'pulling candelabrum from server')
            websocket.send(b'^∙^,03,^∙^')
            candelabrum = websocket.recv()
            logging.debug(f'candelabrum: {candelabrum.shape}')
    except Exception as details:
        logging.error(details)
    finally:
        return False


def __study__(*args, **kwargs):
    """
    Cauldronic machine learning.
    """
    cauldron = ivy_cauldron.Cauldron()
    n_depth = 9
    hours = 1e-5
    checkpoint = 1
    msg = 'Starting study with n_depth of {}, '
    msg += 'for {} hours, and a checkpoint every {} iterations.'
    msg = msg.format(n_depth, hours, checkpoint)
    while True:
        try:
            logging.debug(msg)
            cauldron.train_network(
                n_depth=n_depth,
                hours=hours,
                checkpoint=checkpoint,
            )
            logging.debug('Pushing state_dicts to server.')
            __push_state_dicts__(cauldron)
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
                'teapot.sleep': 0.333,
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
