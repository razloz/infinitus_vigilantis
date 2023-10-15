"""Three blind mice to predict the future."""
import asyncio
import hashlib
import json
import logging
import pandas
import pickle
import secrets
import socket
import time
import torch
import source.ivy_commons as icy
import source.ivy_cauldron as ivy_cauldron
import source.ivy_https as ivy_https
from os import path, listdir
from os.path import abspath, dirname, getmtime, realpath
from source.ivy_cartography import cartography
ROOT_PATH = dirname(realpath(__file__))
CAULDRON_PATH = abspath(path.join(ROOT_PATH, '..', 'cauldron'))
LOG_PATH = abspath(path.join(ROOT_PATH, '..', 'logs', f'{time.time()}.log'))
STATE_PATH = abspath(path.join(ROOT_PATH, '..', 'cauldron', '{}.{}.state'))
SETTINGS_PATH = abspath(path.join(ROOT_PATH, '..', 'resources', 'ivy.settings'))
HTTPS_PATH = abspath(path.join(ROOT_PATH, '..', 'https'))
PATHING = (
    abspath(path.join(ROOT_PATH, '..', 'cauldron', 'cauldron.state')),
    abspath(path.join(ROOT_PATH, '..', 'candelabrum', 'candelabrum.candles')),
    abspath(path.join(ROOT_PATH, '..', 'candelabrum', 'candelabrum.features')),
    abspath(path.join(ROOT_PATH, '..', 'candelabrum', 'candelabrum.symbols')),
)
REQUEST_HEADERS = (
    b'10001000',
    b'10000100',
    b'10000010',
    b'10000001',
)
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2023, Daniel Ward'
__license__ = 'GPL v3'


def chit_chat(msg, log_level=0, log_msg=True, print_msg=True, timestamp=True):
    """
    Talk with the mice over a nice cup of tea.
    """
    msg = repr(msg) if not type(msg) == str else msg
    if timestamp:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        msg = f'{timestamp} {msg}'
    if print_msg:
        print(msg)
    if log_msg:
        if log_level == 0:
            logging.info(msg)
        elif log_level == 1:
            logging.warning(msg)
        elif log_level == 2:
            logging.error(msg)
        else:
            logging.debug(msg)


async def __start_server__(address, *args, **kwargs):
    """
    TCP/IP Server for distributed learning.
    """
    update_key = secrets.token_urlsafe(32)
    chit_chat(f'Server: update_key is {update_key}')
    update_key = bytes(update_key, 'utf-8')
    def __open_connection__(client_socket, address):
        with client_socket as connection:
            chit_chat(f'{address[0]}: connection open')
            connection.sendall(b'11111111' + update_key)
            while True:
                try:
                    data = connection.recv(4096)
                    if not data or len(data) < 8:
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
                        client_hash = data[8:]
                        server_hash = hashlib.sha512(state).digest()
                        if client_hash == server_hash:
                            token = secrets.token_urlsafe(8)
                            real_path = STATE_PATH.format(token, time.time())
                            with open(real_path, 'wb+') as state_file:
                                state_file.write(state)
                            connection.sendall(b'00000000')
                            chit_chat(f'{address[0]}: state transfer complete')
                            chit_chat(f'{address[0]}: saved to {real_path}')
                        break
                    elif data_header in REQUEST_HEADERS:
                        header_index = REQUEST_HEADERS.index(data_header)
                        real_path = PATHING[header_index]
                        with open(real_path, 'rb') as requested_file:
                            binary_file = requested_file.read()
                        nbytes = bytes(str(len(binary_file)), 'utf-8')
                        connection.sendall(b'00010001' + nbytes)
                        while True:
                            data = connection.recv(4096)
                            if not data:
                                break
                            if data == b'00010010':
                                connection.sendall(binary_file)
                            elif data == b'00010100':
                                file_hash = hashlib.sha512(binary_file).digest()
                                connection.sendall(b'00011000' + file_hash)
                            elif data == b'00000000':
                                nbytes = 0
                                binary_file = None
                                connection.sendall(b'11111111' + update_key)
                                chit_chat(f'{address[0]}: sent updated file')
                                break
                    else:
                        chit_chat(f'{address[0]}: got malformed data header')
                        break
                except Exception as details:
                    chit_chat(f'{address[0]}: {repr(details)}')
                    break
        chit_chat(f'{address[0]}: connection closed')
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


def __update_network__(address, *args, **kwargs):
    """
    Get current cauldron state and candelabrum files from server.
    """
    update_key = ''
    with socket.create_connection(address) as connection:
        chit_chat(f'{address[0]}: connected... updating...')
        for header_id, request_header in enumerate(REQUEST_HEADERS):
            chit_chat(f'{address[0]}: requesting file #{header_id + 1}')
            try:
                while True:
                    data = connection.recv(4096)
                    if not data or len(data) < 8:
                        break
                    data_header = data[:8]
                    if data_header == b'11111111':
                        update_key = data[8:]
                        connection.sendall(request_header)
                    elif data_header == b'00010001':
                        file_path = PATHING[header_id]
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
                        server_hash = data[8:]
                        client_hash = hashlib.sha512(binary_file).digest()
                        if client_hash == server_hash:
                            with open(file_path, 'wb+') as local_file:
                                local_file.write(binary_file)
                            connection.sendall(b'00000000')
                            chit_chat(f'{address[0]}: got file {file_path}')
                        else:
                            chit_chat(f'{address[0]}: file hash mismatch')
                        break
                    else:
                        chit_chat(f'{address[0]}: got malformed data header')
                        break
            except Exception as details:
                chit_chat(f'{address[0]}: {repr(details)}')
                break
        chit_chat(f'{address[0]}: exited update loop')
    chit_chat(f'{address[0]}: connection closed')
    return update_key


def __study__(address, update_key, n_depth=9, hours=0.5, checkpoint=1000):
    """
    Cauldronic machine learning.
    """
    cauldron = ivy_cauldron.Cauldron()
    state_path = cauldron.state_path
    loops = 0
    while True:
        new_key = update_key
        try:
            loops += 1
            chit_chat(f'ThreeBlindMice: Starting loop #{loops}')
            cauldron.train_network(
                n_depth=n_depth,
                hours=hours,
                checkpoint=checkpoint,
            )
            with open(state_path, 'rb') as state_file:
                state = state_file.read()
            nbytes = bytes(str(len(state)), 'utf-8')
            with socket.create_connection(address) as connection:
                chit_chat(f'{address[0]}: connected... pushing state...')
                while True:
                    data = connection.recv(4096)
                    if not data or len(data) < 8:
                        break
                    data_header = data[:8]
                    if data_header == b'11111111':
                        new_key = data[8:]
                        connection.sendall(b'00010001' + nbytes)
                    elif data_header == b'00010010':
                        connection.sendall(state)
                    elif data_header == b'00010100':
                        file_hash = hashlib.sha512(state).digest()
                        connection.sendall(b'00011000' + file_hash)
                    elif data_header == b'00000000':
                        chit_chat(f'{address[0]}: state transfer complete')
                        state = None
                        break
            chit_chat(f'{address[0]}: connection closed')
        except Exception as details:
            chit_chat(f'{address[0]}: {repr(details)}')
        if new_key != update_key:
            chit_chat(f'{address[0]}: update available')
            update_key = __update_network__(address)
            cauldron = ivy_cauldron.Cauldron()


def __merge_states__(*args, **kwargs):
    """
    Load client cauldron states and merge with server state.
    """
    from os import remove as __remove_file__
    TENSOR = torch.Tensor
    cauldron_folder = CAULDRON_PATH
    cauldron_files = listdir(cauldron_folder)
    state_count = sum([1 if p[-6:] == '.state' else 0 for p in cauldron_files])
    def __merge_params__(old_state, new_state, finalize=False):
        for key, value in new_state.items():
            if type(value) == TENSOR and key in old_state:
                if finalize:
                    old_state[key] = (old_state[key] + value) / 2
                else:
                    old_state[key] += value * state_count
        return old_state
    __path_exists__ = path.exists
    __path_join__ = path.join
    state_path = str(PATHING[0])
    base_cauldron = ivy_cauldron.Cauldron()
    proxy_cauldron = ivy_cauldron.Cauldron()
    chit_chat('ThreeBlindMice: loading server state')
    proxy_cauldron.load_state(state_path=state_path)
    for file_name in cauldron_files:
        if file_name == 'cauldron.state':
            continue
        if '.state' in file_name and file_name[-6:] == '.state':
            chit_chat(f'ThreeBlindMice: merging {file_name}')
            file_path = __path_join__(cauldron_folder, file_name)
            base_cauldron.load_state(state_path=file_path)
            proxy_cauldron.load_state_dict(
                __merge_params__(
                    proxy_cauldron.state_dict(),
                    base_cauldron.state_dict(),
                )
            )
            proxy_cauldron.network.load_state_dict(
                __merge_params__(
                    proxy_cauldron.network.state_dict(),
                    base_cauldron.network.state_dict(),
                )
            )
            proxy_cauldron.optimizer.load_state_dict(
                __merge_params__(
                    proxy_cauldron.optimizer.state_dict(),
                    base_cauldron.optimizer.state_dict(),
                )
            )
    chit_chat('ThreeBlindMice: finalizing merge')
    base_cauldron = ivy_cauldron.Cauldron()
    base_cauldron.load_state(state_path=state_path)
    base_cauldron.load_state_dict(
        __merge_params__(
            base_cauldron.state_dict(),
            proxy_cauldron.state_dict(),
            finalize=True,
        )
    )
    base_cauldron.network.load_state_dict(
        __merge_params__(
            base_cauldron.network.state_dict(),
            proxy_cauldron.network.state_dict(),
            finalize=True,
        )
    )
    base_cauldron.optimizer.load_state_dict(
        __merge_params__(
            base_cauldron.optimizer.state_dict(),
            proxy_cauldron.optimizer.state_dict(),
            finalize=True,
        )
    )
    base_cauldron.save_state(state_path)
    chit_chat(f'ThreeBlindMice: removing client state files')
    for file_name in cauldron_files:
        if file_name == 'cauldron.state':
            continue
        if '.state' in file_name and file_name[-6:] == '.state':
            chit_chat(f'ThreeBlindMice: removing {file_name}')
            file_path = __path_join__(cauldron_folder, file_name)
            if __path_exists__(file_path):
                __remove_file__(file_path)
    chit_chat('ThreeBlindMice: merge complete')


class ThreeBlindMice():
    """
    Let the daughters of necessity shape the candles of the future.
    """
    def __init__(self):
        """
        Beckon the Norn.
        """
        logging.getLogger('asyncio').setLevel(logging.DEBUG)
        logging.basicConfig(
            filename=LOG_PATH,
            encoding='utf-8',
            level=logging.DEBUG,
        )
        javafy = icy.Javafy()
        if path.exists(SETTINGS_PATH):
            self.settings = javafy.load(file_path=SETTINGS_PATH)
        else:
            self.settings = {
                'host.addr': 'localhost',
                'host.port': '33333',
                'n_depth': '9',
                'hours': '0.5',
                'checkpoint': '1000',
            }
            javafy.save(data=self.settings, file_path=SETTINGS_PATH)
        self.host_addr = str(self.settings['host.addr'])
        self.host_port = int(self.settings['host.port'])
        self.address = (self.host_addr, self.host_port)
        chit_chat('ThreeBlindMice: beckoning the Norn')

    def merge_states(self, *args, **kwargs):
        """
        Take state files and merge values with server.
        """
        chit_chat(f'ThreeBlindMice: merging client states with server')
        try:
            __merge_states__()
        except Exception as details:
            chit_chat(f'ThreeBlindMice: {repr(details)}')

    def start_learning(self, *args, **kwargs):
        """
        Create study thread.
        """
        n_depth = int(self.settings['n_depth'])
        hours = float(self.settings['hours'])
        checkpoint = int(self.settings['checkpoint'])
        chit_chat('ThreeBlindMice: updating neural network')
        update_key = __update_network__(self.address)
        chit_chat('ThreeBlindMice: creating study thread')
        __study__(
            self.address,
            update_key,
            n_depth=n_depth,
            hours=hours,
            checkpoint=checkpoint,
        )

    def start_serving(self, *args, debug=False, **kwargs):
        """
        Create server thread.
        """
        chit_chat('ThreeBlindMice: building website')
        self.build_https()
        chit_chat('ThreeBlindMice: starting server')
        asyncio.run(__start_server__(self.address), debug=debug)

    def build_https(self):
        from pandas import read_csv
        candles_path = abspath(path.join(ROOT_PATH, '..', 'candelabrum'))
        https_path = HTTPS_PATH
        charts_path = abspath(path.join(https_path, 'charts'))
        cauldron = ivy_cauldron.Cauldron()
        chit_chat('build_https: validating neural network')
        cauldron.validate_network()
        chit_chat('build_https: inscribing sigils')
        metrics = cauldron.inscribe_sigil(charts_path)
        symbols = cauldron.symbols
        candelabrum = cauldron.candelabrum
        with open(PATHING[2], 'rb') as features_file:
            features = pickle.load(features_file)
        chit_chat('build_https: plotting charts')
        for symbol in symbols:
            chart_path = abspath(path.join(charts_path, f'{symbol}_market.png'))
            candles = abspath(path.join(candles_path, f'{symbol}.ivy'))
            candles = read_csv(candles)
            candles.set_index('time', inplace=True)
            cartography(symbol, candles, chart_path=chart_path, chart_size=365)
        chit_chat('build_https: building html documents')
        cabinet = ivy_https.build(symbols, features, candelabrum, metrics)
        for file_path, file_data in cabinet.items():
            file_path = abspath(path.join(https_path, file_path))
            with open(file_path, 'w+') as html_file:
                html_file.write(file_data)
