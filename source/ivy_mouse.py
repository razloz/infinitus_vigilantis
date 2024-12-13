"""Three blind mice to predict the future."""
import asyncio
import gc
import hashlib
import json
import logging
import os
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
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2024, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'gardneri'
ROOT_PATH = abspath(path.join(dirname(realpath(__file__)), '..'))
CAULDRON_PATH = path.join(ROOT_PATH, 'cauldron')
LOG_PATH = path.join(ROOT_PATH, 'logs', 'ivy_mouse.log')
STATE_PATH = path.join(ROOT_PATH, 'cauldron', '{}.{}.state')
SETTINGS_PATH = path.join(ROOT_PATH, 'resources', 'ivy.settings')
HASH_PATH = path.join(ROOT_PATH, 'resources', 'ivy.hash')
PUSHED_PATH = path.join(ROOT_PATH, 'resources', 'ivy.pushed')
HTTPS_PATH = path.join(ROOT_PATH, 'https')
PATHING = (
    path.join(ROOT_PATH, 'cauldron', 'cauldron.state'),
    path.join(ROOT_PATH, 'candelabrum', 'candelabrum.candles'),
    path.join(ROOT_PATH, 'candelabrum', 'candelabrum.features'),
    path.join(ROOT_PATH, 'candelabrum', 'candelabrum.symbols'),
)
REQUEST_HEADERS = (
    b'10001000',
    b'10000100',
    b'10000010',
    b'10000001',
)


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


def __load_pickle__(real_path):
    """
    Returns unpickled object.
    """
    file_data = ''
    if path.exists(real_path):
        with open(real_path, 'rb') as file_obj:
            file_data = pickle.load(file_obj)
    return file_data


def __save_pickle__(obj, real_path):
    """
    Save pickled object.
    """
    with open(real_path, 'wb+') as file_obj:
        pickle.dump(obj, file_obj)


def __get_file_hash__(real_path):
    """
    Returns SHA512 digest for local cauldron state.
    """
    file_hash = ''
    try:
        if path.exists(real_path):
            with open(real_path, 'rb') as file_obj:
                file_data = file_obj.read()
            file_hash = hashlib.sha512(file_data).digest()
    except Exception as details:
        chit_chat(f'\b: {repr(details)}', log_level=2)
    finally:
        return file_hash


async def __start_server__(address, *args, **kwargs):
    """
    TCP/IP Server for distributed learning.
    """
    update_key = __get_file_hash__(PATHING[0])
    chit_chat(f'\b: {address[0]} serving on port {address[1]}')
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
                    chit_chat(f'{address[0]}: {repr(details)}', log_level=2)
                    break
        chit_chat(f'{address[0]}: connection closed')
        gc.collect()
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


def __update_network__(address, last_key, *args, **kwargs):
    """
    Get current cauldron state and candelabrum files from server.
    """
    update_key = ''
    binary_file = None
    try:
        with socket.create_connection(address) as connection:
            chit_chat('\b: connected... checking for updates...')
            for header_id, request_header in enumerate(REQUEST_HEADERS):
                while True:
                    data = connection.recv(4096)
                    if not data or len(data) < 8:
                        break
                    data_header = data[:8]
                    if data_header == b'11111111':
                        update_key = data[8:]
                        if update_key == last_key:
                            chit_chat(f'\b: no update available, breaking loop')
                            return last_key
                        chit_chat(f'\b: requesting file #{header_id + 1}')
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
                            chit_chat(f'\b: got file {file_path}')
                        else:
                            chit_chat('\b: file hash mismatch')
                        break
                    else:
                        chit_chat('\b: got malformed data header')
                        break
    except Exception as details:
        chit_chat(f'\b: {repr(details)}', log_level=2)
    finally:
        del(binary_file)
        gc.collect()
        return update_key


def __push_state__(address, state_path, update_key):
    """
    Send cauldron state to the server.
    """
    new_key = update_key
    state = None
    try:
        with open(state_path, 'rb') as state_file:
            state = state_file.read()
        nbytes = bytes(str(len(state)), 'utf-8')
        with socket.create_connection(address) as connection:
            chit_chat('\b: connected... pushing state...')
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
                    chit_chat('\b: state transfer complete')
                    state = None
                    break
    except Exception as details:
        chit_chat(f'\b: {repr(details)}', log_level=2)
    finally:
        del(state)
        gc.collect()
        return new_key


def __study__(address, update_key, last_push, hours=3, checkpoint=1):
    """
    Cauldronic machine learning.
    """
    merge_states = __merge_states__
    hash_path = HASH_PATH
    pushed_path = PUSHED_PATH
    state_path = PATHING[0]
    loops = 0
    while True:
        loops += 1
        chit_chat(f'\b: starting loop #{loops}')
        merge_states()
        cauldron = ivy_cauldron.Cauldron()
        chit_chat(f'\b: training network for {hours} hours')
        cauldron = ivy_cauldron.Cauldron(debug_mode=False, client_mode=True)
        study_sec = hours * 3600
        cauldron.train_network(max_time=study_sec)
        cauldron = None
        del(cauldron)
        gc.collect()
        new_key = ''
        state_hash = __get_file_hash__(state_path)
        if state_hash != last_push:
            new_key = __push_state__(address, state_path, update_key)
            last_push = state_hash
            __save_pickle__(last_push, pushed_path)
        if '' != new_key != update_key:
            chit_chat('\b: update available')
            update_key = __update_network__(address, update_key)
            if update_key != '':
                __save_pickle__(update_key, hash_path)


def __merge_states__(*args, **kwargs):
    """
    Load client cauldron states and merge with server state.
    """
    __remove_file__ = os.remove
    TENSOR = torch.Tensor
    cauldron_folder = CAULDRON_PATH
    cauldron_files = listdir(cauldron_folder)
    state_offset = sum([1 if p[-6:] == '.state' else 0 for p in cauldron_files])
    state_offset -= 1
    if state_offset < 1:
        chit_chat('\b: no state files found, skipping merge.')
        return False
    chit_chat(f'\b: merging {state_offset} state files.')
    state_offset = 1 / (state_offset)
    def __merge_params__(old_state, new_state, finalize=False):
        for key, value in new_state.items():
            if type(value) == TENSOR and key in old_state:
                if not finalize:
                    old_state[key] += value * state_offset
                else:
                    old_state[key] = (old_state[key] + value) / 2
        return old_state
    __path_exists__ = path.exists
    __path_join__ = path.join
    state_path = str(PATHING[0])
    base_cauldron = ivy_cauldron.Cauldron()
    proxy_cauldron = ivy_cauldron.Cauldron()
    proxy_cauldron.load_state(state_path=state_path)
    for file_name in cauldron_files:
        if file_name == 'cauldron.state':
            continue
        elif '.state' in file_name and file_name[-6:] == '.state':
            file_path = __path_join__(cauldron_folder, file_name)
            base_cauldron.load_state(state_path=file_path)
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
            proxy_cauldron.decoder.load_state_dict(
                __merge_params__(
                    proxy_cauldron.decoder.state_dict(),
                    base_cauldron.decoder.state_dict(),
                )
            )
            proxy_cauldron.encoder.load_state_dict(
                __merge_params__(
                    proxy_cauldron.encoder.state_dict(),
                    base_cauldron.encoder.state_dict(),
                )
            )
            proxy_cauldron.normalizer.load_state_dict(
                __merge_params__(
                    proxy_cauldron.normalizer.state_dict(),
                    base_cauldron.normalizer.state_dict(),
                )
            )
    del(base_cauldron)
    gc.collect()
    base_cauldron = ivy_cauldron.Cauldron()
    base_cauldron.load_state(state_path=state_path)
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
    base_cauldron.decoder.load_state_dict(
        __merge_params__(
            base_cauldron.decoder.state_dict(),
            proxy_cauldron.decoder.state_dict(),
            finalize=True,
        )
    )
    base_cauldron.encoder.load_state_dict(
        __merge_params__(
            base_cauldron.encoder.state_dict(),
            proxy_cauldron.encoder.state_dict(),
            finalize=True,
        )
    )
    base_cauldron.normalizer.load_state_dict(
        __merge_params__(
            base_cauldron.normalizer.state_dict(),
            proxy_cauldron.normalizer.state_dict(),
            finalize=True,
        )
    )
    base_cauldron.save_state(state_path)
    for file_name in cauldron_files:
        if file_name == 'cauldron.state':
            continue
        if '.state' in file_name and file_name[-6:] == '.state':
            file_path = __path_join__(cauldron_folder, file_name)
            if __path_exists__(file_path):
                __remove_file__(file_path)
    del(base_cauldron)
    del(proxy_cauldron)
    gc.collect()
    chit_chat('\b: merge complete')
    return True


class ThreeBlindMice():
    """
    Let the daughters of necessity shape the candles of the future.
    """
    def __init__(self):
        """
        Beckon the Norn.
        """
        if path.exists(LOG_PATH):
            os.remove(LOG_PATH)
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
                'hours': '3',
                'checkpoint': '1',
            }
            javafy.save(data=self.settings, file_path=SETTINGS_PATH)
        self.host_addr = str(self.settings['host.addr'])
        self.host_port = int(self.settings['host.port'])
        self.address = (self.host_addr, self.host_port)
        chit_chat('\b: beckoning the Norn')

    def merge_states(self, *args, **kwargs):
        """
        Take state files and merge values with server.
        """
        try:
            __merge_states__()
        except Exception as details:
            chit_chat(f'\b: {repr(details)}', log_level=2)

    def start_learning(self, *args, **kwargs):
        """
        Create study thread.
        """
        address = self.address
        hours = float(self.settings['hours'])
        checkpoint = int(self.settings['checkpoint'])
        hash_path = HASH_PATH
        pushed_path = PUSHED_PATH
        state_path = PATHING[0]
        last_hash = __load_pickle__(hash_path)
        last_push = __load_pickle__(pushed_path)
        state_hash = __get_file_hash__(state_path)
        update_key = last_hash
        if path.exists(state_path):
            if state_hash != last_push:
                chit_chat('\b: pushing state to server')
                update_key = __push_state__(address, state_path, update_key)
                last_push = state_hash
                __save_pickle__(last_push, pushed_path)
        if update_key == '' or update_key != last_hash:
            chit_chat('\b: updating neural network')
            update_key = __update_network__(address, update_key)
            if update_key != '':
                __save_pickle__(update_key, hash_path)
        chit_chat('\b: creating study thread')
        __study__(
            address,
            update_key,
            last_push,
            hours=hours,
            checkpoint=checkpoint,
        )

    def study_cauldron(self, *args, **kwargs):
        """
        Train neural network forever.
        """
        chit_chat('\b: studying the cauldron.')
        hours = float(self.settings['hours'])
        checkpoint = int(self.settings['checkpoint'])
        while True:
            self.merge_states()
            cauldron = ivy_cauldron.Cauldron(verbosity=2, debug_mode=False)
            study_sec = hours * 3600
            cauldron.train_network(max_time=study_sec)
            cauldron = None
            del(cauldron)
            gc.collect()

    def start_serving(self, *args, debug=False, **kwargs):
        """
        Create server thread.
        """
        chit_chat('\b: starting server')
        asyncio.run(__start_server__(self.address), debug=debug)

    def build_https(
        self,
        skip_charts=True,
        skip_validation=True,
        price_limit=55.0,
        ):
        """
        Create website documents.
        """
        from pandas import read_csv
        candles_path = path.join(ROOT_PATH, 'candelabrum', '{}.ivy')
        https_path = HTTPS_PATH
        charts_path = abspath(path.join(https_path, 'charts'))
        cauldron = ivy_cauldron.Cauldron(verbosity=3)
        if not skip_validation:
            chit_chat('\b: validating network')
            metrics = cauldron.validate_network()
        else:
            with open(cauldron.validation_path, 'rb') as file_obj:
                metrics = pickle.load(file_obj)
        features = cauldron.features
        candelabrum = cauldron.candelabrum
        n_batch = cauldron.constants['n_batch']
        _labels = ('close', 'trend', 'price_zs', 'price_wema', 'volume_zs')
        feature_indices = {k: features.index(k) for k in _labels}
        picks = dict()
        chit_chat('\b: generating technical rating')
        for i, (symbol_name, symbol_data) in enumerate(candelabrum.items()):
            symbol_close = symbol_data[-1, feature_indices['close']]
            if symbol_close > price_limit:
                continue
            symbol_metrics = metrics[i]
            accuracy = float(symbol_metrics['accuracy'] * 0.01)
            symbol_open = symbol_data[0, feature_indices['close']]
            symbol_trend = symbol_data[-1, feature_indices['trend']]
            symbol_zs = symbol_data[-1, feature_indices['price_zs']]
            volume_zs = symbol_data[-1, feature_indices['volume_zs']]
            symbol_wema = symbol_data[-1, feature_indices['price_wema']]
            open_close = [symbol_open, symbol_close]
            symbol_max = max(open_close)
            symbol_min = min(open_close)
            median = symbol_max - ((symbol_max - symbol_min) / 2)
            signals = [
                symbol_zs <= 0.8,
                volume_zs <= -0.5,
                symbol_trend > 3,
                symbol_close > symbol_wema,
                symbol_close > symbol_open,
                symbol_close > median,
                ]
            rating = 0
            for signal in signals:
                rating += 1 if signal else 0
            picks[symbol_name] = symbol_metrics
            picks[symbol_name]['rating'] = float(rating)
        picks = pandas.DataFrame(picks).transpose()
        picks = picks.sort_values(by=['rating', 'accuracy'], ascending=False)
        picks = picks.index[:100].tolist()
        if not skip_charts:
            chit_chat('\b: plotting charts')
            read_csv = pandas.read_csv
            chart_path = abspath(path.join(charts_path, '{}_market.png'))
            for ndx, (symbol, candles) in enumerate(candelabrum.items()):
                if symbol in ('QQQ', 'SPY'):
                    continue
                forecast = metrics[ndx]['forecast']
                cartography(
                    symbol,
                    features,
                    candles,
                    read_csv(candles_path.format(symbol)).timestamp.tolist(),
                    chart_path=chart_path.format(symbol),
                    chart_size=200,
                    forecast=forecast,
                    batch_size=n_batch,
                    )
        chit_chat('\b: building html documents')
        cabinet = ivy_https.build(
            list(candelabrum.keys()),
            features,
            candelabrum,
            metrics,
            picks,
        )
        for file_path, file_data in cabinet.items():
            file_path = abspath(path.join(https_path, file_path))
            with open(file_path, 'w+') as html_file:
                html_file.write(file_data)
