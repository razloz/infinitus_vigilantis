"""Recurrent Neural Network for Infinitus Vigilantis."""
import logging
import os
import pickle
import torch
from random import randint
from time import localtime, strftime
from torch.utils.data import DataLoader, TensorDataset
from os import path
from os.path import dirname, realpath, abspath
from source.ivy_candles import Candelabrum
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2024, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'gardneri'
ROOT_FOLDER = abspath(path.join(dirname(realpath(__file__)), '..'))
CANDELABRUM_PATH = abspath(path.join(ROOT_FOLDER, 'candelabrum'))
CANDLES_PATH = path.join(CANDELABRUM_PATH, 'candelabrum.candles')
FEATURES_PATH = path.join(CANDELABRUM_PATH, 'candelabrum.features')
SYMBOLS_PATH = path.join(CANDELABRUM_PATH, 'candelabrum.symbols')
NETWORK_PATH = path.join(ROOT_FOLDER, 'rnn')
STATE_PATH = path.join(NETWORK_PATH, 'rnn.state')
LOGGING_PATH = path.join(ROOT_FOLDER, 'logs')
LOG_FILE = path.join(LOGGING_PATH, 'ivy_rnn.log')
if not path.exists(NETWORK_PATH):
    os.mkdir(NETWORK_PATH)
if not path.exists(LOGGING_PATH):
    os.mkdir(LOGGING_PATH)
if path.exists(LOG_FILE):
    os.remove(LOG_FILE)
logging.basicConfig(filename=LOG_FILE, encoding='utf-8', level=logging.INFO)
INPUT_LABELS = ('price_med', 'pct_chg', 'price_wema', 'price_zs', 'volume_zs')
TARGET_LABELS = ('open', 'close')
get_timestamp = lambda: strftime('%Y-%m-%d %H:%M:%S', localtime())


class RNN(torch.nn.Module):
    def __init__(self):
        """."""
        super(RNN, self).__init__()
        self.DEVICE_TYPE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.DEVICE = torch.device(self.DEVICE_TYPE)
        self.to(self.DEVICE)
        with open(FEATURES_PATH, 'rb') as f:
            self.features = features = pickle.load(f)
        with open(SYMBOLS_PATH, 'rb') as f:
            self.symbols = symbols = pickle.load(f)
        input_indices = [features.index(l) for l in INPUT_LABELS]
        target_indices = [features.index(l) for l in TARGET_LABELS]
        self.dataset = torch.load(CANDLES_PATH, map_location=self.DEVICE_TYPE)
        self.training_data = list()
        self.validation_data = list()
        self.final_data = list()
        n_batch = 5
        cat = torch.cat
        for data in self.dataset:
            n_time = data.shape[0]
            n_slice = n_time if n_time % 2 == 0 else n_time - 1
            n_slice = int(n_slice / 2)
            inputs = data[:n_slice, input_indices][:-n_batch]
            inputs.requires_grad_(True)
            targets = data[:n_slice, target_indices][n_batch:]
            self.training_data.append(
                DataLoader(
                    TensorDataset(
                        cat([inputs.flip(0), inputs]),
                        cat([targets.flip(0), targets]),
                        ),
                    batch_size=n_batch,
                    drop_last=True,
                    ),
                )
            self.validation_data.append(
                DataLoader(
                    TensorDataset(
                        data[n_slice:, input_indices][:-n_batch],
                        data[n_slice:, target_indices][n_batch:],
                        ),
                    batch_size=n_batch,
                    drop_last=True,
                    ),
                )
            self.final_data.append(data[-n_batch:, input_indices])
        self.network = torch.nn.LSTM(
            input_size=len(input_indices),
            hidden_size=128,
            num_layers=40,
            bias=True,
            batch_first=True,
            dropout=0.5,
            device=self.DEVICE,
            dtype=torch.float,
            )
        self.optimizer = torch.optim.RMSprop(
            self.parameters(),
            lr=0.01,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0.01,
            momentum=0.99,
            foreach=True,
            )
        self.load_state()

    def load_state(self, state_path=None, state=None):
        """Loads the Module."""
        if state_path is None:
            state_path = STATE_PATH
        try:
            if state is None:
                state = torch.load(state_path, map_location=self.DEVICE_TYPE)
            if 'optimizer' in state:
                self.optimizer.load_state_dict(state['optimizer'])
            if 'network' in state:
                self.network.load_state_dict(state['network'])
        except FileNotFoundError:
            logging.info('No state found, creating default.')
            self.save_state(state_path)
        except Exception as details:
            logging.error(f'Exception {repr(details.args)}')

    def save_state(self, state_path=None, to_buffer=False, buffer_io=None):
        """Saves the Module."""
        if not to_buffer:
            if state_path is None:
                state_path = STATE_PATH
            torch.save(self.get_state_dicts(), state_path)
            logging.info(f'{get_timestamp()}: Saved state to {state_path}.')
        else:
            bytes_obj = self.get_state_dicts()
            bytes_obj = torch.save(bytes_obj, buffer_io)
            return bytes_obj

    def get_state_dicts(self):
        """Returns module params in a dictionary."""
        return {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }

    def forward(self, inputs):
        """."""
        state = self.network(inputs)[1][0].transpose(0, 1)
        state = state[torch.topk(state, 1, dim=0).indices].flatten()
        state[state.argmax(0)] *= 0
        state = state.mean(0).squeeze(0)
        return state

    def deep_learn(self):
        """."""
        training_data = self.training_data
        forward = self.forward
        loss_fn = torch.nn.HuberLoss(reduction='mean')
        optimizer = self.optimizer
        self.train()
        while True:
            epoch = 0
            for n_sets, dataset in enumerate(training_data):
                n_total = 0
                mse = 0
                for inputs, targets in dataset:
                    optimizer.zero_grad()
                    tail_price = inputs[-1, 0]
                    target_med = targets.max()
                    target_med = target_med - ((target_med - targets.min()) / 2)
                    target = (target_med - tail_price) / tail_price
                    output = forward(inputs)
                    prediction = tail_price * (1 + output)
                    loss = loss_fn(output.sigmoid(), target.sigmoid())
                    loss.backward()
                    optimizer.step()
                    mse += loss.item()
                    n_total += 1
                mse /= n_total
                print(f'({n_total}) output: {output} target: {target}')
                print(f'tail_price: {tail_price}')
                print(f'target_med: {target_med}')
                print(f'prediction: {prediction}')
                print(f'({n_sets + 1}) MSE: {mse}\n')
                self.save_state()
            epoch += 1
        return True
