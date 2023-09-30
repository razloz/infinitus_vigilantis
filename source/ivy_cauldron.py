"""Transformer-based Sentiment Rank generator."""
import torch
import json
import time
from random import randint
from os import path, mkdir, environ
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.init import uniform_
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2023, Daniel Ward'
__license__ = 'GPL v3'
FLOAT = torch.float
PI = torch.pi
topk = torch.topk
vstack = torch.vstack
leaky_relu = torch.nn.functional.leaky_relu
dirname = path.dirname
realpath = path.realpath
abspath = path.abspath


class Cauldron(torch.nn.Module):
    def __init__(
        self,
        candelabrum=None,
        root_folder=None,
        symbols=None,
        input_index=-1,
        target_index=-1,
        verbosity=0,
        no_caching=True,
        set_weights=True,
        try_cuda=True,
        detect_anomaly=False,
        ):
        """Predicts the future sentiment from stock data."""
        super(Cauldron, self).__init__()
        self.start_time = time.time()
        if detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        if try_cuda:
            self.DEVICE_TYPE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            if self.DEVICE_TYPE != 'cpu' and no_caching:
                environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
                if verbosity > 1:
                    print('Disabled CUDA memory caching.')
                torch.cuda.empty_cache()
        else:
            self.DEVICE_TYPE = 'cpu'
        self.DEVICE = torch.device(self.DEVICE_TYPE)
        self.to(self.DEVICE)
        if root_folder is None:
            root_folder = abspath(path.join(dirname(realpath(__file__)), '..'))
        candelabrum_path = abspath(path.join(root_folder, 'candelabrum'))
        candles_path = path.join(candelabrum_path, 'candelabrum.candles')
        symbols_path = path.join(candelabrum_path, 'candelabrum.symbols')
        network_path = path.join(root_folder, 'cauldron')
        if not path.exists(network_path):
            mkdir(network_path)
        self.state_path = path.join(network_path, 'cauldron.state')
        self.session_path = path.join(network_path, f'{self.start_time}.state')
        if candelabrum is None:
            candelabrum = torch.load(
                candles_path,
                map_location=self.DEVICE_TYPE,
                )
        else:
            candelabrum = candelabrum.clone().detach()
        candelabrum.to(self.DEVICE)
        if symbols is None:
            with open(symbols_path, 'r') as f:
                self.symbols = json.loads(f.read())['symbols']
        n_batch = 6
        n_slice = n_batch * 2
        n_time, n_symbols, n_data = candelabrum.shape
        n_data -= 1
        self.pattern_range = range(n_batch)
        self.patterns = torch.zeros(
            [n_batch, 3],
            device=self.DEVICE,
            dtype=FLOAT,
            )
        self.dataset = DataLoader(
            TensorDataset(candelabrum),
            batch_size=n_slice,
            drop_last=True,
            )
        self.input_mask = torch.full(
            [1, n_batch],
            0.382,
            device=self.DEVICE,
            dtype=FLOAT,
            )
        self.network = torch.nn.Transformer(
            d_model=n_batch,
            nhead=n_batch,
            num_encoder_layers=n_slice * 2,
            num_decoder_layers=n_slice * 2,
            dim_feedforward=3 ** n_batch,
            dropout=0.618033988749894,
            activation=leaky_relu,
            layer_norm_eps=1.18e-6,
            batch_first=True,
            norm_first=True,
            device=self.DEVICE,
            dtype=FLOAT,
            )
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adagrad(self.parameters())
        self.metrics = dict(
            training_epochs=0,
            training_error=0,
            training_time=0,
            )
        self.set_weights = set_weights
        self.verbosity = verbosity
        self.candelabrum = candelabrum
        self.n_batch = n_batch
        self.n_slice = n_slice
        self.n_time = n_time
        self.n_symbol = n_symbols - 1
        self.n_data = n_data
        self.batch_max = n_time - n_slice - 1
        self.input_index = input_index
        self.target_index = target_index
        self.load_state()

    def load_state(self, state=None):
        """Loads the Module."""
        state_path = self.state_path
        try:
            if state is None:
                state = torch.load(state_path, map_location=self.DEVICE_TYPE)
            if 'metrics' in state:
                self.metrics = dict(state['metrics'])
            if 'state' in state:
                self.load_state_dict(state['state'])
            if 'optimizer' in state:
                self.optimizer.load_state_dict(state['optimizer'])
            if 'network' in state:
                self.network.load_state_dict(state['network'])
        except FileNotFoundError:
            i = (1 / 137) ** 3
            if self.verbosity > 0:
                print('No state found, creating default.')
            if self.set_weights:
                if self.verbosity > 0:
                    print(f'Initializing weights with bounds of {-i} to {i}')
                for name, param in self.named_parameters():
                    if param.requires_grad:
                        uniform_(param, -i, i)
            self.save_state(state_path)
        except Exception as details:
            if self.verbosity > 0:
                print('Exception', *details.args)

    def get_state_dicts(self):
        """Returns module params in a dictionary."""
        return {
            'metrics': self.metrics,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state': self.state_dict(),
            }

    def save_state(self, real_path, to_buffer=False, buffer_io=None):
        """Saves the Module."""
        if not to_buffer:
            torch.save(self.get_state_dicts(), real_path)
            if self.verbosity > 0:
                print(f'Saved state to {real_path}.')
        else:
            bytes_obj = self.get_state_dicts()
            bytes_obj = torch.save(bytes_obj, buffer_io)
            return bytes_obj

    def random_batch(self):
        """Returns a random batch of inputs and targets"""
        input_index = self.input_index
        target_index = self.target_index
        n_batch = self.n_batch
        n_data = self.n_data
        batch_start = randint(0, self.batch_max)
        batch_end = batch_start + self.n_slice
        batch_symbol = randint(0, self.n_symbol)
        batch = self.candelabrum[batch_start:batch_end, batch_symbol, :]
        return (
            batch[:n_batch, input_index].view(1, n_batch),
            batch[n_batch:, target_index].view(1, n_batch),
            )

    def encoder(self, tensor, softmax=True):
        """Onehot encoding of tensor data."""
        pattern = self.patterns.clone().detach()
        pattern_range = self.pattern_range
        tensor = tensor.flatten()
        for i in pattern_range:
            truth = tensor[i]
            p = 0 if truth > 0 else 1 if truth == 0 else 2
            if softmax:
                pattern[i][p] += truth.abs() + 1
            else:
                pattern[i][p] += 1
        if softmax:
            return pattern.softmax(-1)
        else:
            return pattern

    def forward(self, inputs, mask, use_mask=True):
        """Takes batched inputs and returns the future sentiment."""
        if use_mask:
            state = self.network(inputs * mask, inputs)
        else:
            state = self.network(inputs, inputs)
        return self.encoder(state)

    def train_network(self, n_depth=9, hours=96, checkpoint=60, validate=False):
        """Studies masked random batches for a specified amount of hours."""
        verbosity = self.verbosity
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        input_mask = self.input_mask
        random_batch = self.random_batch
        encoder = self.encoder
        bernoulli = torch.bernoulli
        forward = self.forward
        state_path = self.state_path
        epoch = self.metrics['training_epochs']
        elapsed = 0
        self.train()
        if verbosity > 0:
            print('Training started.')
        start_time = time.time()
        while elapsed < hours:
            epoch_time = time.time()
            epoch += 1
            if verbosity > 0:
                print('Epoch:', epoch)
            optimizer.zero_grad()
            inputs, targets = random_batch()
            targets = encoder(targets, softmax=False)
            mask = bernoulli(input_mask)
            error = 0
            depth = 0
            while depth <= n_depth:
                state_time = time.time()
                state = forward(inputs, mask, use_mask=True)
                loss = loss_fn(state, targets)
                loss.backward()
                optimizer.step()
                error = loss.item()
                depth += 1
            ts = time.time()
            elapsed = ((ts - start_time) / 60) / 60
            epoch_time = ((ts - epoch_time) / 60) / 60
            self.metrics['training_epochs'] = epoch
            self.metrics['training_error'] = error
            self.metrics['training_time'] += epoch_time
            if verbosity > 1:
                print(state, 'state')
                print(targets, 'targets')
            if verbosity > 0:
                print('epoch_time:', epoch_time * 60, 'minutes')
                print('training_error:', self.metrics['training_error'])
                print('training_time:', self.metrics['training_time'], 'hours')
                print('session_time:', elapsed, 'hours')
            if validate and epoch % validate == 0:
                self.validate_network()
            if epoch % checkpoint == 0:
                self.save_state(state_path)
        if validate and epoch % validate != 0:
            self.validate_network()
        if epoch % checkpoint != 0:
            self.save_state(state_path)
        if verbosity > 0:
            print(f'Training finished after {elapsed} hours.')

    def validate_network(self, threshold=0.001):
        """Validates the network and store the results."""
        dataset = self.dataset
        verbosity = self.verbosity
        network = self.network
        n_batch = self.n_batch
        n_slice = self.n_slice
        n_time = self.n_time
        n_symbol = self.n_symbol
        n_data = self.n_data
        batch_max = self.batch_max
        encoder = self.encoder
        forward = self.forward
        input_index = self.input_index
        target_index = self.target_index
        accuracy = 0
        batches = 0
        error = 0
        n_correct = 0
        n_total = 0
        results = dict()
        batch_range = range(n_batch)
        symbol_range = range(n_symbol + 1)
        if verbosity > 0:
            print('Starting validation routine.')
        self.eval()
        start_time = time.time()
        for batches, batch in enumerate(dataset):
            epoch_time = time.time()
            batch = batch[0].transpose(0, 1)
            for symbol in symbol_range:
                time_track = time.time()
                inputs = batch[symbol, :n_batch, input_index].view(1, n_batch)
                targets = batch[symbol, n_batch:, target_index].view(1, n_batch)
                print('select data:', time.time() - time_track, 'seconds.')
                time_track = time.time()
                state = forward(inputs, None, use_mask=False).flatten()
                print('encode state:', time.time() - time_track, 'seconds.')
                time_track = time.time()
                targets = encoder(targets, softmax=False).flatten()
                print('encode targets:', time.time() - time_track, 'seconds.')
                time_track = time.time()
                state_pred = topk(state, n_batch, largest=True, sorted=False)
                target_pred = topk(targets, n_batch, largest=True, sorted=False)
                print('topk:', time.time() - time_track, 'seconds.')
                time_track = time.time()
                state_pred = state_pred.indices
                target_pred = target_pred.indices
                correct = 0
                for i in batch_range:
                    if state_pred[i] == target_pred[i]:
                        correct += 1
                print('check correct:', time.time() - time_track, 'seconds.')
                time_track = time.time()
                error = correct / n_batch
                if symbol not in results.keys():
                    results[symbol] = dict()
                    results[symbol]['correct'] = 0
                    results[symbol]['total'] = 0
                    results[symbol]['error'] = error
                else:
                    results[symbol]['error'] += error
                results[symbol]['correct'] += correct
                results[symbol]['total'] += n_batch
                n_correct += correct
                n_total += n_batch
                print('store results:', time.time() - time_track, 'seconds.')
                print('correct:', correct, 'out of', n_batch, '(', error, ')')
            ts = time.time()
            elapsed = ((ts - start_time) / 60) / 60
            epoch_time = (ts - epoch_time) / 60
            print('n_correct / n_total:', n_correct, '/', n_total)
            print('epoch:', epoch_time, 'minutes.')
            print('elapsed:', elapsed, 'hours.')
        print(batches)
        print(results)

