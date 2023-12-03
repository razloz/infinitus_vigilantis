"""Transformer-based Sentiment Rank generator."""
import torch
import json
import pickle
import time
from random import randint
from os import path, mkdir, environ
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.init import uniform_
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2023, Daniel Ward'
__license__ = 'GPL v3'
π = torch.pi
φ = 1.618033988749894
ε = (1 / 137) ** 3
FLOAT = torch.float
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
        features=None,
        symbols=None,
        input_labels=('pct_chg', 'trend', 'volume_zs', 'price_zs',),
        target_labels=('pct_chg',),
        verbosity=0,
        no_caching=True,
        set_weights=True,
        try_cuda=False,
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
        self.root_folder = root_folder
        candelabrum_path = abspath(path.join(root_folder, 'candelabrum'))
        candles_path = path.join(candelabrum_path, 'candelabrum.candles')
        features_path = path.join(candelabrum_path, 'candelabrum.features')
        symbols_path = path.join(candelabrum_path, 'candelabrum.symbols')
        network_path = path.join(root_folder, 'cauldron')
        if not path.exists(network_path):
            mkdir(network_path)
        self.state_path = path.join(network_path, 'cauldron.state')
        self.validation_path = path.join(network_path, 'cauldron.validation')
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
            with open(symbols_path, 'rb') as f:
                self.symbols = pickle.load(f)
        if features is None:
            with open(features_path, 'rb') as features_file:
                features = pickle.load(features_file)
        input_indices = [features.index(l) for l in input_labels]
        target_indices = [features.index(l) for l in target_labels]
        n_batch = 8
        n_slice = n_batch * 2
        n_inputs = len(input_indices)
        n_targets = len(target_indices)
        n_time, n_symbols, n_data = candelabrum.shape
        n_data -= 1
        self.pi_phi = torch.tensor(
            [-1, π, 1, -φ, 0, φ, -1, -π, 1],
            device=self.DEVICE,
            dtype=FLOAT,
            )
        self.pattern_range = range(n_batch)
        self.patterns = torch.zeros(
            [n_batch, 2],
            device=self.DEVICE,
            dtype=FLOAT,
            )
        self.dataset = DataLoader(
            TensorDataset(candelabrum),
            batch_size=n_slice,
            drop_last=True,
            )
        self.input_mask = torch.full(
            [n_batch, n_inputs],
            2.5 - φ,
            device=self.DEVICE,
            dtype=FLOAT,
            )
        n_heads = n_inputs * 9
        d_model = n_heads
        self.network = torch.nn.Transformer(
            d_model=d_model,
            nhead=n_heads,
            num_encoder_layers=128,
            num_decoder_layers=128,
            dim_feedforward=1024,
            dropout=0.1,
            activation=leaky_relu,
            layer_norm_eps=ε,
            batch_first=True,
            norm_first=True,
            device=self.DEVICE,
            dtype=FLOAT,
            )
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=ε,
            weight_decay=0.01,
            )
        self.metrics = dict(
            training_epochs=0,
            training_error=0,
            training_time=0,
            )
        self.set_weights = set_weights
        self.verbosity = verbosity
        self.candelabrum = candelabrum
        self.input_indices = input_indices
        self.target_indices = target_indices
        self.d_model = d_model
        self.d_split = int(d_model / 2)
        self.n_inputs = n_inputs
        self.n_targets = n_targets
        self.n_heads = n_heads
        self.n_batch = n_batch
        self.n_slice = n_slice
        self.n_time = n_time
        self.n_symbol = n_symbols - 1
        self.n_data = n_data
        self.batch_max = n_time - n_slice - 1
        self.n_forecast = n_batch * 2
        self.load_state()

    def load_state(self, state_path=None, state=None):
        """Loads the Module."""
        if state_path is None:
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
            if self.verbosity > 0:
                print('No state found, creating default.')
            if self.set_weights:
                i = ε
                if self.verbosity > 1:
                    print(f'Initializing weights with bounds of {-i} to {i}')
                for name, param in self.network.named_parameters():
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
            if self.verbosity > 1:
                print(f'Saved state to {real_path}.')
        else:
            bytes_obj = self.get_state_dicts()
            bytes_obj = torch.save(bytes_obj, buffer_io)
            return bytes_obj

    def random_batch(self):
        """Returns a random batch of inputs and targets"""
        input_indices = self.input_indices
        target_indices = self.target_indices
        n_inputs = self.n_inputs
        n_targets = self.n_targets
        n_batch = self.n_batch
        n_data = self.n_data
        batch_start = randint(0, self.batch_max)
        batch_end = batch_start + self.n_slice
        batch_symbol = randint(0, self.n_symbol)
        batch = self.candelabrum[batch_start:batch_end, batch_symbol, :]
        return (
            batch[:n_batch, input_indices].view(n_batch, n_inputs),
            batch[n_batch:, target_indices].view(n_batch, n_targets),
            )

    def encoder(self, tensor, softmax=True):
        """Onehot encoding of tensor data."""
        pattern = self.patterns.clone().detach()
        pattern_range = self.pattern_range
        tensor = tensor.flatten()
        for i in pattern_range:
            truth = tensor[i]
            p = 0 if truth > 0 else 1
            if softmax:
                pattern[i][p] += truth.abs() + 1
            else:
                pattern[i][p] += 1
        if softmax:
            return pattern.softmax(-1)
        else:
            return pattern

    def forward(self, inputs):
        """Takes batched inputs and returns the future sentiment."""
        global ε
        n_batch = self.n_batch
        n_inputs = self.n_inputs
        d_model = self.d_model
        d_split = self.d_split
        πφ = self.pi_phi
        cat = torch.cat
        topk = torch.topk
        sigil = list()
        for d0 in inputs:
            for d1 in d0:
                if d1 != 0:
                    inscription = πφ * (1 / d1)
                else:
                    inscription = (πφ * 0) + ε
                sigil.append(inscription.clone().detach())
        sigil = torch.cat(sigil).view(n_batch, n_inputs, 9)
        sigil = sigil.sigmoid().log_softmax(-1).view(n_batch, d_model)
        state = self.network(sigil, sigil).view(n_batch, 2, d_split)
        return state.sum(-1).log_softmax(-1)

    def train_network(
        self,
        depth=100,
        hours=168,
        checkpoint=10,
        validate=False,
        use_mask=False,
        ):
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
        n_batch = self.n_batch
        depth_range = range(depth)
        epoch = 0
        elapsed = 0
        if verbosity > 0:
            print('Training started.')
        self.train()
        start_time = time.time()
        while elapsed < hours:
            epoch += 1
            if verbosity > 1:
                print('epoch:', epoch, '|| elapsed:', elapsed)
            inputs, targets = random_batch()
            targets = encoder(targets, softmax=False)
            for _ in depth_range:
                optimizer.zero_grad()
                if use_mask:
                    state = forward(inputs * bernoulli(input_mask))
                else:
                    state = forward(inputs)
                loss = loss_fn(state, targets)
                loss.backward()
                optimizer.step()
            elapsed = (time.time() - start_time) / 3600
            if validate and epoch % validate == 0:
                self.validate_network()
            if epoch % checkpoint == 0:
                if verbosity > 0:
                    print('')
                    print(repr(state.exp()))
                    print('state', state.shape)
                    print(repr(targets))
                    print('targets', targets.shape)
                    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                    print(f'{ts}: epoch = {epoch} || elapsed = {elapsed}')
                self.save_state(state_path)
        if validate and epoch % validate != 0:
            self.validate_network()
        if epoch % checkpoint != 0:
            self.save_state(state_path)
        if verbosity > 0:
            print(f'Training finished after {elapsed} hours.')

    def validate_network(self):
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
        input_indices = self.input_indices
        target_indices = self.target_indices
        n_inputs = self.n_inputs
        n_targets = self.n_targets
        batches = 0
        n_correct = 0
        n_total = 0
        results = dict()
        batch_range = range(n_batch)
        symbol_range = range(n_symbol + 1)
        if verbosity > 0:
            print('Starting validation routine.')
        self.eval()
        for batches, batch in enumerate(dataset):
            batch = batch[0].transpose(0, 1)
            for symbol in symbol_range:
                inputs = batch[symbol, :n_batch, input_indices]
                inputs = inputs.view(n_batch, n_inputs)
                targets = batch[symbol, n_batch:, target_indices]
                targets = targets.view(n_batch, n_targets)
                targets = encoder(targets, softmax=False).flatten()
                state = forward(inputs).flatten()
                state_pred = topk(state, n_batch, largest=True, sorted=False)
                target_pred = topk(targets, n_batch, largest=True, sorted=False)
                state_pred = state_pred.indices
                target_pred = target_pred.indices
                correct = 0
                for i in batch_range:
                    if state_pred[i] == target_pred[i]:
                        correct += 1
                if symbol not in results:
                    results[symbol] = dict()
                    results[symbol]['correct'] = 0
                    results[symbol]['total'] = 0
                results[symbol]['correct'] += correct
                results[symbol]['total'] += n_batch
                n_correct += correct
                n_total += n_batch
        results['validation.metrics'] = {
            'correct': n_correct,
            'total': n_total,
            }
        for key in results:
            correct = results[key]['correct']
            total = results[key]['total']
            results[key]['accuracy'] = round((correct / total) * 100, 4)
        with open(self.validation_path, 'wb+') as validation_file:
            pickle.dump(results, validation_file)

    def inscribe_sigil(self, charts_path):
        """Plot final batch predictions from the candelabrum."""
        import matplotlib.pyplot as plt
        # imported values
        symbols = self.symbols
        n_batch = self.n_batch
        n_inputs = self.n_inputs
        forward = self.forward
        input_indices = self.input_indices
        candelabrum = self.candelabrum[-n_batch:].transpose(0, 1)
        forecast_path = path.join(charts_path, '{0}_forecast.png')
        forecast = list()
        # chart stuff
        plt.style.use('dark_background')
        plt.rcParams['figure.figsize'] = [10, 2]
        fig = plt.figure()
        midline_args = dict(color='red', linewidth=1.5, alpha=0.8)
        # plot forecast
        self.eval()
        for index in range(len(symbols)):
            inputs = candelabrum[index, :, input_indices]
            sigil = forward(inputs.view(n_batch, n_inputs)).exp()
            probs = sigil[:, 0].tolist()
            forecast.append(probs)
            ax = fig.add_subplot()
            ax.set_ylabel('Confidence', fontweight='bold')
            ax.grid(True, color=(0.3, 0.3, 0.3))
            ax.set_xlim(0, 9)
            ax.set_ylim(0, 1)
            bar_width = ax.get_tightbbox(fig.canvas.get_renderer()).get_points()
            bar_width = ((bar_width[1][0] - bar_width[0][0]) / 7) * 0.5
            for i, p in enumerate(probs):
                i = i + 1
                ax.plot(
                    [i, i],
                    [p, 1],
                    color='red',
                    linewidth=bar_width,
                    alpha=0.7,
                    )
                ax.plot(
                    [i, i],
                    [0, p],
                    color='green',
                    linewidth=bar_width,
                    alpha=0.9,
                    )
            ax.plot(
                [0, 9],
                [0.5, 0.5],
                color=(0.5, 0.5, 0.5),
                linewidth=1.5,
                alpha=0.9,
                )
            ax.set_xticks(ax.get_xticks())
            x_labels = ax.get_xticklabels()
            x_labels[0].set_text('')
            x_labels[-1].set_text('')
            ax.set_xticklabels(x_labels)
            fig.suptitle('Daily Forecast', fontsize=18)
            plt.savefig(forecast_path.format(symbols[index]))
            fig.clf()
            plt.clf()
        plt.close()
        with open(self.validation_path, 'rb') as validation_file:
            metrics = pickle.load(validation_file)
        return (metrics, forecast)
