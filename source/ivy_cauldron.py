"""Transformer-based Sentiment Rank generator."""
import torch
import json
import logging
import os
import pickle
import time
from itertools import product
from math import sqrt
from random import randint, shuffle
from os import path, mkdir, environ
from os.path import dirname, realpath, abspath
from torch.nn import BatchNorm1d, BCEWithLogitsLoss
from torch.nn.functional import interpolate
from torch.nn.init import uniform_
from torch.utils.data import DataLoader, TensorDataset
from torch.fft import rfft
from torch import topk
φ = (1 + sqrt(5)) / 2
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2023, Daniel Ward'
__license__ = 'GPL v3'


class Cauldron(torch.nn.Module):
    def __init__(
        self,
        candelabrum=None,
        root_folder=None,
        features=None,
        symbols=None,
        input_labels=('pct_chg', 'trend', 'volume_zs', 'price_zs',),
        target_labels=('pct_chg',),
        verbosity=2,
        no_caching=True,
        set_weights=True,
        try_cuda=True,
        detect_anomaly=True,
        ):
        """Predicts the future sentiment from stock data."""
        super(Cauldron, self).__init__()
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
        self.session_path = path.join(network_path, f'{time.time()}.state')
        self.logging_path = path.join(root_folder, 'logs', 'ivy_cauldron.log')
        if path.exists(self.logging_path):
            os.remove(self.logging_path)
        logging.getLogger('asyncio').setLevel(logging.DEBUG)
        logging.basicConfig(
            filename=self.logging_path,
            encoding='utf-8',
            level=logging.DEBUG,
        )
        if detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
            if verbosity > 1:
                logging.info('Enabled autograd anomaly detection.')
        if try_cuda:
            self.DEVICE_TYPE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            if self.DEVICE_TYPE != 'cpu' and no_caching:
                environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
                if verbosity > 1:
                    logging.info('Disabled CUDA memory caching.')
                torch.cuda.empty_cache()
        else:
            self.DEVICE_TYPE = 'cpu'
        self.DEVICE = torch.device(self.DEVICE_TYPE)
        self.to(self.DEVICE)
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
            symbols = self.symbols
        if features is None:
            with open(features_path, 'rb') as features_file:
                features = pickle.load(features_file)
        if input_labels is None:
            input_labels = features
        if target_labels is None:
            target_labels = features
        input_indices = [features.index(l) for l in input_labels]
        target_indices = [features.index(l) for l in target_labels]
        n_time, n_symbols, n_features = candelabrum.shape
        n_slice = n_time if n_time % 2 == 0 else n_time - 1
        n_slice = int(n_slice / 2)
        n_batch = 8
        n_inputs = len(input_indices)
        n_targets = len(target_indices)
        self.temp_targets = torch.zeros(
            n_batch,
            n_targets,
            device=self.DEVICE,
            dtype=torch.float,
            )
        self.temp_weights = torch.zeros(
            n_batch,
            device=self.DEVICE,
            dtype=torch.float,
            )
        n_model = 512
        n_heads = 4
        n_layers = 3
        n_hidden = int(n_batch * n_model * φ)
        n_dropout = φ - 1.37
        n_eps = (1 / 137) ** 3
        self.network = torch.nn.Transformer(
            d_model=n_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=n_hidden,
            dropout=n_dropout,
            activation='gelu',
            layer_norm_eps=n_eps,
            batch_first=True,
            norm_first=False,
            device=self.DEVICE,
            dtype=torch.float,
            )
        n_learning_rate = φ * 0.1
        n_betas = (0.9, 0.999)
        n_weight_decay = 0.01
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=n_learning_rate,
            betas=n_betas,
            eps=n_eps,
            weight_decay=n_weight_decay,
            amsgrad=True,
            foreach=True,
            )
        self.normalizer = BatchNorm1d(
            n_inputs,
            eps=n_eps,
            momentum=0.1,
            affine=False,
            track_running_stats=False,
            device=self.DEVICE,
            dtype=torch.float,
            )
        self.candelabrum = candelabrum
        self.set_weights = set_weights
        self.verbosity = verbosity
        self.input_indices = input_indices
        self.target_indices = target_indices
        self.constants = {
            'n_time': n_time,
            'n_symbols': n_symbols,
            'n_features': n_features,
            'n_slice': n_slice,
            'n_batch': n_batch,
            'n_inputs': n_inputs,
            'n_targets': n_targets,
            'n_model': n_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'n_hidden': n_hidden,
            'n_dropout': n_dropout,
            'n_eps': n_eps,
            'n_learning_rate': n_learning_rate,
            'n_betas': n_betas,
            'n_weight_decay': n_weight_decay,
            }
        self.validation_results = {
            'accuracy': 0,
            }
        if verbosity > 1:
            for k, v in self.constants.items():
                logging.info(f'{k.upper()}: {v}')
        self.load_state()

    def load_state(self, state_path=None, state=None):
        """Loads the Module."""
        if state_path is None:
            state_path = self.state_path
        try:
            if state is None:
                state = torch.load(state_path, map_location=self.DEVICE_TYPE)
            if 'optimizer' in state:
                self.optimizer.load_state_dict(state['optimizer'])
            if 'network' in state:
                self.network.load_state_dict(state['network'])
            if 'validation' in state:
                self.validation_results = dict(state['validation'])
        except FileNotFoundError:
            if self.verbosity > 0:
                logging.info('No state found, creating default.')
            if self.set_weights:
                i = self.constants['n_eps']
                if self.verbosity > 0:
                    logging.info(f'Initializing with bounds of {-i} to {i}')
                for name, param in self.network.named_parameters():
                    if param.requires_grad:
                        uniform_(param, -i, i)
            self.save_state(state_path)
        except Exception as details:
            if self.verbosity > 0:
                logging.info(f'Exception {repr(details.args)}')

    def get_state_dicts(self):
        """Returns module params in a dictionary."""
        return {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'validation': dict(self.validation_results),
            }

    def save_state(self, real_path, to_buffer=False, buffer_io=None):
        """Saves the Module."""
        if not to_buffer:
            torch.save(self.get_state_dicts(), real_path)
            if self.verbosity > 1:
                logging.info(f'Saved state to {real_path}.')
        else:
            bytes_obj = self.get_state_dicts()
            bytes_obj = torch.save(bytes_obj, buffer_io)
            return bytes_obj

    def forward(self, inputs):
        """Returns predictions from inputs."""
        n = (self.constants['n_model'] * 2) - 1
        inputs = rfft(self.normalizer(inputs), n=n, dim=-1)
        inputs = inputs.real * inputs.imag
        return self.network(inputs, inputs).sigmoid().mean(-1)

    def train_network(
        self,
        checkpoint=1,
        hours=168,
        validate=False,
        quicksave=True,
        ):
        """Batched training over hours."""
        constants = self.constants
        n_batch = constants['n_batch']
        n_eps = constants['n_eps']
        n_slice = constants['n_slice']
        n_symbols = constants['n_symbols']
        n_targets = constants['n_targets']
        inf = torch.inf
        verbosity = self.verbosity
        forward = self.forward
        save_state = self.save_state
        state_path = self.state_path
        validate_network = self.validate_network
        snapshot_path = path.join(self.root_folder, 'cauldron', '{}.state')
        epoch = 0
        elapsed = 0
        best_epoch = inf
        candelabrum = self.candelabrum
        training_data = DataLoader(
            TensorDataset(
                candelabrum[:n_slice, :, self.input_indices][:-n_batch],
                candelabrum[:n_slice, :, self.target_indices][n_batch:],
                ),
            batch_size=n_batch,
            shuffle=True,
            drop_last=True,
            )
        temp_targets = self.temp_targets
        pos_weight = self.temp_weights
        symbol_indices = [i for i in range(n_symbols)]
        optimizer = self.optimizer
        if verbosity > 0:
            logging.info('Training started.')
        start_time = time.time()
        while elapsed < hours:
            self.train()
            epoch += 1
            n_error = 0
            epoch_error = 0
            for batch, targets in training_data:
                batch = batch.transpose(0, 1)
                targets = targets.transpose(0, 1)
                shuffle(symbol_indices)
                for symbol in symbol_indices:
                    temp_targets *= 0
                    symbol_targets = targets[symbol].view(n_batch, n_targets)
                    temp_targets[symbol_targets > 0] += 1
                    symbol_targets = temp_targets.flatten()
                    pos_weight *= 0
                    neg_targets = symbol_targets[symbol_targets==0].shape[0]
                    pos_targets = symbol_targets[symbol_targets==1].shape[0]
                    if neg_targets == 0:
                        pos_weight += n_eps
                    elif pos_targets == 0:
                        pos_weight += n_batch - n_eps
                    else:
                        pos_weight += neg_targets / pos_targets
                    loss_fn = BCEWithLogitsLoss(
                        pos_weight=pos_weight,
                        reduction='sum',
                        )
                    optimizer.zero_grad()
                    predictions = forward(batch[symbol])
                    loss = loss_fn(predictions.log(), symbol_targets)
                    loss.backward()
                    optimizer.step()
                    if verbosity > 1:
                        print('temp_target', symbol_targets)
                        print('predictions', predictions)
                    n_error += 1
                    epoch_error += loss.item()
                if quicksave:
                    save_state(state_path)
                if verbosity > 1:
                    print(f'batch: {n_error / n_symbols}')
                    print(f'loss: {epoch_error / n_error}')
                    logging.info(f'loss: {epoch_error / n_error}')
            elapsed = (time.time() - start_time) / 3600
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            epoch_error = epoch_error / n_error
            if verbosity > 0:
                logging.info('')
                logging.info('*********************************************')
                logging.info(f'timestamp: {ts}')
                logging.info(f'epoch: {epoch}')
                logging.info(f'elapsed: {elapsed}')
                logging.info(f'error: {epoch_error}')
                logging.info('*********************************************')
            if epoch_error < best_epoch:
                if validate:
                    validate_network()
                best_epoch = epoch_error
                file_name = ts.replace('-','').replace(':','').replace(' ','')
                file_name += f'.{epoch_error}'
                file_name = snapshot_path.format(file_name)
                if verbosity > 0:
                    logging.info(f'Saving snapshot to {file_name}')
                save_state(file_name)
            if quicksave is False and epoch % checkpoint == 0:
                if verbosity > 1:
                    logging.info(f'Saving state to {state_path}')
                save_state(state_path)
        if quicksave is False and epoch % checkpoint != 0:
            if verbosity > 1:
                logging.info(f'Saving state to {state_path}')
            save_state(state_path)
        if verbosity > 0:
            logging.info(f'Training finished after {elapsed} hours.')

    def validate_network(self):
        """Tests the network on new data."""
        constants = self.constants
        n_batch = constants['n_batch']
        n_slice = constants['n_slice']
        n_symbols = constants['n_symbols']
        n_targets = constants['n_targets']
        verbosity = self.verbosity
        forward = self.forward
        best_results = self.validation_results['accuracy']
        n_correct = 0
        n_total = 0
        results = dict()
        candelabrum = self.candelabrum
        validation_data = DataLoader(
            TensorDataset(
                candelabrum[n_slice:, :, self.input_indices][:-n_batch],
                candelabrum[n_slice:, :, self.target_indices][n_batch:],
                ),
            batch_size=n_batch,
            shuffle=False,
            drop_last=True,
            )
        symbol_indices = [i for i in range(n_symbols)]
        temp_targets = self.temp_targets
        if verbosity > 0:
            logging.info('Starting validation routine.')
        self.eval()
        for batch, targets in validation_data:
            batch = batch.transpose(0, 1)
            targets = targets.transpose(0, 1)
            for symbol in symbol_indices:
                temp_targets *= 0
                symbol_targets = targets[symbol].view(n_batch, n_targets)
                temp_targets[symbol_targets > 0] += 1
                symbol_targets = temp_targets.flatten()
                predictions = forward(batch[symbol])
                correct = 0
                for i, p in enumerate(predictions):
                    t = symbol_targets[i]
                    if any([p > 0.5 and t == 1, p <= 0.5 and t == 0]):
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
        epoch_accuracy = results['validation.metrics']['accuracy']
        if epoch_accuracy > best_results:
            self.validation_results['accuracy'] = epoch_accuracy
            self.save_state(path.join(
                self.root_folder,
                'cauldron',
                'best_results',
                f'{epoch_accuracy}.{time.time()}.state',
                ))
            if verbosity > 0:
                logging.info(f'New best accuracy of {epoch_accuracy}% saved.')
        return results

    def inscribe_sigil(self, charts_path):
        """Plot final batch predictions from the candelabrum."""
        import matplotlib.pyplot as plt
        symbols = self.symbols
        forward = self.forward
        n_batch = self.constants['n_batch']
        final_data = self.candelabrum[-n_batch:, :, self.input_indices]
        forecast_path = path.join(charts_path, '{0}_forecast.png')
        forecast = list()
        for batch in final_data.transpose(0, 1):
            forecast.append(forward(batch).clone().detach().cpu().tolist())
        plt.style.use('dark_background')
        plt.rcParams['figure.figsize'] = [10, 2]
        fig = plt.figure()
        midline_args = dict(color='red', linewidth=1.5, alpha=0.8)
        for symbol, probs in enumerate(forecast):
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
                    color='purple',
                    linewidth=bar_width,
                    alpha=0.85,
                    )
                ax.plot(
                    [i, i],
                    [0, p],
                    color='green',
                    linewidth=bar_width,
                    alpha=1.0,
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
            plt.savefig(forecast_path.format(symbols[symbol]))
            fig.clf()
            plt.clf()
        plt.close()
        with open(self.validation_path, 'rb') as validation_file:
            metrics = pickle.load(validation_file)
        return (metrics, forecast)
