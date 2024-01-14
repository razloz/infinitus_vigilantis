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
from torch.nn import BatchNorm1d, HuberLoss
from torch.nn.functional import interpolate
from torch.nn.init import uniform_
from torch.utils.data import DataLoader, TensorDataset
from torch.fft import rfft
from torch import topk
φ = (1 + sqrt(5)) / 2
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2024, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'gardneri'


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
        debug_mode=True,
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
        self.debug_mode = debug_mode
        if debug_mode:
            logging_level = logging.DEBUG
            torch.autograd.set_detect_anomaly(True)
            verbosity = 3
            logging.info('Enabled debug mode autograd anomaly detection.')
        else:
            logging_level = logging.INFO
        logging.getLogger('asyncio').setLevel(logging_level)
        logging.basicConfig(
            filename=self.logging_path,
            encoding='utf-8',
            level=logging_level,
        )
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
        freq_out = int((n_model * 2) - 1)
        freq_half = int(n_model / 2)
        n_heads = freq_half
        n_layers = 4
        n_hidden = int(n_model * φ)
        n_dropout = φ - 1.5
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
            norm_first=True,
            device=self.DEVICE,
            dtype=torch.float,
            )
        n_learning_rate = 0.0999
        n_betas = (0.9, 0.999)
        n_weight_decay = 0.001
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=n_learning_rate,
            betas=n_betas,
            eps=n_eps,
            weight_decay=n_weight_decay,
            amsgrad=True,
            foreach=True,
            maximize=False,
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
        self.loss_fn = HuberLoss(
            reduction='mean',
            delta=1.0,
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
            'freq_out': freq_out,
            'freq_half': freq_half,
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
        freq_out = self.constants['freq_out']
        freq_half = self.constants['freq_half']
        inputs = rfft(self.normalizer(inputs), n=freq_out, dim=-1)
        inputs = inputs.real * inputs.imag
        outputs = self.network(inputs, inputs).softmax(-1)
        predictions = outputs[:, -freq_half:].sum(-1)
        return predictions

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
        verbosity = self.verbosity
        forward = self.forward
        save_state = self.save_state
        state_path = self.state_path
        validate_network = self.validate_network
        snapshot_path = path.join(self.root_folder, 'cauldron', '{}.state')
        epoch = 0
        elapsed = 0
        best_epoch = torch.inf
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
        symbol_indices = [i for i in range(n_symbols)]
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        debug_mode = self.debug_mode
        debug_anomalies = self.debug_anomalies
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
                    optimizer.zero_grad()
                    predictions = forward(batch[symbol])
                    #logits = 1 - (predictions.log() * -1)
                    loss = loss_fn(predictions, symbol_targets)
                    loss.backward()
                    optimizer.step()
                    if verbosity > 1:
                        print('\ntemp_target', symbol_targets)
                        #print('logits', logits)
                        print('predictions', predictions, '\n')
                    n_error += 1
                    epoch_error += loss.item()
                if debug_mode:
                    debug_anomalies(file_name=f'{time.time()}')
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

    def debug_anomalies(self, *args, **kwargs):
        """Does things..."""
        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt
        self.eval()
        out_proj = {'bias': [], 'weight': []}
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                if 'multihead_attn.out_proj' not in name:
                    continue
                param_keys = name.split('.')
                if param_keys[0] != 'decoder':
                    continue
                value_type = param_keys[-1]
                if value_type not in ('bias', 'weight'):
                    continue
                out_proj[value_type].append(param.clone().detach())
        fname = str(kwargs['file_name']) if 'file_name' in kwargs else ''
        img_path = path.join(self.root_folder, 'logs', '{0}.' + f'{fname}.png')
        plt.style.use('dark_background')
        plt.rcParams['figure.figsize'] = [12, 8]
        cmap = plt.get_cmap('plasma')
        fig = plt.figure()
        fig.suptitle('multihead_attn.out_proj', fontsize=18)
        x_space, y_space = torch.meshgrid(
            torch.linspace(0, 1, steps=512),
            torch.linspace(0, 1, steps=512),
            indexing='ij',
            )
        pane_color = ((0.3, 0.3, 0.3, 0.5))
        for value_type in out_proj.keys():
            s = [len(out_proj[value_type]), *out_proj[value_type][-1].shape]
            out_proj[value_type] = torch.cat(out_proj[value_type]).view(*s)
            #print(value_type, out_proj[value_type].shape)
            if value_type == 'bias':
                z_space = out_proj[value_type][-1].flatten()
            elif value_type == 'weight':
                z_space = out_proj[value_type][-1].mean(-1).flatten()
            else:
                continue
            z_space = 1 - (1 / (-z_space.softmax(0).log() + 1e-48))
            subplot = fig.add_subplot(projection='3d')
            subplot.w_xaxis.set_pane_color(pane_color)
            subplot.w_yaxis.set_pane_color(pane_color)
            subplot.w_zaxis.set_pane_color(pane_color)
            z_min = z_space.min()
            z_max = z_space.max()
            if z_min != z_max:
                subplot.set_zlim(zmin=z_min, zmax=z_max)
            else:
                subplot.set_zlim(zmin=0, zmax=1)
            plot_data = subplot.plot_surface(
                x_space,
                y_space,
                z_space.unsqueeze(1),
                cmap=cmap,
                )
            fig.colorbar(plot_data, ax=subplot, shrink=0.3, aspect=8)
            plt.savefig(img_path.format(value_type))
            fig.clf()
            plt.clf()
        plt.close()

