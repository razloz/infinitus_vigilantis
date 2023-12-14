"""Transformer-based Sentiment Rank generator."""
import torch
import json
import pickle
import time
from random import randint
from os import path, mkdir, environ
from os.path import dirname, realpath, abspath
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.init import uniform_
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
        input_labels=None,
        target_labels=('pct_chg',),
        verbosity=0,
        no_caching=True,
        set_weights=True,
        try_cuda=False,
        detect_anomaly=False,
        ):
        """Predicts the future sentiment from stock data."""
        super(Cauldron, self).__init__()
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
        self.session_path = path.join(network_path, f'{time.time()}.state')
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
        self.training_data = DataLoader(
            TensorDataset(
                candelabrum[:n_slice, :, input_indices][:-n_batch],
                candelabrum[:n_slice, :, target_indices][n_batch:],
                ),
            batch_size=n_batch,
            drop_last=True,
            )

        self.validation_data = DataLoader(
            TensorDataset(
                candelabrum[n_slice:, :, input_indices][:-n_batch],
                candelabrum[n_slice:, :, target_indices][n_batch:],
                ),
            batch_size=n_batch,
            drop_last=True,
            )
        self.final_data = candelabrum[-n_batch:, :, input_indices]
        self.temp_target = torch.zeros(
            n_batch,
            n_symbols,
            device=self.DEVICE,
            dtype=torch.float,
            )
        n_model = n_features
        n_heads = n_features
        n_layers = n_batch
        n_hidden = n_batch * n_symbols
        n_dropout = 1.618033988749894 - 1.5
        n_eps = (1 / 137) ** 3
        self.network = torch.nn.Transformer(
            d_model=n_model,
            nhead=n_heads,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=n_hidden,
            dropout=n_dropout,
            activation=torch.nn.functional.leaky_relu,
            layer_norm_eps=n_eps,
            batch_first=True,
            norm_first=True,
            device=self.DEVICE,
            dtype=torch.float,
            )
        n_learning_rate = 0.001
        n_betas = (0.9, 0.999)
        n_weight_decay = 0.01
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=n_learning_rate,
            betas=n_betas,
            eps=n_eps,
            weight_decay=n_weight_decay,
            )
        self.candelabrum = candelabrum
        self.set_weights = set_weights
        self.verbosity = verbosity
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
        if verbosity > 0:
            for k, v in self.constants.items():
                print(f'{k.upper()}: {v}')
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
        except FileNotFoundError:
            if self.verbosity > 0:
                print('No state found, creating default.')
            if self.set_weights:
                i = self.constants['n_eps']
                if self.verbosity > 0:
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
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
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

    def forward(self, batch, targets=None):
        """Takes batched inputs and returns the future sentiment."""
        n_batch, n_symbols, n_features = batch.shape
        network = self.network
        predictions = list()
        if targets is not None:
            bce = torch.nn.BCEWithLogitsLoss
            optimizer = self.optimizer
            optimizer.zero_grad()
        for i, inputs in enumerate(batch):
            state = network(inputs, inputs).var(dim=1, correction=0)
            if targets is not None:
                target = targets[i]
                pos_targets = target.sum(0)
                if pos_targets > 0:
                    neg_targets = n_symbols - pos_targets
                    target_ratio = (neg_targets / pos_targets)
                    weights = (target * target_ratio) + 1
                else:
                    weights = (target + 1)
                loss_fn = bce(pos_weight=weights)
                loss = loss_fn(state, target)
                loss.backward()
            predictions.append(state.clone().detach())
        if targets is not None:
            optimizer.step()
        predictions = torch.cat(predictions).view(n_batch, n_symbols)
        predictions[predictions > 1] = 1
        predictions[predictions < 0] = 0
        return predictions.clone().detach()

    def train_network(self, depth=1, hours=168, checkpoint=1, quicksave=True, validate=False):
        """Batched training over hours."""
        constants = self.constants
        n_batch = constants['n_batch']
        n_symbols = constants['n_symbols']
        verbosity = self.verbosity
        forward = self.forward
        save_state = self.save_state
        state_path = self.state_path
        dataset = self.training_data
        temp_target = self.temp_target
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
            for batch, targets in dataset:
                temp_target *= 0
                temp_target[targets.view(n_batch, n_symbols) > 0] += 1
                predictions = forward(batch, targets=temp_target)
                if quicksave:
                    save_state(state_path)
                if verbosity > 0:
                    print(repr(predictions))
                    print('predictions', predictions.shape)
            elapsed = (time.time() - start_time) / 3600
            if not quicksave:
                if epoch % checkpoint == 0:
                    if verbosity > 0:
                        ts = time.localtime()
                        ts = time.strftime('%Y-%m-%d %H:%M:%S', ts)
                        print(f'{ts}: epoch = {epoch} || elapsed = {elapsed}')
                save_state(state_path)
        if not quicksave:
            if epoch % checkpoint != 0:
                save_state(state_path)
        if verbosity > 0:
            print(f'Training finished after {elapsed} hours.')

    def validate_network(self):
        """Tests the network on new data."""
        constants = self.constants
        n_batch = constants['n_batch']
        n_symbols = constants['n_symbols']
        dataset = self.validation_data
        verbosity = self.verbosity
        network = self.network
        forward = self.forward
        temp_target = self.temp_target
        n_correct = 0
        n_total = 0
        results = dict()
        if verbosity > 0:
            print('Starting validation routine.')
        self.eval()
        for batch, targets in dataset:
            predictions = forward(batch, targets=None)
            temp_target *= 0
            temp_target[targets.view(n_batch, n_symbols) > 0] += 1
            targets = temp_target.transpose(0, 1)
            for symbol, forecast in enumerate(predictions.transpose(0, 1)):
                correct = 0
                for target_prob, prob in enumerate(forecast):
                    target = targets[symbol, target_prob]
                    target_match = any((
                        target == 1 and prob > 0.5,
                        target == 0 and prob <= 0.5,
                        ))
                    if target_match:
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
        final_data = self.final_data
        symbols = self.symbols
        forecast_path = path.join(charts_path, '{0}_forecast.png')
        forecast = list()
        plt.style.use('dark_background')
        plt.rcParams['figure.figsize'] = [10, 2]
        fig = plt.figure()
        midline_args = dict(color='red', linewidth=1.5, alpha=0.8)
        self.eval()
        predictions = self.forward(final_data, targets=None)
        for symbol, probs in enumerate(predictions.transpose(0, 1)):
            forecast.append(probs.clone().detach())
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
