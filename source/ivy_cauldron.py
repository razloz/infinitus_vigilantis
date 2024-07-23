"""Transformer-based Sentiment Rank generator."""
import torch
import logging
import math
import os
import pickle
import time
import gc
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from itertools import combinations
from time import localtime, strftime
from random import randint
from os import path, mkdir, environ
from os.path import dirname, realpath, abspath
from torch.nn.init import uniform_
from torch.nn import Transformer, LayerNorm
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, TensorDataset
from torch import topk, stack
from torch.fft import fft
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2024, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'gardneri'
ᶓ = math.e
π = math.pi
φ = (1 + math.sqrt(5)) / 2


class ModifiedHuberLoss(torch.nn.Module):
    def __init__(
        self,
        epsilon=1e-10,
        gamma=0.75,
        penalties=(0.3, 0.7),
        reduction='mean',
        ):
        """
        Modified Huber Loss for Binary Classification.
            kwargs:
                epsilon     ->  a small float to avoid division by 0
                gamma       ->  a float less than 1.0
                penalties   ->  a tuple of floats that sums to 1.0
                reduction   ->  either 'mean' or 'sum'
        """
        super(ModifiedHuberLoss, self).__init__()
        self.alpha = penalties[0]
        self.beta = penalties[1]
        self.epsilon = epsilon
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, predictions, targets):
        """
        Tensors should be in the shape of [batch, prob_array]
            args:
                predictions     ->  softmax classifier scores
                targets         ->  binary class labels (1, 0)
        """
        elements = predictions.shape[-1]
        weight_adj = targets.clone().detach()
        weight_adj[weight_adj < 1] *= -(elements - 1.0)
        weight_adj[weight_adj == 1] *= self.alpha
        weight_adj[weight_adj > 1] *= self.beta
        loss = torch.where(
            targets * predictions > -1,
            (1 - targets * predictions).clamp(min=0).pow(2),
            -4 * targets * predictions,
            )
        loss = ((loss + self.epsilon) * weight_adj).pow(1 / self.gamma)
        if self.reduction == 'mean':
            return loss.mean() / predictions.shape[0]
        else:
            return loss.sum() / predictions.shape[0]


class Cauldron(torch.nn.Module):
    def __init__(
        self,
        input_labels=('pct_chg', 'price_zs', 'volume_zs'),
        target_labels=('pct_chg',),
        verbosity=1,
        no_caching=True,
        set_weights=False,
        try_cuda=True,
        debug_mode=False,
        ):
        """Predicts the future sentiment from stock data."""
        super(Cauldron, self).__init__()
        self.get_timestamp = lambda: strftime('%Y-%m-%d %H:%M:%S', localtime())
        root_folder = abspath(path.join(dirname(realpath(__file__)), '..'))
        self.root_folder = root_folder
        candelabrum_path = abspath(path.join(root_folder, 'candelabrum'))
        candles_path = path.join(candelabrum_path, 'candelabrum.candles')
        features_path = path.join(candelabrum_path, 'candelabrum.features')
        symbols_path = path.join(candelabrum_path, 'candelabrum.symbols')
        network_path = path.join(root_folder, 'cauldron')
        if not path.exists(network_path):
            mkdir(network_path)
        best_results_path = path.join(network_path, 'best_results')
        if not path.exists(best_results_path):
            mkdir(best_results_path)
        self.best_state_path = path.join(best_results_path, '{0}.state')
        self.new_state_path = path.join(best_results_path, '{0}.{1}.{2}.state')
        self.state_path = path.join(network_path, 'cauldron.state')
        self.backtest_path = path.join(network_path, 'cauldron.backtest')
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
            logging.debug('Enabled debug mode autograd anomaly detection.')
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
        with open(features_path, 'rb') as features_file:
            self.features = pickle.load(features_file)
        features = self.features
        with open(symbols_path, 'rb') as f:
            self.symbols = pickle.load(f)
        symbols = self.symbols
        candelabrum = torch.load(
            candles_path,
            map_location=self.DEVICE_TYPE,
            )
        data_max = max([t.shape[0] for t in candelabrum])
        n_batch = 5
        data_trim = int(data_max)
        if data_trim % n_batch != 0:
            while data_trim % n_batch != 0:
                data_trim -= 1
        symbol_indices = list()
        candle_stack = list()
        for symbol, candles in enumerate(candelabrum):
            if candles.shape[0] != data_max:
                continue
            symbol_indices.append(symbol)
            candle_stack.append(candles[-data_trim:])
        candle_stack = torch.stack(candle_stack).transpose(0, 1)
        self.candle_stack = candle_stack
        self.symbol_indices = symbol_indices
        if input_labels is None:
            input_labels = features
        if target_labels is None:
            target_labels = features
        input_indices = [features.index(l) for l in input_labels]
        target_indices = [features.index(l) for l in target_labels]
        self.dataset_inputs = candle_stack[:, :, input_indices]
        self.dataset_targets = candle_stack[:, :, target_indices]
        n_time, n_symbols, n_features = candle_stack.shape
        n_inputs = len(input_indices)
        n_targets = len(target_indices)
        n_heads = 5
        n_layers = 3
        n_model = n_symbols
        n_gru = n_symbols
        n_hidden = n_symbols
        n_dropout = φ - 1.5
        n_eps = 1e-10
        n_learning_rate = (φ - 1) ** 21 # 1e-1
        n_betas = (0.9, 0.999) # (0.9, 0.999)
        n_weight_decay = (1 / 137) ** 5 # 1e-2
        n_gamma = 3/4
        penalty_delta = 1.3e-5
        n_penalties = (penalty_delta, 1 - penalty_delta)
        self.gru = torch.nn.GRU(
            input_size=n_gru,
            hidden_size=n_symbols,
            num_layers=n_inputs,
            bias=True,
            batch_first=True,
            dropout=n_dropout,
            bidirectional=True,
            device=self.DEVICE,
            dtype=torch.float,
            )
        self.loss_fn = ModifiedHuberLoss(
            epsilon=n_eps,
            gamma=n_gamma,
            penalties=n_penalties,
            reduction='mean',
            )
        activation_fn = 'gelu'
        layer_kwargs = dict(
            d_model=n_model,
            nhead=n_heads,
            dim_feedforward=n_hidden,
            dropout=n_dropout,
            activation=activation_fn,
            layer_norm_eps=n_eps,
            batch_first=True,
            norm_first=True,
            bias=True,
            device=self.DEVICE,
            dtype=torch.float,
            )
        norm_kwargs = dict(
            normalized_shape=n_model,
            eps=n_eps,
            elementwise_affine=True,
            bias=True,
            device=self.DEVICE,
            dtype=torch.float,
            )
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(**layer_kwargs),
            n_layers,
            LayerNorm(**norm_kwargs),
            )
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(**layer_kwargs),
            n_layers,
            LayerNorm(**norm_kwargs),
            )
        self.network = Transformer(
            custom_encoder=self.encoder,
            custom_decoder=self.decoder,
            **layer_kwargs,
            )
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=n_learning_rate,
            betas=n_betas,
            eps=n_eps,
            weight_decay=n_weight_decay,
            amsgrad=True,
            maximize=False,
            foreach=True,
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
            'n_batch': n_batch,
            'n_inputs': n_inputs,
            'n_targets': n_targets,
            'n_gru': n_gru,
            'n_model': n_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'n_hidden': n_hidden,
            'n_dropout': n_dropout,
            'activation_fn': str(repr(activation_fn)),
            'n_eps': n_eps,
            'n_learning_rate': n_learning_rate,
            'n_betas': n_betas,
            'n_weight_decay': n_weight_decay,
            'n_gamma': n_gamma,
            'n_penalties': n_penalties,
            }
        self.validation_results = {
            'best_backtest': 0.0,
            'best_epoch': torch.inf,
            'best_validation': 0.0,
            'calibration_index': 0,
            }
        if verbosity > 0:
            for k, v in self.constants.items():
                constants_str = f'{k.upper()}: {v}'
                print(constants_str)
                logging.info(constants_str)
        self.to(self.DEVICE)
        self.load_state()
        param_count = 0
        for name, param in self.named_parameters():
            if 'shape' in dir(param):
                param_count += math.prod(param.shape, start=1)
        print('Cauldron parameters:', param_count)

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
            if 'encoder' in state:
                self.encoder.load_state_dict(state['encoder'])
            if 'decoder' in state:
                self.decoder.load_state_dict(state['decoder'])
            if 'gru' in state:
                self.gru.load_state_dict(state['decoder'])
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
                logging.error(f'Exception {repr(details.args)}')

    def get_state_dicts(self):
        """Returns module params in a dictionary."""
        return {
            'decoder': self.decoder.state_dict(),
            'encoder': self.encoder.state_dict(),
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'gru': self.gru.state_dict(),
            'validation': dict(self.validation_results),
            }

    def save_state(self, real_path, to_buffer=False, buffer_io=None):
        """Saves the Module."""
        if not to_buffer:
            torch.save(self.get_state_dicts(), real_path)
            if self.verbosity > 1:
                ts = self.get_timestamp()
                msg = f'{ts}: Saved state to {real_path}.'
                logging.info(msg)
        else:
            bytes_obj = self.get_state_dicts()
            bytes_obj = torch.save(bytes_obj, buffer_io)
            return bytes_obj

    def reset_metrics(self):
        """Reset network metrics."""
        self.validation_results['best_backtest'] = 0.0
        self.validation_results['best_epoch'] = torch.inf
        self.validation_results['best_validation'] = 0.0
        self.validation_results['calibration_index'] = 0

    def forward(self, inputs):
        """Returns predictions from inputs."""
        gru = self.gru
        network = self.network
        state = list()
        for t in inputs.split(1):
            g = gru(t[0].transpose(0, 1))[1]
            state.append(network(g, g)[-1].softmax(-1))
        state = torch.stack(state)
        return state

    def train_network(self, target_accuracy=0.995, target_loss=1e-37):
        """Train network on stock data."""
        constants = self.constants
        n_batch = constants['n_batch']
        n_eps = constants['n_eps']
        n_symbols = constants['n_symbols']
        forward = self.forward
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        verbosity = self.verbosity
        load_state = self.load_state
        save_state = self.save_state
        state_path = self.state_path
        n_correct = 0
        n_total = 0
        results = dict()
        candle_stack = self.candle_stack
        ts = self.get_timestamp
        if verbosity > 0:
            print(f'{ts()}: train_stocks routine start.')
        start_time = time.time()
        self.eval()
        input_dataset = self.dataset_inputs[:-n_batch]
        target_dataset = self.dataset_targets[n_batch:]
        final_batch = self.dataset_inputs[-n_batch:]
        n_half = int(input_dataset.shape[0] / 2)
        training_data = DataLoader(
            TensorDataset(input_dataset[:n_half], target_dataset[:n_half]),
            batch_size=n_batch,
            shuffle=False,
            drop_last=True,
            )
        validation_data = DataLoader(
            TensorDataset(input_dataset[n_half:], target_dataset[n_half:]),
            batch_size=n_batch,
            shuffle=False,
            drop_last=True,
            )
        accuracy = 0
        while accuracy < target_accuracy:
            accuracy = 0
            epoch_loss = 0
            batch_count = 0
            epoch_min = 1
            epoch_max = -1
            for batch, targets in training_data:
                targets[targets > 0] = 1
                targets[targets <= 0] = 0
                targets = targets.squeeze(-1)
                predictions = forward(batch)
                loss = loss_fn(predictions, targets) / n_batch
                correct = 0
                for i, b in enumerate(predictions.split(1)):
                    for ii, p in enumerate(b.flatten().split(1)):
                        t = targets[i, ii]
                        if p > 0.5 < t or p <= 0.5 >= t:
                            correct += 1
                correct = correct / (n_batch * n_symbols)
                loss = loss * ((1 - correct) + n_eps)
                loss.backward()
                optimizer.step()
                p_max = predictions.max()
                p_min = predictions.min()
                if p_max > epoch_max:
                    epoch_max = p_max
                if p_min < epoch_min:
                    epoch_min = p_min
                accuracy += correct
                epoch_loss += loss.item()
                batch_count += 1
            accuracy = accuracy / batch_count
            loss = loss / batch_count
            save_state(state_path)
            if verbosity > 0:
                vmsg = f'\n{ts()}:\n    accuracy; {accuracy}, loss; {loss}'
                vmsg += f'\n    epoch_max; {epoch_max}, epoch_min; {epoch_min}'
                print(vmsg)
            if loss <= target_loss:
                break
        # with open(self.backtest_path, 'wb+') as backtest_file:
            # pickle.dump(results, backtest_file)
        # results['validation.metrics']['accuracy'] = accuracy
        # if accuracy > self.validation_results['best_backtest']:
            # self.save_best_result(accuracy, result_key='best_backtest')
        # _path = self.new_state_path.format('backtest', time.time(), accuracy)
        # save_state(_path)
        # if verbosity > 0:
            # accuracy = results['validation.metrics']['accuracy']
            # elapsed = (time.time() - start_time) / 3600
            # time_msg = ts()
            # print(f'{time_msg}: over-all accuracy: {accuracy}')
            # print(f'{time_msg}: elapsed: {elapsed} hours.')
        print(f'{ts()}: train_stocks routine end.')
        return results

    def validate_network(self):
        """Stock back-testing."""
        constants = self.constants
        n_batch = constants['n_batch']
        n_symbols = constants['n_symbols']
        fibify = self.fibify
        index_to_price = self.index_to_price
        forward = self.forward
        symbols = self.symbols
        verbosity = self.verbosity
        n_correct = 0
        n_total = 0
        results = dict()
        candelabrum = self.candelabrum
        ts = self.get_timestamp
        if verbosity > 0:
            print(f'{ts()}: validate_stocks routine start.')
        self.eval()
        for symbol, candles in enumerate(candelabrum):
            sym_ticker = str(symbols[symbol]).upper()
            print(f'{ts()}: {sym_ticker} ({symbol + 1} / {n_symbols})')
            if symbol not in results:
                results[symbol] = dict()
                results[symbol]['accuracy'] = 0
                results[symbol]['correct'] = 0
                results[symbol]['forecast'] = list()
                results[symbol]['symbol'] = sym_ticker
                results[symbol]['total'] = 0
            data_trim = int(candles.shape[0])
            while data_trim % n_batch != 0:
                data_trim -= 1
            candles = candles[-data_trim:, :]
            datasets = self.get_datasets(
                candles,
                benchmark_data=False,
                training=False,
                )
            input_dataset = datasets[0][:-n_batch, :]
            target_dataset = datasets[1][n_batch:, :]
            final_batch = datasets[0][-n_batch:, :]
            validation_data = DataLoader(
                TensorDataset(input_dataset, target_dataset),
                batch_size=n_batch,
                shuffle=False,
                drop_last=True,
                )
            #self.calibrate_network(validation_data)
            forecast = list()
            for batch, targets in validation_data:
                inputs_probs, inputs_ext = fibify(
                    batch[:, 3],
                    extensions=None,
                    )
                targets_probs, targets_ext = fibify(
                    targets[:, 3],
                    extensions=None,
                    )
                predictions = forward(inputs_probs)
                index_match = predictions.argmax(-1) == targets_probs.argmax(-1)
                correct = predictions[index_match].shape[0]
                forecast.append(index_to_price(predictions, inputs_ext))
                results[symbol]['correct'] += correct
                results[symbol]['total'] += n_batch
                n_correct += correct
                n_total += n_batch
            inputs_probs, inputs_ext = fibify(final_batch[:, 3])
            predictions = forward(inputs_probs)[0]
            forecast.append(index_to_price(predictions, inputs_ext))
            forecast = torch.cat(forecast)
            results[symbol]['forecast'] = forecast.flatten().tolist()
            self.load_state()
        results['validation.metrics'] = {
            'correct': n_correct,
            'total': n_total,
            }
        for key in results:
            correct = results[key]['correct']
            total = results[key]['total']
            results[key]['accuracy'] = round((correct / total) * 100, 4)
        with open(self.backtest_path, 'wb+') as backtest_file:
            pickle.dump(results, backtest_file)
        accuracy = results['validation.metrics']['accuracy']
        if accuracy > self.validation_results['best_backtest']:
            self.save_best_result(accuracy, result_key='best_backtest')
        self.save_state(
            self.new_state_path.format(
                'backtest',
                time.time(),
                accuracy,
                ),
            )
        print(f'{ts()}: back-test routine end.')
        return results

    def debug_anomalies(self, *args, **kwargs):
        """Does things..."""
        self.eval()
        self.train()
        return None
