"""Transformer-based Sentiment Rank generator."""
import torch
import logging
import math
import os
import pickle
import time
from time import localtime, strftime
from os import path, mkdir, environ
from os.path import dirname, realpath, abspath
from torch.nn.init import uniform_
from torch.nn import Transformer, LayerNorm
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, TensorDataset
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2024, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'gardneri'


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
                predictions     ->  real-valued classifier scores
                targets         ->  true binary class labels (+1, -1)
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
        input_labels=None,
        target_labels=None,
        trend_label='trend',
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
        self.trend_index = features.index(trend_label)
        self.dataset_inputs = candle_stack[:, :, input_indices]
        self.dataset_targets = candle_stack[:, :, target_indices]
        n_time, n_symbols, n_features = candle_stack.shape
        n_inputs = len(input_indices)
        n_targets = len(target_indices)
        n_heads = n_batch
        n_layers = 3
        n_model = n_symbols
        n_hidden = 2 ** 11
        n_dropout = 0.3
        n_eps = 1e-10
        n_learning_rate = 1e-5
        n_betas = (0.9, 0.9999)
        n_weight_decay = 1e-8
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
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
            amsgrad=False,
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
        except FileNotFoundError:
            if self.verbosity > 0:
                logging.info('No state found, creating default.')
            if self.set_weights:
                i = 0.01 * self.constants['n_eps']
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

    def train_network(self, target_accuracy=0.995, target_loss=1e-37):
        """Train network on stock data."""
        constants = self.constants
        n_batch = constants['n_batch']
        n_eps = constants['n_eps']
        n_symbols = constants['n_symbols']
        n_features = constants['n_features']
        n_elements = n_features * n_symbols
        network = self.network
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        verbosity = self.verbosity
        load_state = self.load_state
        save_state = self.save_state
        state_path = self.state_path
        trend_index = self.trend_index
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
            TensorDataset(
                input_dataset[:n_half],
                target_dataset[:n_half],
                ),
            batch_size=n_batch,
            shuffle=False,
            drop_last=True,
            )
        validation_data = DataLoader(
            TensorDataset(
                input_dataset[n_half:],
                target_dataset[n_half:],
                ),
            batch_size=n_batch,
            shuffle=False,
            drop_last=True,
            )
        accuracy = 0
        batch_steps = range(n_batch)
        accuracy_check = range(n_symbols)
        while accuracy < target_accuracy:
            accuracy = 0
            epoch_loss = 0
            total_steps = 0
            for batch, targets in training_data:
                for step in batch_steps:
                    inputs = batch[step].transpose(0, 1).sigmoid()
                    target = targets[step].transpose(0, 1).sigmoid()
                    predictions = network(inputs, target)
                    loss = loss_fn(predictions, target)
                    future_trend = predictions[trend_index]
                    target = target[trend_index]
                    correct = 0
                    for i in accuracy_check:
                        condi_pos = future_trend[i] >= 0.5 <= target[i]
                        condi_neg = future_trend[i] < 0.5 > target[i]
                        correct += 1 if condi_pos or condi_neg else 0
                    correct = correct / n_symbols
                    loss = (loss / n_elements) * ((1 - correct) + n_eps)
                    loss.backward()
                    optimizer.step()
                    accuracy += correct
                    epoch_loss += loss.item()
                    total_steps += 1
                    #vmsg = 'steps {0}; loss {1}; accuracy {2};'
                    #print(vmsg.format(total_steps, loss.item(), correct))
            accuracy = accuracy / total_steps
            epoch_loss = epoch_loss / total_steps
            save_state(state_path)
            if verbosity > 0:
                print(f'\n{ts()}:\n    accuracy {accuracy}; loss {epoch_loss};')
            if epoch_loss <= target_loss:
                break
        print(f'{ts()}: train_stocks routine end.')
        return results
