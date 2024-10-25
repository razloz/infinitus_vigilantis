"""Transformer-based Sentiment Rank generator."""
import torch
import logging
import math
import os
import pickle
import time
import warnings
from itertools import product
from random import randint
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
        warnings.simplefilter('ignore')
        torch.set_warn_always(False)
        self.get_timestamp = lambda: strftime('%Y-%m-%d %H:%M:%S', localtime())
        self.set_weights = set_weights
        self.verbosity = verbosity
        root_folder = abspath(path.join(dirname(realpath(__file__)), '..'))
        self.root_folder = root_folder
        candelabrum_path = abspath(path.join(root_folder, 'candelabrum'))
        candles_path = path.join(candelabrum_path, 'candelabrum.candles')
        features_path = path.join(candelabrum_path, 'candelabrum.features')
        symbols_path = path.join(candelabrum_path, 'candelabrum.symbols')
        network_path = path.join(root_folder, 'cauldron')
        if not path.exists(network_path):
            mkdir(network_path)
        wax_path = path.join(network_path, 'wax')
        if not path.exists(wax_path):
            mkdir(wax_path)
        best_results_path = path.join(network_path, 'best_results')
        if not path.exists(best_results_path):
            mkdir(best_results_path)
        self.symbol_state_path = path.join(wax_path, '{0}.state')
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
            torch.set_warn_always(True)
            warnings.simplefilter('always')
            verbosity = 3
            logging.debug('Enabled debug mode autograd anomaly detection.')
        else:
            torch.autograd.set_detect_anomaly(False)
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
                if verbosity > 2:
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
            weights_only=True,
            )
        self.candelabrum = candelabrum
        self.trend_index = features.index(trend_label)
        n_batch = 5
        self.binary_table = torch.tensor(
            [i for i in product(range(2), repeat=n_batch)],
            device=self.DEVICE,
            dtype=torch.float,
            )
        n_table = self.binary_table.shape[0]
        n_symbols = len(self.symbols)
        n_features = len(self.features)
        n_heads = 4
        n_layers = 4
        n_model = n_table
        n_hidden = n_batch * 2 ** 11
        n_dropout = 0.118
        n_eps = 1e-13
        n_betas = (0.9, 0.999)
        zeta = 2 * n_layers * n_hidden
        n_learning_rate = 0.34 / zeta
        n_weight_decay = 9 / zeta
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
        self.normalizer = torch.nn.InstanceNorm1d(
            num_features=n_features,
            eps=n_eps,
            momentum=0.1,
            affine=False,
            track_running_stats=False,
            device=self.DEVICE,
            dtype=torch.float,
            )
        self.constants = {
            'n_table': n_table,
            'n_symbols': n_symbols,
            'n_features': n_features,
            'n_batch': n_batch,
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
            'entropy_rate': 1 - ((((1 + math.sqrt(5)) / 2) - 1) - 0.5),
            'trend_upper': 1, #7/8,
            'trend_lower': -1, #1 - 7/8,
            }
        self.best_results = {
            'accuracy': 0,
            'correct': 0,
            'total': 0,
            }
        self.to(self.DEVICE)
        self.load_state()
        if not debug_mode:
            warnings.simplefilter('default')
        if verbosity > 2:
            for k, v in self.constants.items():
                constants_str = f'{k.upper()}: {v}'
                print(constants_str)
                logging.info(constants_str)
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
                state = torch.load(
                    state_path,
                    map_location=self.DEVICE_TYPE,
                    weights_only=True,
                    )
            if 'best_results' in state:
                self.best_results = dict(state['best_results'])
            if 'decoder' in state:
                self.decoder.load_state_dict(state['decoder'])
            if 'encoder' in state:
                self.encoder.load_state_dict(state['encoder'])
            if 'network' in state:
                self.network.load_state_dict(state['network'])
            if 'normalizer' in state:
                self.normalizer.load_state_dict(state['normalizer'])
            if 'optimizer' in state:
                self.optimizer.load_state_dict(state['optimizer'])
        except FileNotFoundError:
            if self.verbosity > 2:
                logging.info('No state found, creating default.')
            if self.set_weights:
                i = 1
                if self.verbosity > 2:
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
            'best_results': dict(self.best_results),
            'decoder': self.decoder.state_dict(),
            'encoder': self.encoder.state_dict(),
            'network': self.network.state_dict(),
            'normalizer': self.normalizer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }

    def save_state(self, real_path, to_buffer=False, buffer_io=None):
        """Saves the Module."""
        if not to_buffer:
            torch.save(self.get_state_dicts(), real_path)
            if self.verbosity > 2:
                ts = self.get_timestamp()
                msg = f'{ts}: Saved state to {real_path}.'
                logging.info(msg)
        else:
            bytes_obj = self.get_state_dicts()
            bytes_obj = torch.save(bytes_obj, buffer_io)
            return bytes_obj

    def random_dataset(self, auto_flip=True):
        """Choose one symbol from the candelabrum for training."""
        if auto_flip:
            flip_data = True if randint(0, 1) == 1 else False
        else:
            flip_data = False
        selection = randint(0, self.constants['n_symbols'] - 1)
        dataset_symbol = self.symbols[selection]
        dataset = self.candelabrum[dataset_symbol].clone().detach()
        if flip_data:
            dataset = dataset.flip(0)
        return dataset_symbol, dataset

    def randomize_symbols(self, n_symbols):
        """Randomize the order of symbol indices."""
        import random
        symbol_list = [i for i in range(n_symbols)]
        sorted_symbols = []
        while len(symbol_list) > 0:
            i = random.choice(symbol_list)
            sorted_symbols.append(symbol_list.pop(symbol_list.index(i)))
        return sorted_symbols

    def get_dataset(self, symbol_index, training=False):
        constants = self.constants
        n_batch = self.constants['n_batch']
        tt_upper = constants['trend_upper']
        tt_lower = constants['trend_lower']
        trend_index = self.trend_index
        dataset = self.candelabrum[self.symbols[symbol_index]]
        data_steps = int(dataset.shape[0])
        while data_steps % n_batch != 0:
            data_steps -= 1
        # data_split = int(data_steps / 2)
        # if training:
            # dataset = dataset[:data_split]
        # else:
            # dataset = dataset[-data_split:]
        # dataset = torch.cat([
            # dataset.flip(0).clone().detach(),
            # dataset.clone().detach(),
            # ])
        dataset = dataset[-data_steps:]
        dataset = self.normalizer(dataset.transpose(0, 1))
        dataset = dataset.transpose(0, 1).tanh()
        trends = dataset[:, trend_index]
        trends[trends > 0] = tt_upper
        trends[trends <= 0] = tt_lower
        return dataset.clone().detach()

    def prepare_dataset(self, dataset):
        """Expand features to fit model and create batch targets."""
        print(f'{self.get_timestamp()}: preparing data...')
        binary_table = self.binary_table
        constants = self.constants
        trend_index = self.trend_index
        n_batch = constants['n_batch']
        n_model = constants['n_model']
        inputs = dataset[:-n_batch]
        targets = dataset[n_batch:]
        batch_targets = list()
        expanded_inputs = torch.zeros(inputs.shape[0], n_model)
        for i, epoch in enumerate(inputs):
            for ii, feature in enumerate(epoch):
                expanded_inputs[i, ii] += feature
        for batch_start in range(0, targets.shape[0], n_batch):
            batch_stop = batch_start + n_batch
            batch = targets[batch_start:batch_stop, trend_index]
            target_array = list()
            for pattern in binary_table:
                correct = 0
                for i, trend in enumerate(pattern):
                    if batch[i] > 0 and trend > 0:
                        correct += 1
                    elif batch[i] <=0 and trend <= 0:
                        correct += 1
                target_array.append(correct / n_batch)
            batch_targets.append(target_array)
        batch_targets = torch.tensor(batch_targets)
        return (expanded_inputs, batch_targets)

    def forward(self, inputs, use_mask=False):
        """Returns sentiment from inputs."""
        constants = self.constants
        trend_index = self.trend_index
        features = constants['n_features']
        entropy_rate = constants['entropy_rate']
        if use_mask:
            foci = inputs[:, trend_index].clone().detach()
            entropy = torch.bernoulli(torch.full_like(inputs, entropy_rate))
            inputs *= entropy
            inputs[:, trend_index] = foci
        outputs = self.network(inputs, inputs).sigmoid()
        return outputs[-1]

    def train_network(self, target_accuracy=0.60, target_loss=1e-37):
        """Train network on stock data."""
        self.train()
        sqrt = math.sqrt
        ts = self.get_timestamp
        verbosity = self.verbosity
        if verbosity > 0:
            print(f'{ts()}: train_network routine start.\n')
        constants = self.constants
        n_batch = constants['n_batch']
        n_eps = constants['n_eps']
        n_symbols = constants['n_symbols']
        n_features = constants['n_features']
        n_model = constants['n_model']
        n_hidden = constants['n_hidden']
        n_layers = constants['n_layers']
        n_elements = n_batch * n_model
        n_reduce = 2 * n_layers * n_elements * n_hidden
        get_dataset = self.get_dataset
        network = self.network
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        normalizer = self.normalizer
        save_state = self.save_state
        state_path = self.state_path
        trend_index = self.trend_index
        symbols = self.symbols
        forward = self.forward
        prepare_dataset = self.prepare_dataset
        sorted_symbols = self.randomize_symbols(n_symbols)
        start_time = time.time()
        for symbol_index in sorted_symbols:
            symbol_name = symbols[symbol_index]
            dataset = get_dataset(symbol_index, training=True)
            dataset_inputs, dataset_targets = prepare_dataset(dataset)
            epoch_steps = range(dataset_inputs.shape[0])
            print(f'{ts()}: Studying {symbol_name}...')
            accuracy = 0
            epoch_loss = 0
            total_steps = 0
            for batch_start in range(0, dataset_inputs.shape[0], n_batch):
                batch_stop = batch_start + n_batch
                inputs = dataset_inputs[batch_start:batch_stop]
                targets = dataset_targets[total_steps]
                sentiment = forward(inputs, use_mask=True)
                acc = targets[sentiment.argmax()]
                accuracy += acc
                cost = ((1 - acc) ** (1 / n_batch)) - 1
                cost = abs(n_batch * n_elements * cost) + n_eps
                loss = loss_fn(sentiment, targets) / n_reduce
                loss = loss ** sqrt(1 / cost)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                total_steps += 1
                if verbosity > 2:
                    vv = (accuracy / total_steps, epoch_loss / total_steps)
                    print(sentiment)
                    print(f'accuracy: {vv[0]} ; loss: {vv[1]} ;\n')
            accuracy = accuracy / total_steps
            epoch_loss = epoch_loss / total_steps
            save_state(state_path)
            if verbosity > 0:
                vmsg = '{0} {1}: steps {2}; loss {3}; accuracy {4};\n'
                print(vmsg.format(
                    ts(),
                    symbol_name,
                    total_steps,
                    epoch_loss,
                    accuracy,
                    ))

    def validate_network(self):
        """Test network on new data."""
        self.eval()
        ts = self.get_timestamp
        verbosity = self.verbosity
        if verbosity > 0:
            print(f'{ts()}: validate_network routine start.')
        network = self.network
        normalizer = self.normalizer
        n_batch = self.constants['n_batch']
        n_eps = self.constants['n_eps']
        n_stack_size = constants['n_stack_size']
        tt_upper = constants['trend_upper']
        tt_lower = constants['trend_lower']
        stack = torch.stack
        stack_range = range(n_stack_size)
        trend_index = self.trend_index
        accuracy = 0
        total_steps = 0
        metrics = {
            'validation.metrics': {
                'accuracy': 0,
                'correct': 0,
                'total': 0,
                },
            }
        validation_accuracy = 0
        validation_correct = 0
        validation_total = 0
        for ndx, (symbol_name, dataset) in enumerate(self.candelabrum.items()):
            print(f'{ts()} Testing: {ndx}; {symbol_name};')
            if ndx not in metrics.keys():
                metrics[ndx] = {
                    'symbol': symbol_name,
                    'accuracy': 0,
                    'correct': 0,
                    'total': 0,
                    'forecast': [],
                    'target': [],
                    }
            dataset = normalizer(dataset.transpose(0, 1))
            dataset = dataset.transpose(0, 1).sigmoid()
            trends = dataset[:, trend_index]
            trends[trends > 0.5] = tt_upper
            trends[trends <= 0.5] = tt_lower
            dataset_inputs = dataset[:-n_batch]
            dataset_targets = dataset[n_batch:]
            for step in range(dataset_inputs.shape[0]):
                inputs = dataset_inputs[step]
                inputs = stack([inputs for _ in stack_range])
                targets = dataset_targets[step]
                sentiment = network(inputs, inputs)[-1].sigmoid()
                ft = sentiment[trend_index]
                tt = targets[trend_index]
                correct = any([bool(ft > 0.5 < tt), bool(ft <= 0.5 >= tt)])
                metrics[ndx]['forecast'].append(float(ft))
                metrics[ndx]['target'].append(float(tt))
                metrics[ndx]['correct'] += 1 if correct else 0
                metrics[ndx]['total'] += 1
                validation_correct += 1 if correct else 0
                validation_total += 1
            accuracy = metrics[ndx]['correct'] / metrics[ndx]['total']
            metrics[ndx]['accuracy'] = accuracy
            for final_batch in dataset[-n_batch:]:
                final_batch = stack([final_batch for _ in stack_range])
                sentiment = network(final_batch, final_batch)[-1]
                ft = sentiment[trend_index].sigmoid()
                metrics[ndx]['forecast'].append(float(ft))
                metrics[ndx]['target'].append(0.5)
            print(f'{ts()} Results: {ndx}; {symbol_name}; accuracy {accuracy};')
        validation_accuracy = validation_correct / validation_total
        metrics['validation.metrics']['accuracy'] = validation_accuracy
        metrics['validation.metrics']['correct'] = validation_correct
        metrics['validation.metrics']['total'] = validation_total
        if validation_accuracy > self.best_results['accuracy']:
            self.save_state(
                self.new_state_path.format(
                    validation_accuracy,
                    validation_correct,
                    validation_total,
                    ),
                )
            self.best_results['accuracy'] = validation_accuracy
            self.best_results['correct'] = validation_correct
            self.best_results['total'] = validation_total
        with open(self.validation_path, 'wb+') as file_obj:
            pickle.dump(metrics, file_obj)
        if verbosity > 0:
            for k, v in metrics['validation.metrics'].items():
                print(f'{k}: {v}')
            print(f'{ts()}: validate_network routine end.')
        return metrics
