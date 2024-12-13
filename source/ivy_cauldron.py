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


class Cauldron(torch.nn.Module):
    def __init__(
        self,
        input_labels=None,
        target_labels=None,
        target_label='price_zs',
        foci_label='price_zs',
        verbosity=1,
        no_caching=True,
        set_weights=False,
        try_cuda=True,
        debug_mode=False,
        client_mode=False,
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
        best_results_path = path.join(network_path, 'best_results')
        if not path.exists(best_results_path):
            mkdir(best_results_path)
        self.best_state_path = path.join(best_results_path, '{0}.state')
        self.new_state_path = path.join(best_results_path, '{0}.{1}.{2}.state')
        self.state_path = path.join(network_path, 'cauldron.state')
        self.validation_path = path.join(candelabrum_path, 'validation.results')
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
        if input_labels is None:
            input_labels = ['close', 'price_zs', 'price_wema']
            foci_label = 'price_zs'
        self.input_labels = input_labels
        self.inputs_index = [features.index(v) for v in input_labels]
        n_inputs = len(self.inputs_index)
        self.target_index = features.index(target_label)
        self.foci_index = input_labels.index(foci_label)
        n_batch = 10
        self.binary_table = torch.tensor(
            [i for i in product(range(2), repeat=n_batch)],
            device=self.DEVICE,
            dtype=torch.float,
            )
        n_table = self.binary_table.shape[0]
        n_symbols = len(self.symbols)
        n_features = len(self.features)
        n_model = n_table
        n_heads = 2
        n_layers = 3
        n_hidden = 2 ** 8
        n_dropout = 0.118
        n_eps = 1e-6
        n_learning_rate = 9.99e-3
        n_lr_decay = 9.99e-4
        n_weight_decay = 9.99e-7
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
        self.optimizer = torch.optim.Adagrad(
            self.parameters(),
            lr=n_learning_rate,
            lr_decay=n_lr_decay,
            weight_decay=n_weight_decay,
            initial_accumulator_value=n_eps,
            eps=n_eps,
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
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.constants = {
            'n_symbols': n_symbols,
            'n_features': n_features,
            'n_inputs': n_inputs,
            'n_batch': n_batch,
            'n_table': n_table,
            'n_model': n_model,
            'n_heads': n_heads,
            'n_layers': n_layers,
            'n_hidden': n_hidden,
            'n_dropout': n_dropout,
            'activation_fn': str(repr(activation_fn)),
            'n_eps': n_eps,
            'n_learning_rate': n_learning_rate,
            'n_lr_decay': n_lr_decay,
            'n_weight_decay': n_weight_decay,
            'entropy_rate': 0.892,
            }
        self.best_results = {
            'accuracy': 0,
            'correct': 0,
            'total': 0,
            }
        self.inscribed_candles = {
            'last_update': 0,
            'symbols': list(),
            'inputs': list(),
            'targets': list(),
            'batch_targets': list(),
            'final': list(),
            }
        self.to(self.DEVICE)
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
        self.load_state()
        if not client_mode:
            mtime = os.path.getmtime(symbols_path)
            last_inscription = self.inscribed_candles['last_update']
            if verbosity > 1:
                vmsg = '{0}: datasets were updated on {1}.'
                print(vmsg.format(self.get_timestamp(), time.ctime(mtime)))
            if last_inscription == 0 or last_inscription < mtime:
                if verbosity > 1:
                    vmsg = '{0}: found new candles to inscribe.'
                    print(vmsg.format(self.get_timestamp()))
                candles_symbols = list()
                candles_inputs = list()
                candles_targets = list()
                candles_batch_targets = list()
                candles_final = list()
                for symbol_index, symbol in enumerate(self.symbols):
                    if verbosity > 1:
                        vmsg = '{0}: inscribing sigils for {1} ({2} / {3})'
                        print(vmsg.format(
                            self.get_timestamp(),
                            symbol,
                            symbol_index + 1,
                            n_symbols,
                            ))
                    dataset = self.get_dataset(symbol_index)
                    datasets = self.prepare_dataset(dataset)
                    candles_symbols.append(symbol)
                    candles_inputs.append(datasets[0])
                    candles_targets.append(datasets[1])
                    candles_batch_targets.append(datasets[2])
                    candles_final.append(datasets[3])
                self.inscribed_candles['last_update'] = float(mtime)
                self.inscribed_candles['symbols'] = candles_symbols
                self.inscribed_candles['inputs'] = candles_inputs
                self.inscribed_candles['targets'] = candles_targets
                self.inscribed_candles['batch_targets'] = candles_batch_targets
                self.inscribed_candles['final'] = candles_final
                self.save_state(self.state_path)

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
            if 'inscribed_candles' in state:
                self.inscribed_candles = dict(state['inscribed_candles'])
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
            'inscribed_candles': dict(self.inscribed_candles),
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

    def randomize_symbols(self, n_symbols):
        """Randomize the order of symbol indices."""
        import random
        symbol_list = [i for i in range(n_symbols)]
        sorted_symbols = []
        while len(symbol_list) > 0:
            i = random.choice(symbol_list)
            sorted_symbols.append(symbol_list.pop(symbol_list.index(i)))
        return sorted_symbols

    def get_dataset(self, symbol_index):
        """Returns normalized feature data from symbol_index."""
        constants = self.constants
        n_batch = self.constants['n_batch']
        dataset = self.candelabrum[self.symbols[symbol_index]]
        data_steps = int(dataset.shape[0])
        while data_steps % n_batch != 0:
            data_steps -= 1
        dataset = dataset[-data_steps:]
        dataset = self.normalizer(dataset.transpose(0, 1))
        dataset = dataset.transpose(0, 1).tanh()
        return dataset.clone().detach()

    def prepare_dataset(self, dataset):
        """Expand features to fit model and create batch targets."""
        if self.verbosity > 1:
            print(f'{self.get_timestamp()}: preparing data...')
        DEVICE = self.DEVICE
        DTYPE = torch.float
        binary_table = self.binary_table
        constants = self.constants
        n_batch = constants['n_batch']
        n_model = constants['n_model']
        n_eps = constants['n_eps']
        lim_lower = 0.001
        lim_upper = 1 - lim_lower
        def expand_inputs(inputs):
            expanded_inputs = torch.zeros(
                inputs.shape[0],
                n_model,
                device=DEVICE,
                dtype=DTYPE,
                )
            for i, epoch in enumerate(inputs):
                for ii, feature in enumerate(epoch):
                    expanded_inputs[i, ii] += feature
            expanded_inputs[expanded_inputs == 0] += n_eps
            return expanded_inputs
        inputs = expand_inputs(dataset[:-n_batch, self.inputs_index])
        targets = dataset[n_batch:, self.target_index]
        batch_targets = list()
        for batch_start in range(0, targets.shape[0], n_batch):
            batch_stop = batch_start + n_batch
            batch = targets[batch_start:batch_stop]
            target_array = list()
            for pattern in binary_table:
                correct = 0
                for i, t in enumerate(pattern):
                    if batch[i] > 0 < t or batch[i] <= 0 >= t:
                        correct += 1
                target_array.append(correct / n_batch)
            batch_targets.append(target_array)
        batch_targets = torch.tensor(
            batch_targets,
            device=DEVICE,
            dtype=DTYPE,
            )
        batch_targets[batch_targets < lim_lower] = lim_lower
        batch_targets[batch_targets > lim_upper] = lim_upper
        last_batch = expand_inputs(dataset[-n_batch:])
        return (inputs, targets, batch_targets, last_batch)

    def forward(self, inputs, use_mask=False):
        """Returns pattern sentiment from inputs."""
        if use_mask:
            foci_index = self.foci_index
            entropy_rate = self.constants['entropy_rate']
            foci = inputs[:, foci_index].clone().detach()
            entropy = torch.bernoulli(torch.full_like(inputs, entropy_rate))
            inputs *= entropy
            inputs[:, foci_index] = foci
        return self.network(inputs, inputs)[-1].sigmoid()

    def train_network(
        self,
        max_time=3600,
        min_accuracy=0.77,
        max_depth=5,
        n_activations=3,
        ):
        """Train network on stock data."""
        self.train()
        sqrt = math.sqrt
        topk = torch.topk
        ts = self.get_timestamp
        verbosity = self.verbosity
        if verbosity > 1:
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
        network = self.network
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        normalizer = self.normalizer
        save_state = self.save_state
        state_path = self.state_path
        binary_table = self.binary_table
        symbols = self.symbols
        forward = self.forward
        datasets = self.inscribed_candles
        sorted_symbols = self.randomize_symbols(n_symbols)
        def timed_out(start_time):
            elapsed = time.time() - start_time
            if verbosity > 2:
                print(f'{ts()}: {elapsed / 60} minutes elapsed\n')
            if elapsed >= max_time:
                if verbosity > 1:
                    print(f'{ts()}: max time reached...terminating.\n')
                return True
            else:
                return False
        start_time = time.time()
        for symbol_index in sorted_symbols:
            symbol_name = symbols[symbol_index]
            if verbosity > 1:
                print(f'{ts()}: Studying {symbol_name}...')
            dataset_inputs = datasets['inputs'][symbol_index]
            dataset_targets = datasets['targets'][symbol_index]
            dataset_batch_targets = datasets['batch_targets'][symbol_index]
            training_slice = int(dataset_inputs.shape[0] / 2)
            while training_slice % n_batch != 0:
                training_slice -= 1
            training_batches = int(training_slice / n_batch)
            dataset_inputs = dataset_inputs[:training_slice]
            dataset_targets = dataset_targets[:training_slice]
            dataset_batch_targets = dataset_batch_targets[:training_batches]
            #if randint(0, 1) == 1:
            #    dataset_inputs = dataset_inputs.flip(0).flip(-1)
            #    dataset_targets = dataset_targets.flip(-1)
            #    dataset_batch_targets = dataset_batch_targets.flip(0).flip(-1)
            #    print(dataset_inputs[-1])
            #    print(dataset_targets[-1])
            #    print(dataset_batch_targets[-1])
            total_accuracy = 0
            total_loss = 0
            total_steps = 0
            depth = 0
            loss = None
            for batch_start in range(0, dataset_inputs.shape[0], n_batch):
                batch_stop = batch_start + n_batch
                inputs = dataset_inputs[batch_start:batch_stop]
                targets = dataset_targets[batch_start:batch_stop]
                batch_targets = dataset_batch_targets[total_steps]
                encoded_targets = batch_targets.softmax(-1)
                sentiment = forward(inputs, use_mask=True)
                prediction = binary_table[sentiment.argmax()]
                confidence = sentiment.max().item()
                correct = 0
                for i, p in enumerate(prediction):
                    t = targets[i]
                    if t > 0 < p or t <= 0 >= p:
                        correct += 1
                accuracy = correct / n_batch
                cost = (1 - (accuracy / n_batch)) ** (1 / n_batch)
                loss = loss_fn(sentiment.log(), encoded_targets) * cost
                loss.backward()
                optimizer.step()
                total_accuracy += accuracy
                total_loss += loss.item()
                total_steps += 1
                if verbosity > 2:
                    print('pattern index:', sentiment.argmax().item())
                    print('accuracy:', accuracy)
                    print('confidence:', confidence)
                    print('loss:', loss.item())
            total_accuracy = total_accuracy / total_steps
            total_loss = total_loss / total_steps
            save_state(state_path)
            if verbosity > 1:
                vmsg = '{0} {1}: steps {2}; loss {3}; accuracy {4};'
                print(vmsg.format(
                    ts(),
                    symbol_name,
                    total_steps,
                    total_loss,
                    total_accuracy,
                    ))
            if timed_out(start_time):
                break

    def validate_network(self):
        """Test network on new data."""
        self.eval()
        DEVICE = self.DEVICE
        DTYPE = torch.float
        binary_table = self.binary_table
        sqrt = math.sqrt
        topk = torch.topk
        stack = torch.stack
        ts = self.get_timestamp
        verbosity = self.verbosity
        if verbosity > 1:
            print(f'{ts()}: validate_network routine start.\n')
        constants = self.constants
        n_batch = constants['n_batch']
        n_eps = constants['n_eps']
        n_symbols = constants['n_symbols']
        n_features = constants['n_features']
        n_model = constants['n_model']
        n_hidden = constants['n_hidden']
        n_layers = constants['n_layers']
        n_elements = n_batch * n_model
        n_reduce = n_elements * n_hidden
        network = self.network
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        normalizer = self.normalizer
        save_state = self.save_state
        state_path = self.state_path
        symbols = self.symbols
        forward = self.forward
        validation_accuracy = 0
        validation_correct = 0
        validation_total = 0
        metrics = {
            'validation.metrics': {
                'accuracy': 0,
                'correct': 0,
                'total': 0,
                },
            }
        datasets = self.inscribed_candles
        start_time = time.time()
        for symbol_index in range(n_symbols):
            symbol_name = symbols[symbol_index]
            if verbosity > 1:
                print(f'{ts()}: testing {symbol_name}...')
            if symbol_index not in metrics.keys():
                metrics[symbol_index] = {
                    'symbol': symbol_name,
                    'accuracy': 0,
                    'correct': 0,
                    'total': 0,
                    'forecast': [],
                    }
            dataset_inputs = datasets['inputs'][symbol_index]
            dataset_targets = datasets['targets'][symbol_index]
            last_batch = datasets['final'][symbol_index]
            total_accuracy = 0
            total_correct = 0
            total_steps = 0
            total_batches = 0
            predictions = list()
            for batch_start in range(0, dataset_inputs.shape[0], n_batch):
                batch_stop = batch_start + n_batch
                inputs = dataset_inputs[batch_start:batch_stop]
                targets = dataset_targets[batch_start:batch_stop]
                sentiment = forward(inputs, use_mask=False)
                prediction = binary_table[sentiment.argmax()]
                confidence = sentiment.max().item()
                correct = 0
                for i, p in enumerate(prediction):
                    ti = targets[i]
                    if ti < 0 > p or ti > 0 < p:
                        correct += 1
                accuracy = correct / n_batch
                total_correct += correct
                total_steps += n_batch
                validation_correct += correct
                validation_total += n_batch
                total_batches += 1
                predictions.append(prediction.clone().detach())
            sentiment = forward(last_batch, use_mask=False)
            prediction = binary_table[sentiment.argmax()]
            predictions.append(prediction.clone().detach())
            metrics[symbol_index]['forecast'] = stack(predictions).flatten()
            metrics[symbol_index]['correct'] = total_correct
            metrics[symbol_index]['total'] = total_steps
            metrics[symbol_index]['accuracy'] = total_correct / total_steps
            if verbosity > 1:
                vmsg = '{0} {1}: correct {2}; total {3}; accuracy {4};'
                print(vmsg.format(
                    ts(),
                    symbol_name,
                    metrics[symbol_index]['correct'],
                    metrics[symbol_index]['total'],
                    metrics[symbol_index]['accuracy'],
                    ))
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
        if verbosity > 1:
            for k, v in metrics['validation.metrics'].items():
                print(f'{k}: {v}')
            print(f'{ts()}: validate_network routine end.')
        return metrics

