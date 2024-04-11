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
from torch.nn.functional import interpolate
from torch.nn.init import uniform_
from torch.utils.data import DataLoader, TensorDataset
from torch import topk, stack
from torch.fft import rfft
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2024, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'gardneri'
ᶓ = math.e
π = math.pi
φ = (1 + math.sqrt(5)) / 2


class Cauldron(torch.nn.Module):
    def __init__(
        self,
        input_labels=('price_zs', 'price_sdev', 'price_wema', 'close'),
        target_labels=('price_zs', 'price_sdev', 'price_wema', 'close'),
        verbosity=1,
        no_caching=True,
        set_weights=True,
        try_cuda=True,
        debug_mode=False,
        ):
        """Predicts the future sentiment from stock data."""
        super(Cauldron, self).__init__()
        self.get_timestamp = lambda: strftime('%Y-%m-%d %H:%M:%S', localtime())
        root_folder = abspath(path.join(dirname(realpath(__file__)), '..'))
        self.root_folder = root_folder
        candelabrum_path = abspath(path.join(root_folder, 'candelabrum'))
        benchmarks_path = path.join(candelabrum_path, 'candelabrum.benchmarks')
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
        benchmarks = torch.load(
            benchmarks_path,
            map_location=self.DEVICE_TYPE,
            )
        with open(symbols_path, 'rb') as f:
            self.symbols = pickle.load(f)
        symbols = self.symbols
        n_symbols = len(symbols)
        candelabrum = torch.load(
            candles_path,
            map_location=self.DEVICE_TYPE,
            )
        if input_labels is None:
            input_labels = features
        if target_labels is None:
            target_labels = features
        input_indices = [features.index(l) for l in input_labels]
        target_indices = [features.index(l) for l in target_labels]
        n_benchmarks, n_time, n_features = benchmarks.shape
        n_batch = 34
        n_inputs = len(input_indices)
        n_targets = len(target_indices)
        self.training_targets = torch.zeros(
            n_batch,
            n_targets,
            device=self.DEVICE,
            dtype=torch.float,
            )
        self.validation_targets = torch.zeros(
            n_batch,
            n_targets,
            device=self.DEVICE,
            dtype=torch.float,
            )
        self.fib_ext = torch.tensor(
            [0.00, 0.118, 0.250, 0.382, 0.500, 0.618, 0.750, 0.882, 1.00],
            device=self.DEVICE,
            dtype=torch.float,
            )
        n_model = self.fib_ext.shape[0] * 2
        n_hidden = 64
        n_heads = 3
        n_layers = 128
        n_dropout = 1 - (φ - 1)
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
        n_learning_rate = 0.0999
        n_betas = (0.9, 0.999)
        n_weight_decay = 0.0099
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=n_learning_rate,
            betas=n_betas,
            eps=n_eps,
            weight_decay=n_weight_decay,
            amsgrad=True,
            foreach=True,
            maximize=False,
            )
        self.piphi = torch.tensor(
            [
                [-1,  π,  1],
                [-φ,  0,  φ],
                [-1, -π, 1],
            ],
            device=self.DEVICE,
            dtype=torch.float,
            )
        piphi_indices = [i for i in range(9) if i != 4]
        self.piphi_indices = list(combinations(piphi_indices, 2))
        self.piphi_indices = [[*i] for i in self.piphi_indices]
        self.benchmarks = benchmarks
        self.candelabrum = candelabrum
        self.set_weights = set_weights
        self.verbosity = verbosity
        self.input_indices = input_indices
        self.target_indices = target_indices
        self.constants = {
            'n_benchmarks': n_benchmarks,
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
            'n_eps': n_eps,
            'n_learning_rate': n_learning_rate,
            'n_betas': n_betas,
            'n_weight_decay': n_weight_decay,
            'batch_affix': n_eps ** 2,
            }
        self.validation_results = {
            'best_backtest': 0.0,
            'best_epoch': torch.inf,
            'best_validation': 0.0,
            }
        if verbosity > 1:
            for k, v in self.constants.items():
                constants_str = f'{k.upper()}: {v}'
                print(constants_str)
                logging.info(constants_str)
        self.to(self.DEVICE)
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
                logging.error(f'Exception {repr(details.args)}')

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
                ts = self.get_timestamp()
                msg = f'{ts}: Saved state to {real_path}.'
                logging.info(msg)
        else:
            bytes_obj = self.get_state_dicts()
            bytes_obj = torch.save(bytes_obj, buffer_io)
            return bytes_obj

    def fibify(self, price_array, extensions=None):
        """Fibonacci Extensions."""
        if extensions is None:
            price_delta = price_array.max() - price_array.min()
            last_close = price_array[-1]
            fib_ext = price_delta * self.fib_ext
            fib_high = last_close + fib_ext
            fib_low = last_close - fib_ext
            extensions = torch.cat([fib_high.flip(0), fib_low]).unsqueeze(-1)
        ext_target = torch.full_like(extensions, 0.0)
        price_locations = list()
        for closing_price in price_array:
            ext_target *= 0
            ext_target += closing_price
            ext_dist = torch.cdist(extensions, ext_target, p=1.0)
            ext_dist = ((1 - ext_dist.mean(-1)) ** 2).softmax(-1)
            price_locations.append(ext_dist.clone().detach())
        price_locations = torch.stack(price_locations)
        return (price_locations, extensions)

    def index_to_price(self, predictions, extensions):
        """"""
        prices = list()
        for probs in predictions:
            ext_index = probs.argmax()
            price = extensions[ext_index]
            prices.append(price)
        prices = torch.cat(prices)
        return prices.flatten().unsqueeze(-1)

    def forward(self, inputs):
        """Returns predictions from inputs."""
        return self.network(inputs, inputs).softmax(-1)

    def get_datasets(self, tensor_data, benchmark_data=False, training=False):
        """Prepare tensor data for network input."""
        if not benchmark_data:
            tensor_data = tensor_data.unsqueeze(0)
        inputs = tensor_data[:, :, self.input_indices].clone().detach()
        targets = tensor_data[:, :, self.target_indices].clone().detach()
        if not benchmark_data:
            inputs = inputs.squeeze(0)
            targets = targets.squeeze(0)
        return (inputs.clone().detach(), targets.clone().detach())

    def save_best_result(self, metric, result_key='best_epoch', file_name=None):
        """Set aside good results to avoid merging."""
        if self.debug_mode:
            self.debug_anomalies(file_name=f'{time.time()}')
        self.validation_results[result_key] = float(metric)
        if file_name is None:
            file_name = self.best_state_path.format(metric)
        self.save_state(file_name)

    def reset_metrics(self):
        """Reset network metrics."""
        self.validation_results['best_backtest'] = 0.0
        self.validation_results['best_epoch'] = torch.inf
        self.validation_results['best_validation'] = 0.0

    def train_network(
        self,
        checkpoint=100,
        epoch_samples=5,
        hours=3,
        validate=True,
        quicksave=False,
        ):
        """Batched training over hours."""
        get_state_dicts = self.get_state_dicts
        constants = self.constants
        n_batch = constants['n_batch']
        verbosity = self.verbosity
        forward = self.forward
        load_state = self.load_state
        save_state = self.save_state
        state_path = self.state_path
        validate_network = self.validate_network
        snapshot_path = path.join(self.root_folder, 'cauldron', '{}.state')
        epoch = 0
        elapsed = 0
        best_epoch = self.validation_results['best_epoch']
        benchmarks = self.benchmarks
        n_benchmarks = benchmarks.shape[0] - 1
        datasets = self.get_datasets(
            benchmarks,
            benchmark_data=True,
            training=True,
            )
        input_dataset = datasets[0][:, :-n_batch, :]
        target_dataset = datasets[1][:, n_batch:, :]
        batch_max = input_dataset.shape[1] - n_batch - 1
        def random_batch():
            benchmark = randint(0, n_benchmarks)
            batch_start = randint(0, batch_max)
            batch_end = batch_start + n_batch
            return (
                input_dataset[benchmark, batch_start:batch_end],
                target_dataset[benchmark, batch_start:batch_end],
                )
        optimizer = self.optimizer
        loss_fn = torch.nn.MSELoss(reduction='mean')
        sqrt = math.sqrt
        debug_mode = self.debug_mode
        debug_anomalies = self.debug_anomalies
        save_best_result = self.save_best_result
        get_timestamp = self.get_timestamp
        batch_range = range(2, n_batch + 1)
        epoch_checkpoint = False
        checkpoint_error = 0
        inf = torch.inf
        fibify = self.fibify
        if verbosity > 0:
            logging.info('Training started.')
        start_time = time.time()
        while elapsed < hours:
            self.train()
            epoch += 1
            epoch_error = 0
            n_sample = 0
            loss = None
            optimizer.zero_grad()
            while n_sample < epoch_samples:
                batch, targets = random_batch()
                inputs_probs, inputs_ext = fibify(
                    batch[:, 3],
                    extensions=None,
                    )
                targets_probs, targets_ext = fibify(
                    targets[:, 3],
                    extensions=inputs_ext,
                    )
                predictions = forward(inputs_probs)
                if loss is None:
                    loss = loss_fn(predictions, targets_probs)
                else:
                    loss += loss_fn(predictions, targets_probs)
                n_sample += 1
            loss.backward()
            optimizer.step()
            epoch_error = sqrt(loss.item() / epoch_samples)
            if debug_mode:
                debug_anomalies(file_name=f'{time.time()}')
            elapsed = (time.time() - start_time) / 3600
            ts = get_timestamp()
            epoch_checkpoint = epoch % checkpoint == 0
            if quicksave or epoch_checkpoint:
                save_state(state_path)
            if epoch_checkpoint and verbosity > 0:
                print(inputs_probs)
                print('inputs_probs', inputs_probs.shape)
                print(targets_probs)
                print('targets_probs', targets_probs.shape)
                print(predictions)
                print('predictions', predictions.shape)
                epoch_msg = ''
                epoch_msg += '\n*********************************************'
                epoch_msg += f'\ntimestamp: {ts}'
                epoch_msg += f'\nepoch: {epoch}'
                epoch_msg += f'\nelapsed: {elapsed}'
                epoch_msg += f'\nerror: {epoch_error}'
                epoch_msg += '\n*********************************************\n'
                print(epoch_msg)
                logging.info(epoch_msg)
        if validate:
            r = validate_network()
            if verbosity > 0:
                ts = get_timestamp()
                for k, v in r.items():
                    print(f'{ts}: {k}: {v}')
        if quicksave is False and not epoch_checkpoint:
            save_state(state_path)
        if verbosity > 0:
            logging.info(f'Training finished after {elapsed} hours.')

    def validate_network(self):
        """Benchmarks back-testing."""
        constants = self.constants
        n_batch = constants['n_batch']
        fibify = self.fibify
        index_to_price = self.index_to_price
        forward = self.forward
        verbosity = self.verbosity
        n_correct = 0
        n_total = 0
        results = dict()
        ts = self.get_timestamp
        if verbosity > 0:
            print(f'{ts()}: validation routine start.')
            logging.info('Starting validation routine.')
        self.eval()
        benchmarks = self.benchmarks
        datasets = self.get_datasets(
            benchmarks,
            benchmark_data=True,
            training=False,
            )
        input_dataset = datasets[0][:, :-n_batch, :]
        target_dataset = datasets[1][:, n_batch:, :]
        final_batch = datasets[0][:, -n_batch:, :]
        for benchmark, candles in enumerate(input_dataset):
            if benchmark not in results:
                results[benchmark] = dict()
                results[benchmark]['accuracy'] = 0
                results[benchmark]['correct'] = 0
                results[benchmark]['forecast'] = list()
                results[benchmark]['symbol'] = 'QQQ' if benchmark < 2 else 'SPY'
                results[benchmark]['total'] = 0
            validation_data = DataLoader(
                TensorDataset(candles, target_dataset[benchmark]),
                batch_size=n_batch,
                shuffle=False,
                drop_last=True,
                )
            forecast = list()
            for batch, targets in validation_data:
                inputs_probs, inputs_ext = fibify(
                    batch[:, 3],
                    extensions=None,
                    )
                targets_probs, targets_ext = fibify(
                    targets[:, 3],
                    extensions=inputs_ext,
                    )
                predictions = forward(inputs_probs)
                forecast.append(index_to_price(predictions, inputs_ext))
                predictions_indices = predictions.argmax(-1)
                targets_indices = targets_probs.argmax(-1)
                index_check = predictions_indices == targets_indices
                correct = 0
                for result in index_check:
                    if result is True:
                        correct += 1
                results[benchmark]['correct'] += correct
                results[benchmark]['total'] += n_batch
                n_correct += correct
                n_total += n_batch
            inputs_probs, inputs_ext = fibify(final_batch[benchmark, :, 3])
            predictions = forward(inputs_probs)
            forecast.append(index_to_price(predictions, inputs_ext))
            forecast = torch.cat(forecast)
            results[benchmark]['forecast'] = forecast.flatten().tolist()
        results['validation.metrics'] = {
            'accuracy': 0.0,
            'correct': n_correct,
            'total': n_total,
            }
        for key in results:
            correct = results[key]['correct']
            total = results[key]['total']
            if total > 0:
                results[key]['accuracy'] = round((correct / total) * 100, 4)
        with open(self.validation_path, 'wb+') as validation_file:
            pickle.dump(results, validation_file)
        accuracy = results['validation.metrics']['accuracy']
        if accuracy > self.validation_results['best_validation']:
            self.save_best_result(accuracy, result_key='best_validation')
        self.save_state(
            self.new_state_path.format(
                'validate',
                time.time(),
                accuracy,
                ),
            )
        print(f'{ts()}: validation routine end.')
        return results

    def backtest_network(self):
        """Candelabrum back-testing."""
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
            print(f'{ts()}: back-test routine start.')
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
            print('data_trim before: ', data_trim)
            while data_trim % n_batch != 0:
                data_trim -= 1
            print('data_trim after: ', data_trim)
            print('candles before: ', candles.shape)
            candles = candles[-data_trim:, :]
            print('candles after: ', candles.shape)
            datasets = self.get_datasets(
                candles,
                benchmark_data=False,
                training=False,
                )
            input_dataset = datasets[0][:-n_batch, :]
            target_dataset = datasets[1][n_batch:, :]
            print('datasets mod batch: ', datasets[0].shape[0] % n_batch == 0)
            final_batch = datasets[0][-n_batch:, :]
            validation_data = DataLoader(
                TensorDataset(input_dataset, target_dataset),
                batch_size=n_batch,
                shuffle=False,
                drop_last=True,
                )
            forecast = list()
            for batch, targets in validation_data:
                inputs_probs, inputs_ext = fibify(
                    batch[:, 3],
                    extensions=None,
                    )
                targets_probs, targets_ext = fibify(
                    targets[:, 3],
                    extensions=inputs_ext,
                    )
                predictions = forward(inputs_probs)
                forecast.append(index_to_price(predictions, inputs_ext))
                predictions_indices = predictions.argmax(-1)
                targets_indices = targets_probs.argmax(-1)
                index_check = predictions_indices == targets_indices
                correct = 0
                for result in index_check:
                    if result is True:
                        correct += 1
                results[symbol]['correct'] += correct
                results[symbol]['total'] += n_batch
                n_correct += correct
                n_total += n_batch
            inputs_probs, inputs_ext = fibify(final_batch[:, 3])
            predictions = forward(inputs_probs)
            forecast.append(index_to_price(predictions, inputs_ext))
            forecast = torch.cat(forecast)
            results[symbol]['forecast'] = forecast.flatten().tolist()
            print('total days:', results[symbol]['total'])
            print('forecast len:', len(results[symbol]['forecast']))
            #print(results[symbol]['forecast'])
            #print('symbol', symbol, 'shape', forecast.shape)
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

    def inscribe_sigil(self, charts_path):
        """Plot final batch predictions from the candelabrum."""
        symbols = self.symbols
        forward = self.forward
        n_forecast = self.constants['n_forecast']
        forecast_path = path.join(charts_path, '{0}_forecast.png')
        plt.style.use('dark_background')
        plt.rcParams['figure.figsize'] = [2, 10]
        fig = plt.figure()
        #midline_args = dict(color='red', linewidth=1.5, alpha=0.8)
        xlim_max = n_forecast + 1
        with open(self.backtest_path, 'rb') as backtest_file:
            metrics = pickle.load(backtest_file)
        for symbol, results in metrics.items():
            if symbol == 'validation.metrics':
                continue
            forecast = sum(results['forecast']) / len(results['forecast'])
            ax = fig.add_subplot()
            ax.set_xlabel('Up-trend Likelihood', fontweight='bold')
            ax.grid(True, color=(0.3, 0.3, 0.3))
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 1)
            plt.setp(ax.xaxis.get_ticklabels()[:], visible=False)
            plt.setp(ax.yaxis.get_ticklabels()[:], visible=False)
            bar_width = ax.get_tightbbox(fig.canvas.get_renderer()).get_points()
            bar_width = ((bar_width[1][0] - bar_width[0][0]) / n_forecast) * 0.4
            ax.plot(
                [1, 1],
                [0, forecast],
                color='#FFA600',
                linewidth=bar_width,
                alpha=0.98,
                )
            ax.plot(
                [0, xlim_max],
                [0.5, 0.5],
                color=(0.5, 0.5, 0.5),
                linewidth=1.5,
                alpha=0.9,
                )
            fig_title = 'Sentiment'
            fig_title += f'\n({round(forecast, 4) * 100}%)'
            fig.suptitle(fig_title, fontsize=18)
            plt.savefig(forecast_path.format(symbols[symbol]))
            fig.clf()
            plt.clf()
        plt.close()
        del(fig)
        gc.collect()
        return metrics

    def debug_anomalies(self, *args, **kwargs):
        """Does things..."""
        get_timestamp = self.get_timestamp
        self.eval()
        _param_types_ = ('bias', 'weight', 'in_proj_bias', 'in_proj_weight')
        _params_ = dict()
        _params_count_ = 0
        for name, param in self.network.named_parameters():
            _params_count_ += int(param.flatten().shape[0])
            if param.requires_grad:
                param_keys = name.split('.')
                if len(param_keys) != 5:
                    continue
                _id_ = f'{param_keys[0]}.{param_keys[3]}'
                _type_ = param_keys[-1]
                if _type_ not in _param_types_:
                    continue
                if _id_ not in _params_:
                    _params_[_id_] = {'bias': [], 'weight': []}
                if 'bias' in _type_:
                    _type_ = 'bias'
                else:
                    _type_ = 'weight'
                _params_[_id_][_type_].append(param.clone().detach())
        for _id_ in _params_:
            for _type_ in _params_[_id_]:
                p = _params_[_id_][_type_]
                s = [len(p), *p[-1].shape]
                _params_[_id_][_type_] = torch.cat(
                    p,
                    ).view(*s).transpose(0, -1).sum(-1)
        fname = str(kwargs['file_name']) if 'file_name' in kwargs else ''
        img_path = path.join(self.root_folder, 'logs', fname + '{0}.png')
        plt.style.use('dark_background')
        plt.rcParams['figure.figsize'] = [38.4, 21.6]
        cmap = plt.get_cmap('ocean')
        fig = plt.figure(dpi=100, constrained_layout=False)
        pane_color = ((0.3, 0.3, 0.3, 0.5))
        spec = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)
        subplots = list()
        def plot_parameter(bias, weight, plot_title, row_num, col_num):
            """3D graph of network parameter."""
            subplots.append(
                fig.add_subplot(
                    spec[row_num, col_num],
                    projection='3d',
                    )
                )
            subplot = subplots[-1]
            subplot.w_xaxis.set_pane_color(pane_color)
            subplot.w_yaxis.set_pane_color(pane_color)
            subplot.w_zaxis.set_pane_color(pane_color)
            subplot.set_title(plot_title)
            z_space = (weight * bias)
            z_x = weight.shape[0]
            z_y = weight.shape[1]
            x_space, y_space = torch.meshgrid(
                torch.linspace(1, z_x, steps=z_x),
                torch.linspace(1, z_y, steps=z_y),
                indexing='ij',
                )
            plot_data = subplot.plot_surface(
                x_space,
                y_space,
                z_space,
                cmap=cmap,
                )
            fig.colorbar(
                plot_data,
                ax=subplot,
                orientation='horizontal',
                shrink=0.7,
                aspect=20,
                )
        n_encoders = 0
        n_decoders = 0
        for _id_ in _params_:
            if 'multihead' in _id_:
                continue
            if len(_params_[_id_]['weight'].shape) != 1:
                if _id_.split('.')[0] == 'encoder':
                    row_num = 0
                    col_num = n_encoders
                    n_encoders += 1
                else:
                    row_num = 1
                    col_num = n_decoders
                    n_decoders += 1
                plot_parameter(
                    _params_[_id_]['bias'],
                    _params_[_id_]['weight'],
                    _id_,
                    row_num,
                    col_num,
                    )
        multihead_id = 'decoder.multihead_attn'
        plot_parameter(
            _params_[multihead_id]['bias'],
            _params_[multihead_id]['weight'],
            multihead_id,
            row_num=1,
            col_num=3,
            )
        fig_title = "Network Parameter's Bias Modified Weights ("
        fig_title += get_timestamp() + ')'
        fig_title += f'\n({_params_count_} Total Network Parameters)'
        fig.suptitle(fig_title, fontsize=18)
        plt.savefig(img_path.format('.network.parameters'))
        fig.clf()
        plt.clf()
        plt.close()
        del(fig)
        del(_params_)
        gc.collect()
        self.train()
