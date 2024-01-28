"""Transformer-based Sentiment Rank generator."""
import torch
import logging
import os
import pickle
import time
from math import sqrt
from random import randint
from os import path, mkdir, environ
from os.path import dirname, realpath, abspath
from torch.nn import HuberLoss
from torch.nn.functional import interpolate
from torch.nn.init import uniform_
from torch.utils.data import DataLoader, TensorDataset
from torch import topk
φ = (1 + sqrt(5)) / 2
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2024, Daniel Ward'
__license__ = 'GPL v3'
__version__ = 'gardneri'


class Cauldron(torch.nn.Module):
    def __init__(
        self,
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
        self.best_state_path = path.join(best_results_path, '{0}.{1}.state')
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
        with open(features_path, 'rb') as features_file:
            features = pickle.load(features_file)
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
        n_time, n_features = benchmarks.shape
        n_batch = 5
        n_inputs = len(input_indices)
        n_targets = len(target_indices)
        self.temp_targets = torch.zeros(
            n_batch,
            n_targets,
            device=self.DEVICE,
            dtype=torch.float,
            )
        n_heads = 2
        n_layers = 128
        n_hidden = 2 ** n_batch
        n_model = n_heads * n_hidden
        n_dropout = φ - 1.5
        n_eps = (1 / 137) ** 3
        self.batch_prefix = torch.tensor(
            [float(n_eps)],
            device=self.DEVICE,
            dtype=torch.float,
            )
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
            self.network.parameters(),
            lr=n_learning_rate,
            betas=n_betas,
            eps=n_eps,
            weight_decay=n_weight_decay,
            amsgrad=True,
            foreach=True,
            maximize=False,
            )
        self.loss_fn = HuberLoss(
            reduction='sum',
            delta=φ,
            )
        self.benchmarks = benchmarks
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
                logging.info(f'Saved state to {real_path}.')
        else:
            bytes_obj = self.get_state_dicts()
            bytes_obj = torch.save(bytes_obj, buffer_io)
            return bytes_obj

    def forward(self, inputs):
        """Returns predictions from inputs."""
        n_model = self.constants['n_model']
        inputs_shape = inputs.shape
        inputs = interpolate(
            inputs.view(1, *inputs_shape),
            size=n_model,
            mode='area',
            ).view(inputs_shape[0], n_model)
        predictions = self.network(inputs, inputs).sigmoid().mean(-1)
        return predictions.unsqueeze(1)

    def train_network(
        self,
        checkpoint=1,
        epoch_samples=100,
        hours=168,
        validate=False,
        quicksave=False,
        reinforce=True,
        ):
        """Batched training over hours."""
        constants = self.constants
        n_batch = constants['n_batch']
        n_eps = constants['n_eps']
        n_symbols = constants['n_symbols']
        n_targets = constants['n_targets']
        verbosity = self.verbosity
        forward = self.forward
        save_state = self.save_state
        state_path = self.state_path
        validate_network = self.validate_network
        snapshot_path = path.join(self.root_folder, 'cauldron', '{}.state')
        stack = torch.stack
        epoch = 0
        elapsed = 0
        best_epoch = torch.inf
        benchmarks = self.benchmarks
        input_dataset = benchmarks[:-n_batch, self.input_indices]
        target_dataset = benchmarks[n_batch:, self.target_indices]
        batch_max = input_dataset.shape[0] - n_batch - 1
        batch_prefix = self.batch_prefix
        def random_batch():
            batch_start = randint(0, batch_max)
            batch_end = batch_start + n_batch
            return (
                input_dataset[batch_start:batch_end],
                target_dataset[batch_start:batch_end],
                )
        temp_targets = self.temp_targets
        optimizer = self.optimizer
        loss_fn = self.loss_fn
        debug_mode = self.debug_mode
        debug_anomalies = self.debug_anomalies
        batch_range = range(2, n_batch + 1)
        if verbosity > 0:
            logging.info('Training started.')
        start_time = time.time()
        while elapsed < hours:
            self.train()
            epoch += 1
            n_error = 0
            epoch_error = 0
            if not reinforce:
                optimizer.zero_grad()
            while n_error < epoch_samples:
                batch, targets = random_batch()
                temp_targets *= 0
                temp_targets[targets > 0] += 1
                if reinforce:
                    for i in batch_range:
                        optimizer.zero_grad()
                        batch_step = batch[:i]
                        target_step = temp_targets[:i]
                        predictions = forward(batch_step)
                        loss = loss_fn(predictions, target_step)
                        loss.backward()
                        optimizer.step()
                else:
                    predictions = forward(batch)
                    loss = loss_fn(predictions, targets)
                    loss.backward()
                n_error += 1
                epoch_error += loss.item()
                if quicksave:
                    save_state(state_path)
                if verbosity > 1:
                    print(f'\nbatch: {n_error}')
                    print(f'targets: \n{temp_targets}')
                    print(f'predictions: \n{predictions}')
                    print(f'loss: {epoch_error / n_error}', '\n')
            if not reinforce:
                optimizer.step()
            if debug_mode:
                debug_anomalies(file_name=f'{time.time()}')
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
        n_symbols = constants['n_symbols']
        n_targets = constants['n_targets']
        verbosity = self.verbosity
        forward = self.forward
        best_results = self.validation_results['accuracy']
        n_correct = 0
        n_total = 0
        results = dict()
        candelabrum = self.candelabrum
        input_indices = self.input_indices
        target_indices = self.target_indices
        temp_targets = self.temp_targets
        if verbosity > 0:
            ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            print(f'{ts}: validation routine start.')
            logging.info('Starting validation routine.')
        self.eval()
        for symbol, candles in enumerate(candelabrum):
            if symbol not in results:
                results[symbol] = dict()
                results[symbol]['accuracy'] = 0
                results[symbol]['correct'] = 0
                results[symbol]['total'] = 0
                results[symbol]['forecast'] = list()
            validation_data = DataLoader(
                TensorDataset(
                    candles[:-n_batch, input_indices],
                    candles[n_batch:, target_indices],
                    ),
                batch_size=n_batch,
                shuffle=False,
                drop_last=True,
                )
            for batch, targets in validation_data:
                temp_targets *= 0
                temp_targets[targets > 0] += 1
                targets = temp_targets.flatten()
                predictions = forward(batch)
                correct = 0
                for i, p in enumerate(predictions):
                    t = targets[i]
                    if any([p > 0.5 and t == 1, p <= 0.5 and t == 0]):
                        correct += 1
                results[symbol]['correct'] += correct
                results[symbol]['total'] += n_batch
                n_correct += correct
                n_total += n_batch
            predictions = forward(candles[-n_batch:, input_indices])
            results[symbol]['forecast'] = predictions.flatten().tolist()
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
            _path = self.best_state_path.format(time.time(), epoch_accuracy)
            self.save_state(_path)
            if verbosity > 0:
                logging.info(f'New best accuracy of {epoch_accuracy}% saved.')
        ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        print(f'{ts}: validation routine end.')
        return results

    def inscribe_sigil(self, charts_path):
        """Plot final batch predictions from the candelabrum."""
        import matplotlib.pyplot as plt
        symbols = self.symbols
        forward = self.forward
        n_batch = self.constants['n_batch']
        forecast_path = path.join(charts_path, '{0}_forecast.png')
        plt.style.use('dark_background')
        plt.rcParams['figure.figsize'] = [10, 2]
        fig = plt.figure()
        midline_args = dict(color='red', linewidth=1.5, alpha=0.8)
        with open(self.validation_path, 'rb') as validation_file:
            metrics = pickle.load(validation_file)
        for symbol, results in metrics.items():
            if symbol == 'validation.metrics':
                continue
            forecast = results['forecast']
            ax = fig.add_subplot()
            ax.set_ylabel('Confidence', fontweight='bold')
            ax.grid(True, color=(0.3, 0.3, 0.3))
            ax.set_xlim(0, 9)
            ax.set_ylim(0, 1)
            bar_width = ax.get_tightbbox(fig.canvas.get_renderer()).get_points()
            bar_width = ((bar_width[1][0] - bar_width[0][0]) / 7) * 0.5
            for i, p in enumerate(forecast):
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
        return metrics

    def debug_anomalies(self, *args, **kwargs):
        """Does things..."""
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        from mpl_toolkits import mplot3d
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
        fig_title += time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) + ')'
        fig_title += f'\n({_params_count_} Total Network Parameters)'
        fig.suptitle(fig_title, fontsize=18)
        plt.savefig(img_path.format('.network.parameters'))
        fig.clf()
        plt.clf()
        plt.close()
