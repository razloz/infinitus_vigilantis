"""Three blind mice to predict the future."""
import time
import torch
import torch.nn as nn
import traceback
import source.ivy_commons as icy
from torch.optim import NAdam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as WarmRestarts
from math import sqrt
from os.path import abspath
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, verbosity=0, *args, **kwargs):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._features_ = [
            'open', 'high', 'low', 'close', 'vol_wma_price', 'trend',
            'fib_retrace_0.236', 'fib_retrace_0.382', 'fib_retrace_0.5',
            'fib_retrace_0.618', 'fib_retrace_0.786', 'fib_retrace_0.886',
            'price_zs', 'price_sdev', 'price_wema', 'price_dh', 'price_dl',
            'price_mid', 'cdl_change',
            ]
        self._targets_ = 'price_med'
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._prefix_ = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._state_path_ = abspath('./rnn')
        self.verbosity = int(verbosity)
        self._n_features_ = len(self._features_)
        self._n_hidden_ = int(self._n_features_ * 3)
        self._n_layers_ = 1024
        self._tolerance_ = 0.01618033988749894
        self.torch_batch = nn.BatchNorm1d(
            self._n_features_,
            )
        self.torch_gate = nn.GRU(
            input_size=self._n_features_,
            hidden_size=self._n_hidden_,
            num_layers=self._n_layers_,
            bias=True,
            batch_first=True,
            dropout=float(self._tolerance_ * 10),
            device=self._device_,
            )
        self.torch_linear = nn.Linear(
            in_features=self._n_hidden_,
            out_features=1,
            bias=True,
            )
        self.torch_loss = nn.HuberLoss(
            reduction='mean',
            delta=0.99,
            )
        self.metrics = nn.ParameterDict()
        self.optimizer = NAdam(
            params=self.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=0,
            momentum_decay=3e-3,
            foreach=True,
            )
        self.scheduler = WarmRestarts(
            optimizer=self.optimizer,
            T_0=5,
            T_mult=1,
            eta_min=1e-5,
            )
        self.tensors = dict(
            coated=None,
            sealed=list(),
            inputs=None,
            targets=None,
            )
        self.threshold = 0
        self.wax = 0
        self.predictions = nn.ParameterDict()
        self.to(self._device_)
        if self.verbosity > 1:
            print(self._prefix_, 'set device_type to', self._device_type_)
            print(self._prefix_, 'set n_features to', self._n_features_)
            print(self._prefix_, 'set tolerance to', self._tolerance_)
            print(self._prefix_, 'set input_size to', self._n_features_)
            print(self._prefix_, 'set hidden_size to', self._n_hidden_)
            print(self._prefix_, 'set num_layers to', self._n_layers_)
            print(self._prefix_, 'set dropout to', float(self._tolerance_ * 10))

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        try:
            state_path = f'{self._state_path_}/moirai.state'
            if call_type == 0:
                state = torch.load(state_path, map_location=self._device_type_)
                self.load_state_dict(state['moirai'])
                for key, value in state['predictions'].items():
                    self.predictions[key] = value
                if self.verbosity > 2:
                    print(self._prefix_, 'Loaded RNN state.')
            elif call_type == 1:
                torch.save(
                    {
                        'moirai': self.state_dict(),
                        'predictions': self.predictions,
                        },
                    state_path,
                    )
                if self.verbosity > 2:
                    print(self._prefix_, 'Saved RNN state.')
        except Exception as details:
            if self.verbosity > 2:
                print(self._prefix_, 'Encountered an exception.')
                traceback.print_exception(details)

    def __time_step__(self, indices, mode, epochs=0, study=False):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles
           Let Awen contain the wax"""
        batch_start, batch_stop = indices
        if mode == 'train':
            self.train()
        elif mode == 'eval':
            self.eval()
        wax = self.wax
        if study is True:
            threshold = self.threshold
            candles = self.tensors['inputs'][batch_start:batch_stop]
            targets = self.tensors['targets'][batch_start:batch_stop]
            targets = torch.vstack(tuple(t for t in targets.split(1)))
            batch_size = candles.shape[0]
            self.optimizer.zero_grad()
            candles = (self(candles) * wax)
            loss = self.torch_loss(candles, targets)
            loss.backward()
            difference = (candles - targets)
            correct = difference[difference >= -threshold]
            correct = correct[correct <= threshold].shape[0]
            self.metrics['loss'] = loss.item()
            self.tensors['coated'] = candles.clone()
            self.tensors['sealed'].append(candles.clone())
            self.metrics['acc'][0] += int(correct)
            self.metrics['acc'][1] += int(batch_size)
            if not self.metrics['mae']:
                self.metrics['mae'] = 0
            if not self.metrics['mse']:
                self.metrics['mse'] = 0
            self.metrics['mae'] += difference.abs().sum().item()
            self.metrics['mse'] += (difference ** 2).sum().item()
        else:
            with torch.no_grad():
                try:
                    self.metrics['mae'] = self.metrics['mae'] / epochs
                    self.metrics['mse'] = sqrt(self.metrics['mse'] / epochs)
                except Exception as details:
                    if self.verbosity > 0:
                        print(self._prefix_, 'Encountered an exception.')
                        traceback.print_exception(details)
                finally:
                    candles = self.tensors['inputs'][batch_start:]
                    candles = (self(candles) * wax)
                    self.tensors['coated'] = candles.clone()
                    self.tensors['sealed'].append(candles.clone())
        if self.verbosity == 2 and study is False:
            msg = f'{self._prefix_} ' + '{}: {}'
            lr = self.scheduler._last_lr
            print(msg.format('Learning Rate', lr))
            print(msg.format('Accuracy', self.metrics['acc']))
            print(msg.format('Loss', self.metrics['loss']))
            print(msg.format('Mean Absolute Error', self.metrics['mae']))
            print(msg.format('Mean Squared Error', self.metrics['mse']))
        elif self.verbosity > 2:
            msg = f'{self._prefix_} ' + '{}: {}'
            lr = self.scheduler._last_lr
            print(msg.format('Learning Rate', lr))
            print(msg.format('candles:\n', candles))
            print(msg.format('candles shape:\n', candles.shape))
            if study is True:
                print(msg.format('targets:\n', targets))
                print(msg.format('targets shape:\n', targets.shape))
            print('coated:\n', len(self.tensors['coated']))

    def forward(self, candles):
        """Gate activation."""
        candles = self.torch_batch(candles)
        candles = candles.tanh().log_softmax(0)
        candles = self.torch_gate(candles)
        candles = self.torch_linear(candles[0])
        candles = torch.exp(candles).relu()
        return candles.clone()

    def research(self, symbol, dataframe, mode, timeout=1, epoch_save=False):
        """Moirai research session, fully stocked with cheese and drinks."""
        _TK_ = icy.TimeKeeper()
        time_start = _TK_.reset
        df_keys = dataframe.keys()
        if self._targets_ not in df_keys:
            return False
        for key in self._features_:
            if key not in df_keys:
                return False
        self.__manage_state__(call_type=0)
        symbol = symbol.upper()
        time_step = self.__time_step__
        prefix = self._prefix_
        p_tensor = self._p_tensor_
        tolerance = self._tolerance_
        hstack = torch.hstack
        vstack = torch.vstack
        tensor = torch.tensor
        target = self._targets_
        features = self._features_
        fib_seq = icy.FibonacciSequencer().skip(6).next(4)
        small_batch = fib_seq[:2]
        large_batch = fib_seq[2:]
        large_batch.reverse()
        if self.verbosity > 1:
            print(prefix, f'fib batch seq: {fib_seq}')
            print(prefix, f'final batch size: {large_batch[-1]}')
        reverse_candles = True
        total_epochs = 0
        for i in range(2):
            batch_sizes = (small_batch[i], large_batch[i])
            for batch_size in batch_sizes:
                batch_index = 0
                batch_fit = 0
                for n_batch in range(len(dataframe.index)):
                    batch_index += 1
                    if batch_index % batch_size == 0:
                        batch_fit += batch_size
                        batch_index = 0
                timestamps = int(batch_fit)
                batch_range = range(timestamps)
                for _ in range(2):
                    if reverse_candles:
                        candles = dataframe[::-1]
                        candles = candles[-batch_fit:]
                    else:
                        candles = dataframe[-batch_fit:]
                    self.release_tensors()
                    self.reset_metrics()
                    inputs = candles[features].to_numpy()
                    targets = candles[target].to_numpy()[batch_size:]
                    self.tensors['inputs'] = tensor(inputs, **p_tensor)
                    self.tensors['inputs'].requires_grad_(True)
                    self.tensors['targets'] = tensor(targets, **p_tensor)
                    self.wax = round(self.tensors['targets'].mean().item(), 3)
                    self.threshold = self.wax * self._tolerance_
                    if self.verbosity > 1:
                        print(prefix, 'inputs:', self.tensors['inputs'].shape)
                        print(prefix, 'targets:', self.tensors['targets'].shape)
                        print(prefix, 'wax:', self.wax)
                        print(prefix, 'threshold:', self.threshold)
                    target_accuracy = 99.0
                    target_loss = 0.1 / batch_size
                    target_mae = 1e-3
                    target_mse = target_mae ** 2
                    if self.verbosity > 1:
                        print(prefix, 'target_accuracy', target_accuracy)
                        print(prefix, 'target_loss', target_loss)
                        print(prefix, 'target_mae', target_mae)
                        print(prefix, 'target_mse', target_mse)
                    epochs = 0
                    accuracy = 0
                    while accuracy < target_accuracy:
                        time_update = _TK_.update[0]
                        if self.verbosity > 1:
                            msg = f'{prefix} epoch {epochs} '
                            msg += f'elapsed time {time_update}.'
                            print(msg)
                        break_check = all((
                            self.metrics['loss'],
                            self.metrics['mae'],
                            self.metrics['mse'],
                            ))
                        if break_check:
                            break_condition = all((
                                self.metrics['loss'] < target_loss,
                                self.metrics['mae'] < target_mae,
                                self.metrics['mse'] < target_mse,
                                ))
                        else:
                            break_condition = False
                        if epochs == timeout or break_condition:
                            break
                        epochs += 1
                        total_epochs += 1
                        self.release_candles()
                        self.reset_metrics()
                        accuracy = 0
                        last_batch = 0
                        for i in batch_range:
                            indices = (i, i + batch_size)
                            if i < last_batch:
                                continue
                            else:
                                last_batch = int(indices[1])
                            if last_batch <= batch_range[-1]:
                                time_step(indices, mode, study=True)
                            else:
                                n_total = 0
                                n_correct = 0
                                indices = (-batch_size, None)
                                time_step(indices, mode, epochs=timestamps)
                                n_correct = self.metrics['acc'][0]
                                n_total = self.metrics['acc'][1]
                                n_wrong = n_total - n_correct
                                accuracy = abs((n_wrong - n_total) / n_total)
                                accuracy = round(100 * accuracy, 3)
                                break
                        self.optimizer.step()
                        self.scheduler.step()
                        if epoch_save:
                            self.__manage_state__(call_type=1)
                        if self.verbosity > 0:
                            msg = f'A moment of research yielded '
                            msg += f'an accuracy of {accuracy}% '
                            msg += f'(epoch: {total_epochs}, '
                            msg += f'batch: {batch_size}, '
                            msg += f'reverse: {reverse_candles})'
                            print(prefix, msg)
                    reverse_candles = not reverse_candles
        last_pred = float(self.tensors['coated'][-1].item())
        last_price = float(candles['close'][-1])
        proj_gain = float(((last_pred - last_price) / last_price) * 100)
        timestamp = time.time()
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        if symbol not in self.predictions.keys():
            self.predictions[symbol] = dict(num_epochs=total_epochs)
        else:
            self.predictions[symbol]['num_epochs'] += total_epochs
        self.predictions[symbol]['threshold'] = self.threshold
        self.predictions[symbol]['accuracy'] = accuracy
        self.predictions[symbol]['mae'] = self.metrics['mae']
        self.predictions[symbol]['mse'] = self.metrics['mse']
        self.predictions[symbol]['loss'] = self.metrics['loss']
        self.predictions[symbol]['last_price'] = round(last_price, 3)
        self.predictions[symbol]['batch_size'] = batch_size
        self.predictions[symbol]['batch_pred'] = round(last_pred, 3)
        self.predictions[symbol]['proj_gain'] = round(proj_gain, 3)
        self.predictions[symbol]['proj_timestamp'] = timestamp
        self.predictions[symbol]['proj_time_str'] = time_str
        self.tensors['coated'] = self.tensors['coated'].detach().cpu().numpy()
        self.tensors['sealed'] = vstack(self.tensors['sealed']).H
        self.tensors['sealed'] = self.tensors['sealed'].detach().cpu().numpy()
        self.__manage_state__(call_type=1)
        if self.verbosity > 0:
            if self.verbosity > 1:
                for k, v in self.predictions.items():
                    print(f'{prefix} {symbol} predictions {k}: {v}')
            epoch_str = 'epoch' if epochs == 1 else 'epochs'
            msg = 'After {} {}, an accuracy of {}% was realized.'
            print(prefix, msg.format(epochs, epoch_str, accuracy))
            time_elapsed = _TK_.final[0]
            print(f'{prefix} final elapsed time {time_elapsed}.')
            print('')
        return True

    def release_candles(self):
        """Clear stored candles from memory."""
        self.tensors['coated'] = None
        self.tensors['sealed'] = list()
        return True

    def release_tensors(self):
        """Clear stored tensors from memory."""
        self.tensors = dict(
            coated=None,
            sealed=list(),
            inputs=None,
            targets=None,
            )
        return True

    def reset_metrics(self):
        """Clear previous metrics."""
        self.metrics['acc'] = [0, 0]
        self.metrics['loss'] = None
        self.metrics['mae'] = None
        self.metrics['mse'] = None
        return True
