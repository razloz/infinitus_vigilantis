"""Three blind mice to predict the future."""
import time
import torch
import torch.nn as nn
import traceback
import source.ivy_commons as icy
from torch.optim import Rprop
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as WarmRestarts
from math import sqrt
from os.path import abspath
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class GRNN(nn.Module):
    """Gated Recurrent Neural Network"""
    def __init__(self, name, n_output, **kwargs):
        """Register modules and candles parameter."""
        super(GRNN, self).__init__()
        self._device_ = kwargs['device']
        self._n_output_ = int(n_output)
        self.__gru__ = nn.GRU(**kwargs)
        self.__batch_fn__ = nn.BatchNorm1d(int(kwargs['hidden_size']))
        self.__pool_fn__ = nn.AdaptiveMaxPool1d(self._n_output_)
        self.candles = nn.ParameterDict()
        self.loss_fn = nn.HuberLoss(reduction='sum', delta=0.97)
        self.metrics = nn.ParameterDict()
        self.name = str(name)
        self.optimizer = Rprop(self.__gru__.parameters(), lr=0.1, foreach=True)
        self.scheduler = WarmRestarts(self.optimizer, int(kwargs['num_layers']))
        self.to(self._device_)

    def forward(self, inputs):
        """Batch pooling after GRU activation."""
        inputs = self.__gru__(inputs)
        if type(inputs) is tuple: inputs = inputs[0]
        inputs = self.__batch_fn__(inputs)
        inputs = self.__pool_fn__(inputs).H
        outputs = list()
        for i in range(self._n_output_):
            stacked_output = torch.vstack([t for t in inputs[i].split(1)])
            outputs.append(stacked_output.clone())
        outputs = torch.hstack(outputs)
        return outputs

    def give(self, cheese):
        """Sample cheese, create candles."""
        if type(cheese) is list:
            cheese = torch.hstack([*cheese])
        candles = self(cheese)
        return candles

    def release_tensors(self):
        """Remove batch and target tensors from memory."""
        self.candles = nn.ParameterDict()
        self.candles['batch_inputs'] = None
        self.candles['targets_base'] = None
        self.candles['targets_loss'] = None
        self.candles['targets_pred'] = None

    def reset_candles(self):
        """Clear previous candles."""
        self.candles['coated'] = None
        self.candles['sealed'] = list()

    def reset_metrics(self):
        """Clear previous metrics."""
        self.metrics['acc'] = [0, 0]
        self.metrics['loss'] = 0
        self.metrics['mae'] = 0
        self.metrics['mse'] = 0


class Cauldron(GRNN):
    """A wax encrusted cauldron sits before you, bubbling occasionally."""
    def __init__(self, *args, **kwargs):
        """Assists in the creation of candles."""
        super(Cauldron, self).__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """*bubble**bubble* *bubble*"""
        return super(Cauldron, self).forward(*args, **kwargs)


class Moira(GRNN):
    """A clever, cheese candle constructing, future predicting mouse."""
    def __init__(self, *args, **kwargs):
        """The number of the counting shall be three."""
        super(Moira, self).__init__(*args, **kwargs)


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, n_features, verbosity=0, min_size=90, *args, **kwargs):
        """Beckon the Moirai and Cauldron."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._prefix_ = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._state_path_ = abspath('./rnn/moirai.state')
        self.verbosity = int(verbosity)
        self._batch_size_ = 8
        self._min_size_ = int(min_size)
        self._n_features_ = int(n_features)
        self._n_hidden_ = int(self._n_features_ * self._batch_size_)
        self._tolerance_ = 1.618033988749894e-3
        if self.verbosity > 1:
            print(self._prefix_, 'set device_type to', self._device_type_)
            print(self._prefix_, 'set batch_size to', self._batch_size_)
            print(self._prefix_, 'set min_size to', self._min_size_)
            print(self._prefix_, 'set n_features to', self._n_features_)
            print(self._prefix_, 'set n_hidden to', self._n_hidden_)
            print(self._prefix_, 'set tolerance to', self._tolerance_)
        p_gru = dict()
        p_gru['num_layers'] = int(self._min_size_ / 3)
        p_gru['bias'] = True
        p_gru['batch_first'] = True
        p_gru['dropout'] = self._tolerance_
        p_gru['device'] = self._device_
        p_cauldron = dict(p_gru)
        p_cauldron['input_size'] = 3
        p_cauldron['hidden_size'] = int(self._batch_size_ * 3)
        p_moira = dict(p_gru)
        p_moira['input_size'] = self._n_features_
        p_moira['hidden_size'] = self._n_hidden_
        # Awen the sentient cauldron.
        self.Awen = Cauldron('Awen', 3, **p_cauldron)
        # Atropos the mouse, sister of Clotho and Lachesis.
        self.Atropos = Moira('Atropos', 1, **p_moira)
        # Clotho the mouse, sister of Lachesis and Atropos.
        self.Clotho = Moira('Clotho', 1, **p_moira)
        # Lachesis the mouse, sister of Atropos and Clotho.
        self.Lachesis = Moira('Lachesis', 1, **p_moira)
        # Store predictions for sorting top picks
        self.predictions = nn.ParameterDict()
        self.to(self._device_)

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        try:
            p = self._state_path_
            if call_type == 0:
                d = self._device_type_
                self.load_state_dict(torch.load(p, map_location=d))
                if self.verbosity > 2:
                    print(self._prefix_, 'Loaded RNN state.')
            elif call_type == 1:
                torch.save(self.state_dict(), p)
                if self.verbosity > 2:
                    print(self._prefix_, 'Saved RNN state.')
        except Exception as details:
            if self.verbosity > 2:
                print(self._prefix_, 'Encountered an exception.')
                traceback.print_exception(details)

    def __time_step__(self, mouse, indices, epochs=0, study=False):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles"""
        vstack = torch.vstack
        if study is True:
            threshold = self._tolerance_
            cheese = mouse.candles['batch_inputs'][indices[0]:indices[1]]
            t_base = mouse.candles['targets_base'][indices[0]:indices[1]]
            t_loss = mouse.candles['targets_loss'][indices[0]:indices[1]]
            t_pred = mouse.candles['targets_pred'][indices[0]:indices[1]]
            t_base = vstack([t for t in t_base.split(1)])
            t_loss = vstack([t for t in t_loss.split(1)])
            t_pred = vstack([t for t in t_pred.split(1)])
            batch_size = cheese.shape[0]
            mouse.optimizer.zero_grad()
            candles = mouse.give(cheese)
            loss = mouse.loss_fn(candles, t_loss)
            loss.backward()
            mouse.metrics['loss'] = int(loss.item())
            mouse.optimizer.step()
            mouse.scheduler.step()
            candles = (t_base + (t_base * candles))
            mouse.candles['coated'] = candles.clone()
            difference = (candles - t_pred)
            correct = difference[difference >= -threshold]
            correct = correct[correct <= threshold].shape[0]
            mouse.metrics['acc'][0] += int(correct)
            mouse.metrics['acc'][1] += int(batch_size)
            mouse.metrics['mae'] += difference.abs().sum().item()
            mouse.metrics['mse'] += (difference ** 2).sum().item()
        else:
            with torch.no_grad():
                try:
                    mouse.metrics['mae'] = mouse.metrics['mae'] / epochs
                    mouse.metrics['mse'] = sqrt(mouse.metrics['mse'] / epochs)
                except Exception as details:
                    if self.verbosity > 0:
                        msg = '{} Encountered an exception.'
                        print(self._prefix_, msg.format(mouse.name))
                        traceback.print_exception(details)
                finally:
                    cheese = mouse.candles['batch_inputs'][indices[0]:]
                    t_base = mouse.candles['targets_base'][indices[0]:]
                    t_base = vstack([t for t in t_base.split(1)])
                    candles = mouse.give(cheese)
                    candles = (t_base + (t_base * candles))
                    mouse.candles['coated'] = candles.clone()
        if self.verbosity == 2 and study is False:
            lr = mouse.scheduler.get_last_lr()[0]
            acc = mouse.metrics['acc']
            mae = mouse.metrics['mae']
            mse = mouse.metrics['mse']
            msg = f'{self._prefix_} {mouse.name} ' + '{}: {}'
            print(msg.format('Learning Rate', lr))
            print(msg.format('Accuracy', acc))
            print(msg.format('Mean Absolute Error', mae))
            print(msg.format('Mean Squared Error', mse))
        elif self.verbosity > 2:
            print('candles:\n', candles)
            print('candles shape:\n', candles.shape)
            print('t_base:\n', t_base)
            print('t_base shape:\n', t_base.shape)
            print('t_loss:\n', t_loss)
            print('t_loss shape:\n', t_loss.shape)
            print('t_pred:\n', t_pred)
            print('t_pred shape:\n', t_pred.shape)
            print('coated:\n', len(mouse.candles['coated']))

    def research(self, symbol, candles, timeout=1):
        """Moirai research session, fully stocked with cheese and drinks."""
        if not all((
            len(candles.keys()) == self._n_features_,
            len(candles.index) >= self._min_size_
            )): return False
        self.__manage_state__(call_type=0)
        Awen = self.Awen
        Atropos = self.Atropos
        Clotho = self.Clotho
        Lachesis = self.Lachesis
        time_step = self.__time_step__
        batch_size = self._batch_size_
        prefix = self._prefix_
        p_tensor = self._p_tensor_
        tolerance = self._tolerance_
        hstack = torch.hstack
        vstack = torch.vstack
        tensor = torch.tensor
        candles = candles[self._min_size_:]
        timestamps = len(candles.index)
        batch_range = range(timestamps - batch_size)
        base_targets = ('price_wema', 'price_wema', 'cdl_median')
        loss_targets = ('wdist_close', 'wdist_open', 'wdist_median')
        pred_targets = ('close', 'open', 'price_wema')
        moirai = [Atropos, Clotho, Lachesis]
        _cdls = candles.to_numpy()
        for m_i, mouse in enumerate(moirai):
            _mc = mouse.candles
            _base = candles[base_targets[m_i]].to_numpy()
            _loss = candles[loss_targets[m_i]].to_numpy()
            _pred = candles[pred_targets[m_i]].to_numpy()
            _mc['batch_inputs'] = tensor(_cdls, **p_tensor)
            _mc['batch_inputs'].requires_grad_(True)
            _mc['targets_base'] = tensor(_base, **p_tensor)
            _mc['targets_loss'] = tensor(_loss[batch_size:], **p_tensor)
            _mc['targets_pred'] = tensor(_pred[batch_size:], **p_tensor)
            if self.verbosity > 1:
                _mn = mouse.name
                print(_mn, 'batch_inputs shape:', _mc['batch_inputs'].shape)
                print(_mn, 'targets_base shape:', _mc['targets_base'].shape)
                print(_mn, 'targets_loss shape:', _mc['targets_loss'].shape)
                print(_mn, 'targets_pred shape:', _mc['targets_pred'].shape)
        target_accuracy = 97.0
        epochs = 0
        final_accuracy = 0
        mouse_accuracy = 0
        Awen.reset_metrics()
        wicks, t_base, t_loss, t_pred = [[], [], [], []]
        while final_accuracy < target_accuracy:
            if epochs == timeout:
                break
            else:
                epochs += 1
            final_accuracy = 0
            mouse_accuracy = 0
            Awen.reset_candles()
            for mouse in moirai:
                mouse.reset_candles()
                mouse.reset_metrics()
            for i in batch_range:
                indices = (i, i + batch_size)
                if indices[1] <= batch_range[-1]:
                    for mouse in moirai:
                        time_step(mouse, indices, study=True)
                    Awen.optimizer.zero_grad()
                    wicks, t_base, t_loss, t_pred = [[], [], [], []]
                    ii, iii = indices
                    for mouse in moirai:
                        wicks.append(mouse.candles['coated'])
                        t_base.append(mouse.candles['targets_base'][ii:iii])
                        t_loss.append(mouse.candles['targets_loss'][ii:iii])
                        t_pred.append(mouse.candles['targets_pred'][ii:iii])
                    wicks = hstack(wicks)
                    t_base = vstack(t_base).H
                    t_loss = vstack(t_loss).H
                    t_pred = vstack(t_pred).H
                    coated_wicks = Awen.give(wicks)
                    loss = Awen.loss_fn(coated_wicks, t_loss)
                    loss.backward()
                    Awen.optimizer.step()
                    Awen.scheduler.step()
                    threshold = float(t_pred.max().item() * tolerance)
                    coated_candles = (t_base + (t_base * coated_wicks))
                    Awen.candles['coated'] = coated_candles.clone()
                    Awen.candles['sealed'].append(coated_candles[0].clone())
                    difference = (coated_candles - t_pred)
                    correct = difference[difference >= -threshold]
                    correct = correct[correct <= threshold].shape[0]
                    cauldron_size = difference.shape[0] * difference.shape[1]
                    Awen.metrics['acc'][0] += int(correct)
                    Awen.metrics['acc'][1] += int(cauldron_size)
                    Awen.metrics['loss'] += loss.item()
                    Awen.metrics['mae'] += difference.abs().sum().item()
                    Awen.metrics['mse'] += (difference ** 2).sum().item()
                    if self.verbosity > 2:
                        for m_k, m_v in Awen.metrics.items():
                            print(f'Awen {m_k}: {m_v}')
                else:
                    for mouse in moirai:
                        time_step(mouse, (-batch_size, None), epochs=timestamps)
                    wicks = [m.candles['coated'] for m in moirai]
                    t_key = 'targets_base'
                    t_base = [m.candles[t_key][-batch_size:] for m in moirai]
                    t_base = vstack(t_base).H
                    coated_wicks = Awen.give(wicks)
                    coated_candles = t_base + (t_base * coated_wicks)
                    Awen.candles['coated'] = coated_candles.clone()
                    seal = vstack(Awen.candles['sealed'])
                    Awen.candles['sealed'] = vstack([seal, coated_candles])
                    Awen.metrics['loss'] = Awen.metrics['loss'] / epochs
                    Awen.metrics['mae'] = Awen.metrics['mae'] / epochs
                    Awen.metrics['mse'] = Awen.metrics['mse'] / epochs
                    n_correct = Awen.metrics['acc'][0]
                    n_total = Awen.metrics['acc'][1]
                    n_wrong = n_total - n_correct
                    final_accuracy = 100 * abs((n_wrong - n_total) / n_total)
                    final_accuracy = round(final_accuracy, 3)
                    n_correct = 0
                    n_total = 0
                    for mouse in moirai:
                        n_correct += mouse.metrics['acc'][0]
                        n_total += mouse.metrics['acc'][1]
                    n_wrong = n_total - n_correct
                    mouse_accuracy = 100 * abs((n_wrong - n_total) / n_total)
                    mouse_accuracy = round(mouse_accuracy, 3)
                    break
            if self.verbosity > 0:
                print(f'{prefix} ({epochs}) A moment of research yielded;')
                print(f'    a cauldron accuracy of {final_accuracy}%')
                print(f'    a mouse accuracy of {mouse_accuracy}%')
                if self.verbosity > 1:
                    print('coated_candles:\n', coated_candles)
                    print('metrics:')
                    for metric_k, metric_v in Awen.metrics.items():
                        print(f'    {metric_k}:', metric_v)
                    print('')
        coated = Awen.candles['coated'].detach().cpu().numpy()
        sealed = Awen.candles['sealed'].detach().cpu().numpy()
        print(f'sealed: {sealed.shape}\n', sealed)
        last_pred = float(Awen.candles['coated'][-1][0].item())
        last_price = float(candles['close'][-1])
        proj_gain = float(((last_pred - last_price) / last_price) * 100)
        self.predictions[symbol] = {
            'cauldron loss': float(Awen.metrics['loss']),
            'cauldron_accuracy': final_accuracy,
            'mouse_accuracy': mouse_accuracy,
            'coated_candles': coated,
            'sealed_candles': sealed,
            'last_price': round(last_price, 2),
            'batch_pred': round(last_pred, 2),
            'num_epochs': epochs,
            'metrics': dict(Awen.metrics),
            'proj_gain': proj_gain,
            'proj_time': time.time()
            }
        if self.verbosity > 0:
            epoch_str = 'epoch' if epochs == 1 else 'epochs'
            msg = 'After {} {}, an accuracy of {}% was realized.'
            print(prefix, msg.format(epochs, epoch_str, final_accuracy))
            print(f'{prefix} {symbol} Metrics;')
            for k, v in self.predictions[symbol].items():
                if k in ['cauldron_accuracy', 'mouse_accuracy']:
                    print(f'{prefix}     {k}: {v}%')
                elif k in ['sealed_candles', 'coated_candles']:
                    if self.verbosity > 1:
                        print(f'{prefix}     {k}: {v}')
                        print(f'{prefix}     {k} shape: {v.shape}')
                else:
                    print(f'{prefix}     {k}: {v}')
            print('')
        Awen.release_tensors()
        for mouse in moirai:
            mouse.release_tensors()
        self.__manage_state__(call_type=1)
        return True
