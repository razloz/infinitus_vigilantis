"""Three blind mice to predict the future."""
import time
import torch
import torch.nn as nn
import torch.optim as optim
import traceback
import source.ivy_commons as icy
from math import sqrt, log
from os.path import abspath
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class MouseGate(nn.Sequential):
    """Inside the mind of a mouse."""
    def __init__(self, n_features=0, n_hidden=0, batch_size=0, n_gates=0):
        """Register gate modules in sequence."""
        super(MouseGate, self).__init__()
        gru_params = dict(
            num_layers=batch_size,
            bias=True,
            batch_first=True,
            dropout=0.34,
            )
        input_size = n_features
        output_size = n_hidden
        for gate_num in range(n_gates):
            self.register_module(
                f'GRU_{gate_num}',
                nn.GRU(input_size, output_size, **gru_params)
                )
            self.register_module(f'GLU_{gate_num}', nn.GLU())
            output_size = int(output_size / 2)
            input_size = int(output_size)

    def forward(self, *inputs):
        """Walk through the gates until the desired candle shape is reached."""
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(inputs[0])
            else:
                inputs = module(inputs)
        return torch.vstack(inputs.max(dim=1).values.split(1))


class Cauldron(nn.Module):
    """A wax encrusted cauldron sits before you, bubbling occasionally."""
    def __init__(self, cauldron_size, batch_size, device, *args, **kwargs):
        """Assists in the creation of candles."""
        super(Cauldron, self).__init__(*args, **kwargs)
        self._device_ = device
        self._n_hidden_ = int(cauldron_size ** 2)
        self._n_layers_ = int(batch_size)
        self._n_size_ = 3
        self._name_ = 'Awen'
        self._wax_ = nn.GRU(
            input_size=self._n_size_,
            hidden_size=self._n_hidden_,
            num_layers=self._n_layers_,
            bias=True,
            dropout=0.34,
            device=self._device_,
            )
        self.__batch_fn__ = nn.BatchNorm1d(self._n_hidden_)
        self.__pool_fn__ = nn.AdaptiveMaxPool1d(self._n_size_)
        self.to(self._device_)

    def forward(self, wicks):
        """*bubble**bubble* *bubble*"""
        wicks = torch.hstack([*wicks])
        wicks = self._wax_(wicks)[0]
        wicks = self.__batch_fn__(wicks)
        wicks = self.__pool_fn__(wicks).H
        wicks = [
            torch.vstack([t for t in wicks[0].split(1)]).clone(),
            torch.vstack([t for t in wicks[1].split(1)]).clone(),
            torch.vstack([t for t in wicks[2].split(1)]).clone(),
            ]
        return wicks


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, n_features, verbosity=0, min_size=90, *args, **kwargs):
        """Inputs: n_features and n_targets must be of type int()"""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._min_size_ = int(min_size)
        self._prefix_ = 'Moirai:'
        self._state_path_ = abspath('./rnn/moirai.state')
        self._tolerance_ = 1.618033988749894e-3
        self.verbosity = int(verbosity)
        batch_size = self._batch_size_ = 8
        n_features = self._n_features_ = int(n_features)
        n_gates = 7
        n_hidden = self._batch_size_
        for i in range(n_gates):
            n_hidden *= 2
        gate_args = dict(
            n_features=int(n_features),
            n_hidden=int(n_hidden),
            batch_size=int(batch_size),
            n_gates=int(n_gates),
            )
        if self.verbosity > 0:
            print(self._prefix_, 'set batch_size to', batch_size)
            print(self._prefix_, 'set n_features to', n_features)
            print(self._prefix_, 'set n_hidden to', n_hidden)
            print(self._prefix_, 'set n_gates to', n_gates)
        self._tensor_args_ = dict(
            device=self._device_,
            dtype=torch.float,
            )
        rprop_params = dict(
            lr=0.01,
            etas=(0.5, 1.2),
            step_sizes=(1e-06, 50),
            foreach=True,
            )
        sgd_params = dict(
            lr=0.1,
            momentum=0.9,
            dampening=0,
            weight_decay=0,
            nesterov=True,
            maximize=False,
            foreach=True
            )
        # Setup a cauldron for our candles
        cauldron = self._cauldron_ = nn.ParameterDict()
        cauldron['coated_candles'] = None
        cauldron['sealed_candles'] = list()
        cauldron['size'] = int(batch_size * 3)
        cauldron['wax'] = Cauldron(cauldron['size'], batch_size, self._device_)
        cauldron['loss_fn'] = nn.HuberLoss(reduction='sum', delta=0.97)
        cauldron['targets'] = torch.zeros(batch_size, 1, **self._tensor_args_)
        cauldron['metrics'] = {}
        cauldron['optim'] = optim.Rprop(
            cauldron['wax'].parameters(),
            **rprop_params
            )
        cauldron['warm_lr'] = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            cauldron['optim'],
            cauldron['size'],
            )
        # Atropos the mouse, sister of Clotho and Lachesis.
        Atropos = self._Atropos_ = nn.ParameterDict()
        Atropos['name'] = 'Atropos'
        # Clotho the mouse, sister of Lachesis and Atropos.
        Clotho = self._Clotho_ = nn.ParameterDict()
        Clotho['name'] = 'Clotho'
        # Lachesis the mouse, sister of Atropos and Clotho.
        Lachesis = self._Lachesis_ = nn.ParameterDict()
        Lachesis['name'] = 'Lachesis'
        for mouse in [Atropos, Clotho, Lachesis]:
            mouse['candles'] = None
            mouse['loss_fn'] = nn.HuberLoss(reduction='sum')
            mouse['metrics'] = {}
            mouse['nn_gates'] = MouseGate(**gate_args)
            mouse['optim'] = optim.SGD(
                mouse['nn_gates'].parameters(),
                **sgd_params
                )
        self.predictions = nn.ParameterDict()
        self.to(self._device_)
        if self.verbosity > 0:
            print(self._prefix_, f'Set device to {self._device_type_.upper()}.')

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

    def __time_step__(self, mouse, cheese, epochs=0, study=False):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles"""
        vstack = torch.vstack
        if study is True:
            threshold = self._tolerance_
            fresh = vstack(cheese[0])
            aged = vstack(cheese[1])
            batch_size = fresh.shape[0]
            mouse['optim'].zero_grad()
            mouse['candles'] = mouse['nn_gates'](fresh)
            loss = mouse['loss_fn'](mouse['candles'], aged)
            loss.backward()
            mouse['metrics']['loss'] = int(loss.item())
            mouse['optim'].step()
            difference = (mouse['candles'] - aged)
            correct = difference[difference >= -threshold]
            correct = correct[correct <= threshold].shape[0]
            mouse['metrics']['acc'][0] += int(correct)
            mouse['metrics']['acc'][1] += int(batch_size)
            mouse['metrics']['mae'] += difference.abs().sum()
            mouse['metrics']['mse'] += (difference ** 2).sum()
        else:
            with torch.no_grad():
                try:
                    n_mae = mouse['metrics']['mae']
                    n_mse = mouse['metrics']['mse']
                    mouse['metrics']['mae'] = n_mae / epochs
                    mouse['metrics']['mse'] = sqrt(n_mse / epochs)
                except Exception as details:
                    if self.verbosity > 0:
                        msg = '{} Encountered an exception.'
                        print(self._prefix_, msg.format(mouse['name']))
                        traceback.print_exception(details)
                finally:
                    mouse['candles'] = mouse['nn_gates'](cheese)
        verbosity_check = [
            self.verbosity > 2,
            self.verbosity == 2 and study is False
            ]
        if any(verbosity_check):
            lr = self._cauldron_['warm_lr'].get_last_lr()[0]
            acc = mouse['metrics']['acc']
            mae = mouse['metrics']['mae']
            mse = mouse['metrics']['mse']
            msg = f'{self._prefix_} {mouse["name"]} ' + '{}: {}'
            print(msg.format('Learning Rate', lr))
            print(msg.format('Accuracy', acc))
            print(msg.format('Mean Absolute Error', mae))
            print(msg.format('Mean Squared Error', mse))

    def research(self, symbol, candles, timeout=8):
        """Moirai research session, fully stocked with cheese and drinks."""
        if not all((
            len(candles.keys()) == self._n_features_,
            len(candles.index) >= self._min_size_
            )): return False
        self.__manage_state__(call_type=0)
        Atropos = self._Atropos_
        Clotho = self._Clotho_
        Lachesis = self._Lachesis_
        time_step = self.__time_step__
        batch = self._batch_size_
        prefix = self._prefix_
        t_args = self._tensor_args_
        tolerance = self._tolerance_
        hstack = torch.hstack
        vstack = torch.vstack
        tensor = torch.tensor
        candles = candles[self._min_size_:]
        a_targets = candles['chg_close'].to_numpy()
        a_values = candles['close'].to_numpy()
        c_targets = candles['chg_open'].to_numpy()
        c_values = candles['open'].to_numpy()
        l_targets = candles['chg_wema'].to_numpy()
        l_values = candles['price_wema'].to_numpy()
        timestamps = len(candles.index)
        candles_index = range(timestamps - batch)
        candles = candles.to_numpy()
        moirai = [Atropos, Clotho, Lachesis]
        for mouse in moirai:
            if mouse['name'] == 'Atropos':
                m_values = a_values
                m_targets = a_targets
            elif mouse['name'] == 'Clotho':
                m_values = c_values
                m_targets = c_targets
            elif mouse['name'] == 'Lachesis':
                m_values = l_values
                m_targets = l_targets
            mouse['aged_cheese'] = tensor(m_targets[batch:], **t_args)
            mouse['raw_cheese'] = tensor(m_values[:-batch], **t_args)
            mouse['raw_final'] = tensor(m_values[-batch:], **t_args)
            mouse['final_cheese'] = tensor(candles[-batch:], **t_args)
            mouse['fresh_cheese'] = tensor(candles[:-batch], **t_args)
            mouse['fresh_cheese'].requires_grad_(True)
        target_accuracy = 90.0
        epochs = 0
        final_accuracy = 0
        mouse_accuracy = 0
        cauldron = self._cauldron_
        cauldron['metrics']['acc'] = [0, 0]
        cauldron['metrics']['loss'] = 0
        cauldron['metrics']['mae'] = 0
        cauldron['metrics']['mse'] = 0
        while final_accuracy < target_accuracy:
            if epochs == timeout: break
            fresh = [[],[],[]]
            aged = [[],[],[]]
            raw = [[],[],[]]
            batch_count = 0
            final_accuracy = 0
            mouse_accuracy = 0
            cauldron['coated_candles'] = None
            cauldron['sealed_candles'] = list()
            for mouse in moirai:
                mouse['metrics']['acc'] = [0, 0]
                mouse['metrics']['loss'] = 0
                mouse['metrics']['mae'] = 0
                mouse['metrics']['mse'] = 0
            for i in candles_index:
                if batch_count == batch:
                    for m_i, mouse in enumerate(moirai):
                        cheese = (fresh[m_i], aged[m_i])
                        time_step(mouse, cheese, study=True)
                    cauldron['optim'].zero_grad()
                    wicks = [m['candles'] for m in moirai]
                    raw_cheese = hstack([vstack(raw[bi]) for bi in range(3)])
                    aged_cheese = hstack([vstack(aged[bi]) for bi in range(3)])
                    coated_wicks = hstack(cauldron['wax'](wicks))
                    loss = cauldron['loss_fn'](coated_wicks, aged_cheese)
                    loss.backward()
                    cauldron['optim'].step()
                    cauldron['warm_lr'].step()
                    coated = raw_cheese + (raw_cheese * coated_wicks)
                    targets = raw_cheese + (raw_cheese * aged_cheese)
                    difference = (coated - targets)
                    threshold = float(raw_cheese.max().item() * tolerance)
                    correct = difference[difference >= -threshold]
                    correct = correct[correct <= threshold].shape[0]
                    cauldron_size = difference.shape[0] * difference.shape[1]
                    cauldron['metrics']['acc'][0] += int(correct)
                    cauldron['metrics']['acc'][1] += int(cauldron_size)
                    cauldron['metrics']['loss'] += loss.item()
                    cauldron['metrics']['mae'] += difference.abs().sum()
                    cauldron['metrics']['mse'] += (difference ** 2).sum()
                    cauldron['coated_candles'] = coated.clone()
                    cauldron['sealed_candles'].append(coated.clone())
                    if self.verbosity > 2:
                        print('raw_cheese\n', raw_cheese)
                        print('aged_cheese\n', aged_cheese)
                        print('coated_wicks\n', coated_wicks)
                        print('coated\n', coated)
                        print('targets\n', targets)
                        print('threshold', threshold)
                        print('correct\n', correct)
                        print('batch total\n', targets.shape[0])
                    fresh = [[],[],[]]
                    aged = [[],[],[]]
                    raw = [[],[],[]]
                    batch_count = 0
                else:
                    for m_i, mouse in enumerate(moirai):
                        fresh[m_i].append(mouse['fresh_cheese'][i].clone())
                        aged[m_i].append(mouse['aged_cheese'][i].clone())
                        raw[m_i].append(mouse['raw_cheese'][i].clone())
                    batch_count += 1
            epochs += 1
            raw_final = list()
            for m_i, mouse in enumerate(moirai):
                time_step(mouse, mouse['final_cheese'], epochs=timestamps)
                raw_final.append(mouse['raw_final'])
            wicks = [m['candles'] for m in moirai]
            coated_candles = hstack(cauldron['wax'](wicks))
            raw_final = vstack([*raw_final]).H
            coated_candles = raw_final + (raw_final * coated_candles)
            cauldron['coated_candles'] = coated_candles.clone()
            cauldron['sealed_candles'].append(coated_candles.clone())
            cauldron['metrics']['loss'] = cauldron['metrics']['loss'] / epochs
            cauldron['metrics']['mae'] = cauldron['metrics']['mae'] / epochs
            cauldron['metrics']['mae'] = cauldron['metrics']['mae'].item()
            cauldron['metrics']['mse'] = cauldron['metrics']['mse'] / epochs
            cauldron['metrics']['mse'] = cauldron['metrics']['mse'].item()
            n_correct = cauldron['metrics']['acc'][0]
            n_total = cauldron['metrics']['acc'][1]
            n_wrong = n_total - n_correct
            final_accuracy = 100 * abs((n_wrong - n_total) / n_total)
            final_accuracy = round(final_accuracy, 3)
            n_correct = 0
            n_total = 0
            for mouse in moirai:
                n_correct += mouse['metrics']['acc'][0]
                n_total += mouse['metrics']['acc'][1]
            n_wrong = n_total - n_correct
            mouse_accuracy = 100 * abs((n_wrong - n_total) / n_total)
            mouse_accuracy = round(mouse_accuracy, 3)
            if self.verbosity > 0:
                print(f'{prefix} ({epochs}) A moment of research yielded;')
                print(f'    a cauldron accuracy of {final_accuracy}%')
                print(f'    a mouse accuracy of {mouse_accuracy}%')
                if self.verbosity > 1:
                    print('coated_candles:\n', coated_candles)
                    print('raw_final:\n', raw_final)
                    print('metrics:')
                    for metric_k, metric_v in cauldron['metrics'].items():
                        print(f'    {metric_k}:', metric_v)
                    print('')
        coated = cauldron['coated_candles'].detach().cpu().numpy()
        sealed = vstack(cauldron['sealed_candles']).detach().cpu().numpy()
        last_pred = float(cauldron['coated_candles'][-1][0].item())
        last_price = float(Atropos['raw_final'][-1].item())
        proj_gain = float(((last_pred - last_price) / last_price) * 100)
        self.predictions[symbol] = {
            'cauldron loss': float(cauldron['metrics']['loss']),
            'cauldron_accuracy': final_accuracy,
            'mouse_accuracy': mouse_accuracy,
            'coated_candles': coated,
            'sealed_candles': sealed,
            'last_price': round(last_price, 2),
            'batch_pred': round(last_pred, 2),
            'num_epochs': epochs,
            'metrics': dict(cauldron['metrics']),
            'proj_gain': proj_gain,
            'proj_time': time.time()
            }
        if self.verbosity > 0:
            epoch_str = 'epoch' if epochs == 1 else 'epochs'
            msg = 'After {} {}, an accuracy of {}% was realized.'
            print(prefix, msg.format(epochs, epoch_str, final_accuracy))
            print(f'{prefix} {symbol} Metrics;')
            for k, v in self.predictions[symbol].items():
                if k in ['cauldron_accuracy', 'final_accuracy']:
                    print(f'{prefix}     {k}: {v}%')
                elif k in ['sealed_candles', 'coated_candles']:
                    if self.verbosity > 1:
                        print(f'{prefix}     {k}: {v}')
                        print(f'{prefix}     {k} shape: {v.shape}')
                else:
                    print(f'{prefix}     {k}: {v}')
            print('')
        self.__manage_state__(call_type=1)
        return True
