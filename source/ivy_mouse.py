"""Three blind mice to predict the future."""
import time
import torch
import traceback
import source.ivy_commons as icy
from math import sqrt
from os.path import abspath
from torch.nn import GLU, GRU, Module, ParameterDict, SmoothL1Loss, Softplus
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts as WarmRestarts
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class GatedSequence(torch.nn.Sequential):
    """Subclass of torch.nn.Sequential"""
    def forward(self, *inputs):
        """Link GRU and GLU inputs to outputs."""
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(inputs[0])
            else:
                inputs = module(inputs)
        return inputs


class ThreeBlindMice(Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, n_features, verbosity=0, batch_size=8, min_size=90):
        """Inputs: n_features and n_targets must be of type int()"""
        super(ThreeBlindMice, self).__init__()
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._min_size_ = min_size
        self._prefix_ = 'Moirai:'
        self._state_path_ = abspath('./rnn/moirai.state')
        self._batch_size_ = int(batch_size)
        self._n_features_ = int(n_features)
        self._tensor_args_ = dict(
            device=self._device_,
            dtype=torch.float,
            requires_grad=True
            )
        gru_params = dict(
            input_size=self._n_features_,
            hidden_size=int(batch_size ** 2),
            num_layers=self._n_features_,
            bias=True,
            batch_first=True,
            dropout=0.003,
            bidirectional=True,
            )
        opt_params = dict(
            lr=0.9,
            momentum=0.9,
            nesterov=True,
            maximize=True,
            )
        # Atropos the mouse, sister of Clotho and Lachesis.
        Atropos = self._Atropos_ = ParameterDict()
        Atropos['name'] = 'Atropos'
        # Clotho the mouse, sister of Lachesis and Atropos.
        Clotho = self._Clotho_ = ParameterDict()
        Clotho['name'] = 'Clotho'
        # Lachesis the mouse, sister of Atropos and Clotho.
        Lachesis = self._Lachesis_ = ParameterDict()
        Lachesis['name'] = 'Lachesis'
        for mouse in [Atropos, Clotho, Lachesis]:
            mouse['candles'] = None
            mouse['loss_fn'] = SmoothL1Loss()
            mouse['metrics'] = {}
            mouse['nn_gates'] = GatedSequence(
                GRU(**gru_params), GLU(),
                GLU(), GLU(), GLU(),
                )
            mouse['optim'] = SGD(mouse['nn_gates'].parameters(), **opt_params)
            mouse['warm_lr'] = WarmRestarts(mouse['optim'], 55)
        self.predictions = ParameterDict()
        self.to(self._device_)
        self.verbosity = verbosity
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

    def __time_step__(self, mouse, inputs, targets, epochs=0, study=False):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles"""
        vstack = torch.vstack
        if study is True:
            inputs = vstack(inputs)
            targets = vstack(targets)
            percent_change = icy.percent_change
            batch_size = inputs.shape[0]
            mouse['optim'].zero_grad()
            mouse['candles'] = mouse['nn_gates'](inputs)[0]
            mouse['candles'] = vstack(mouse['candles'].split(1))
            difference = mouse['candles'] - targets
            loss_target = torch.ones(batch_size, 1, **self._tensor_args_)
            loss = mouse['loss_fn'](mouse['candles'], targets)
            loss.backward()
            mouse['optim'].step()
            mouse['warm_lr'].step()
            correct = difference[difference == 1].shape[0]
            wrong = batch_size - correct
            prob = loss.item() / batch_size if loss.item() != 0 else 0
            mouse['metrics']['prob'] = prob
            mouse['metrics']['acc'] = 100 - percent_change(correct, wrong) * -1
            mouse['metrics']['mae'] += torch.abs(difference).sum().data
            mouse['metrics']['mse'] += (difference ** 2).sum().data
        else:
            with torch.no_grad():
                try:
                    for key in ['acc', 'mae', 'mse']:
                        batch_average = mouse['metrics'][key] / epochs
                        if key == 'mse':
                            batch_average = sqrt(batch_average)
                        mouse['metrics'][key] = batch_average
                except Exception as details:
                    if self.verbosity > 0:
                        msg = '{} Encountered an exception.'
                        print(self._prefix_, msg.format(mouse['name']))
                        traceback.print_exception(details)
                finally:
                    mouse['candles'] = mouse['nn_gates'](inputs)[0]
                    mouse['candles'] = vstack(mouse['candles'].split(1))
        verbosity_check = [
            self.verbosity > 2,
            self.verbosity == 2 and study is False
            ]
        if any(verbosity_check):
            lr = mouse['warm_lr'].get_last_lr()[0]
            prob = mouse['metrics']['prob']
            acc = mouse['metrics']['acc']
            mae = mouse['metrics']['mae']
            mse = mouse['metrics']['mse']
            msg = f'{self._prefix_} {mouse["name"]} ' + '{}: {}'
            print(msg.format('Learning Rate', lr))
            print(msg.format('Probability', prob))
            print(msg.format('Accuracy', acc))
            print(msg.format('Mean Absolute Error', mae))
            print(msg.format('Mean Squared Error', mse))
            print(msg.format('Target', mouse['candles'][-1]))

    def research(self, symbol, candles, timeout=3):
        """Moirai research session, fully stocked with cheese and drinks."""
        if not all((
            len(candles.keys()) == self._n_features_,
            len(candles.index) >= self._min_size_
            )): return False
        self.__manage_state__(call_type=0)
        Clotho = self._Clotho_
        Lachesis = self._Lachesis_
        Atropos = self._Atropos_
        time_step = self.__time_step__
        batch = self._batch_size_
        prefix = self._prefix_
        t_args = self._tensor_args_
        percent_change = icy.percent_change
        tensor = torch.tensor
        c_targets = candles['open'].to_numpy()
        l_targets = candles['volume'].to_numpy()
        a_targets = candles['close'].to_numpy()
        timestamps = len(candles.index)
        candles_index = range(timestamps - batch)
        candles = candles.to_numpy()
        t_args['requires_grad'] = True
        Clotho['data_inputs'] = tensor(candles[:-batch], **t_args)
        Lachesis['data_inputs'] = tensor(candles[:-batch], **t_args)
        Atropos['data_inputs'] = tensor(candles[:-batch], **t_args)
        t_args['requires_grad'] = False
        Clotho['data_final'] = tensor(candles[-batch:], **t_args)
        Lachesis['data_final'] = tensor(candles[-batch:], **t_args)
        Atropos['data_final'] = tensor(candles[-batch:], **t_args)
        Clotho['data_targets'] = tensor(c_targets[batch:], **t_args)
        Lachesis['data_targets'] = tensor(l_targets[batch:], **t_args)
        Atropos['data_targets'] = tensor(a_targets[batch:], **t_args)
        moirai = [Atropos, Clotho, Lachesis]
        target_accuracy = 97.0
        epochs = 0
        final_accuracy = 0
        volume_accuracy = 0
        while final_accuracy < target_accuracy:
            if epochs == timeout: break
            inputs = [[],[],[]]
            targets = [[],[],[]]
            batch_count = 0
            final_accuracy = 0
            volume_accuracy = 0
            for mouse in moirai:
                mouse['metrics']['acc'] = 0
                mouse['metrics']['mae'] = 0
                mouse['metrics']['mse'] = 0
            for i in candles_index:
                if batch_count == batch:
                    for m_i, mouse in enumerate(moirai):
                        time_step(mouse, inputs[m_i], targets[m_i], study=True)
                    if self.verbosity > 2:
                        print('\n')
                    inputs = [[],[],[]]
                    targets = [[],[],[]]
                    batch_count = 0
                else:
                    for m_i, mouse in enumerate(moirai):
                        inputs[m_i].append(mouse['data_inputs'][i].clone())
                        targets[m_i].append(mouse['data_targets'][i].clone())
                    batch_count += 1
            for mouse in moirai:
                time_step(
                    mouse,
                    mouse['data_final'].clone(),
                    None,
                    epochs=timestamps
                    )
                name = mouse['name']
                accuracy = mouse['metrics']['acc']
                if name == 'Lachesis':
                    volume_accuracy += accuracy
                else:
                    final_accuracy += accuracy
                if self.verbosity > 1:
                    print(prefix, 'Target:', mouse['data_targets'][-1], '\n')
            if epochs != 0:
                final_accuracy = float(final_accuracy)
                if final_accuracy != 0:
                    final_accuracy = final_accuracy / epochs
                volume_accuracy = float(volume_accuracy)
                if volume_accuracy != 0 != epochs:
                    volume_accuracy = volume_accuracy / epochs
            epochs += 1
            if self.verbosity > 0:
                msg = f'{prefix} ({epochs}) A moment of research '
                msg += 'yielded a price / volume accuracy of {}% / {}%'
                if self.verbosity > 1:
                    print('\n')
                print(msg.format(final_accuracy, volume_accuracy))
        sealed = [mouse['candles'].detach().cpu().numpy() for mouse in moirai]
        last_pred = float(sealed[0][-1].item())
        last_price = float(Atropos['data_final'][-1:, 3:4].item())
        proj_gain = percent_change(last_pred, last_price)
        self.predictions[symbol] = {
            'final_accuracy': final_accuracy,
            'volume_accuracy': volume_accuracy,
            'sealed_candles': sealed,
            'last_price': last_price,
            'batch_pred': sealed[0][-1].item(),
            'num_epochs': epochs,
            'metrics': dict(mouse['metrics']),
            'proj_gain': proj_gain,
            'proj_time': time.time()
            }
        if self.verbosity > 0:
            epoch_str = 'epoch' if epochs == 1 else 'epochs'
            msg = 'After {} {}, an accuracy of {}% was realized.'
            print(prefix, msg.format(epochs, epoch_str, final_accuracy))
            print(f'{prefix} {symbol} Metrics;')
            for k, v in self.predictions[symbol].items():
                if k == 'sealed_candles':
                    if self.verbosity > 1:
                        print(f'{prefix}     {k}: {v}')
                elif k == 'metrics':
                    print(f'{prefix}     {k}:')
                    for m_k, m_v in v.items():
                        print(f'{prefix}         {m_k}: {m_v}')
                else:
                    print(f'{prefix}     {k}: {v}')
            print('\n')
        data_labels = ['data_inputs', 'data_targets', 'data_final']
        for label in data_labels:
            Clotho[label] = None
            Lachesis[label] = None
            Atropos[label] = None
        self.__manage_state__(call_type=1)
        return True

