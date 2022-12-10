"""Three blind mice to predict the future."""
import time
import torch
import traceback
import matplotlib.pyplot as plt
import numpy as np
import source.ivy_commons as icy
from torch import nn
from torch.optim import Adagrad
from torch.utils.data import DataLoader, TensorDataset
from math import sqrt, inf
from os import mkdir
from os.path import abspath, exists
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, *args, cook_time=0, features=0, verbosity=0, **kwargs):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        iota = 1 / 137
        phi = 1.618033988749894
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self._epochs_path_ = abspath('./rnn/epochs')
        if not exists(self._epochs_path_): mkdir(self._epochs_path_)
        self._constants_ = constants = {
            'batch_size': 34,
            'batch_step': 17,
            'cluster_shape': 3,
            'cook_time': cook_time,
            'dim': 5,
            'dropout': iota,
            'eps': iota * 1e-6,
            'features': int(features),
            'hidden': 5 ** 3,
            'layers': 128,
            'lr_decay': iota / (phi - 1),
            'lr_init': iota ** (phi - 1),
            'momentum': phi * (phi - 1),
            'weight_decay': iota / phi,
            }
        self._prefix_ = prefix = 'Moirai:'
        self._p_tensor_ = dict(device=self._device_, dtype=torch.float)
        self._symbol_ = None
        self.cauldron = nn.GRU(
            input_size=constants['features'],
            hidden_size=constants['hidden'],
            num_layers=constants['layers'],
            dropout=constants['dropout'],
            bias=True,
            batch_first=True,
            device=self._device_,
            )
        self.loss_fn = torch.nn.BCELoss(
            reduction='sum',
            )
        self.melt = nn.InstanceNorm1d(
            constants['features'],
            momentum=constants['momentum'],
            eps=constants['eps'],
            **self._p_tensor_,
            )
        self.optimizer = Adagrad(
            self.parameters(),
            lr=constants['lr_init'],
            lr_decay=constants['lr_decay'],
            weight_decay=constants['weight_decay'],
            eps=constants['eps'],
            foreach=True,
            maximize=False,
            )
        self.stir = nn.Conv3d(
            in_channels=constants['layers'],
            out_channels=constants['layers'],
            kernel_size=constants['cluster_shape'],
            **self._p_tensor_,
            )
        self.metrics = nn.ParameterDict({
            'acc': [0, 0],
            'epochs': 0,
            'loss': 0,
            })
        self.predictions = nn.ParameterDict()
        self.verbosity = int(verbosity)
        self.to(self._device_)
        self.__manage_state__(call_type=0)
        if self.verbosity > 1:
            for key, value in constants.items():
                print(prefix, f'set {key} to {value}')
            print('')

    def __manage_state__(self, call_type=0, singular=True):
        """Handles loading and saving of the RNN state."""
        state_path = f'{self._state_path_}/'
        if singular:
            state_path += '.norn.state'
        else:
            state_path += f'.{self._symbol_}.state'
        if call_type == 0:
            try:
                state = torch.load(state_path, map_location=self._device_type_)
                self.load_state_dict(state['moirai'])
                for key, value in state['metrics'].items():
                    self.metrics[key] = value
                for key, value in state['predictions'].items():
                    self.predictions[key] = value
                if self.verbosity > 2:
                    print(self._prefix_, 'Loaded RNN state.')
            except FileNotFoundError:
                if not singular:
                    self.metrics['epochs'] = 0
                    self.predictions = nn.ParameterDict()
                self.__manage_state__(call_type=1)
            except Exception as details:
                if self.verbosity > 1:
                    print(self._prefix_, *details.args)
        elif call_type == 1:
            torch.save(
                {
                    'metrics': self.metrics,
                    'moirai': self.state_dict(),
                    'predictions': self.predictions,
                    },
                state_path,
                )
            if self.verbosity > 2:
                print(self._prefix_, 'Saved RNN state.')

    def __time_plot__(self, predictions, targets):
        fig = plt.figure(figsize=(5.12, 3.84), dpi=100)
        ax = fig.add_subplot()
        ax.grid(True, color=(0.6, 0.6, 0.6))
        ax.set_ylabel('Probability', fontweight='bold')
        ax.set_xlabel('Batch', fontweight='bold')
        accuracy = self.metrics['acc']
        epoch = self.metrics['epochs']
        y_p = predictions.detach().cpu().numpy()
        y_t = targets.detach().cpu().numpy()
        y_t = np.vstack([y_t, [None, None, None]])
        x_range = range(0, y_p.shape[0] * 5, 5)
        n_y = 0
        for r_x in x_range:
            n_x = 0
            for prob in y_t[n_y]:
                if prob is None:
                    continue
                x = r_x + n_x
                ax.plot([x, x], [0, prob], color='#FF9600') #cheddar
                n_x += 1
            n_x = 0
            for prob in y_p[n_y]:
                x = r_x + n_x
                ax.plot([x, x], [0, prob], color='#FFE88E') #gouda
                n_x += 1
            n_y += 1
        symbol = self._symbol_
        title = f'{symbol}\n'
        title += f'Accuracy: {accuracy}, '
        title += f'Epochs: {epoch}, '
        fig.suptitle(title, fontsize=13)
        plt.savefig(f'{self._epochs_path_}/{int(time.time())}.png')
        plt.clf()
        plt.close()

    def __time_step__(self, candles, study=False):
        """Let Clotho mold the candles
           Let Lachesis measure the candles
           Let Atropos seal the candles
           Let Awen contain the wax"""
        if study:
            self.train()
            self.optimizer.zero_grad()
            return self.forward(candles)
        else:
            self.eval()
            with torch.no_grad():
                return self.forward(candles)

    def forward(self, candles):
        """**bubble*bubble**bubble**"""
        constants = self._constants_
        batch = constants['batch_step']
        dim = constants['dim']
        inscribe = torch.topk
        candles = self.melt(candles)
        bubbles = self.cauldron(candles)[1]
        bubbles = bubbles.view(bubbles.shape[0], dim, dim, dim)
        bubbles = self.stir(bubbles)
        candles = inscribe(bubbles, 1)[0].flatten(0)
        candles = inscribe(candles, 3)[0].softmax(0)
        return candles.clone()

    def research(self, offering, dataframe, plot=True, keep_predictions=True):
        """Moirai research session, fully stocked with cheese and drinks."""
        symbol = self._symbol_ = str(offering).upper()
        constants = self._constants_
        verbosity = self.verbosity
        batch = constants['batch_size']
        step = constants['batch_step']
        batch_range = range(batch - 2)
        cook_time = constants['cook_time']
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        p_tensor = self._p_tensor_
        research = self.__time_step__
        tensor = torch.tensor
        hstack = torch.hstack
        vstack = torch.vstack
        data_len = len(dataframe)
        if data_len < batch * 2:
            return False
        while data_len % batch != 0:
            dataframe = dataframe[1:]
            data_len = len(dataframe)
        cheese = tensor(
            dataframe.to_numpy(),
            requires_grad=True,
            **p_tensor,
            )
        sample = TensorDataset(cheese)
        candles = DataLoader(sample, batch_size=batch)
        self.metrics['loss'] = 0
        epochs = 0
        cooking = True
        t_cook = time.time()
        target_tmp = tensor(
            [0, 1, 0],
            **p_tensor,
            )
        targets = list()
        n_split = int(batch / 2)
        while cooking:
            self.metrics['acc'] = [0, 0]
            predictions = list()
            targets = list()
            for candle in iter(candles):
                candle = candle[0]
                target = candle[n_split:]
                candle = candle[:n_split]
                coated_candles = research(candle, study=True)
                target_tmp *= 0
                cdl_mean = candle[:, :4].mean(1).mean(0)
                tgt_mean = target[:, :4].mean(1).mean(0)
                if tgt_mean > cdl_mean:
                    target_tmp[0] += 1
                elif tgt_mean < cdl_mean:
                    target_tmp[2] += 1
                else:
                    target_tmp[1] += 1
                loss = loss_fn(coated_candles, target_tmp)
                loss.backward()
                optimizer.step()
                predictions.append(coated_candles)
                targets.append(target_tmp.clone())
                adj_loss = sqrt(loss.item())
                if coated_candles.argmax(0) == target_tmp.argmax(0):
                    correct = 1
                else:
                    correct = 0
                self.metrics['acc'][0] += correct
                self.metrics['acc'][1] += 1
                self.metrics['loss'] += adj_loss
                if verbosity == 3:
                    prefix = self._prefix_
                    print(prefix, 'prediction tail:', predictions[-1].tolist())
                    print(prefix, 'target tail:', targets[-1].tolist())
                    print(prefix, 'accuracy:', self.metrics['acc'])
                    print(prefix, 'loss:', self.metrics['loss'])
                    print(prefix, 'epochs:', self.metrics['epochs'])
            self.metrics['epochs'] += 1
            epochs += 1
            if verbosity == 2:
                prefix = self._prefix_
                print(prefix, 'prediction tail:', predictions[-1].tolist())
                print(prefix, 'target tail:', targets[-1].tolist())
                print(prefix, 'accuracy:', self.metrics['acc'])
                print(prefix, 'loss:', self.metrics['loss'])
                print(prefix, 'epochs:', self.metrics['epochs'])
            if time.time() - t_cook >= cook_time:
                cooking = False
        self.metrics['loss'] = self.metrics['loss'] / epochs
        predictions.append(research(cheese[-batch:], study=False))
        predictions = vstack(predictions)
        targets = vstack(targets)
        if plot:
            self.__time_plot__(predictions, targets)
        if keep_predictions:
            self.predictions[symbol] = predictions.flatten(0).tolist()
        if verbosity == 1:
            prefix = self._prefix_
            print(prefix, 'prediction tail:', predictions[-1].tolist())
            print(prefix, 'target tail:', targets[-1].tolist())
            print(prefix, 'accuracy:', self.metrics['acc'])
            print(prefix, 'loss:', self.metrics['loss'])
            print(prefix, 'epochs:', self.metrics['epochs'])
        self.__manage_state__(call_type=1)
        return True
