"""Three blind mice to predict the future."""
import json
import hashlib
import io
import time
import torch
import traceback
import numpy as np
import source.ivy_commons as icy
from torch import bernoulli, fft, nan_to_num, nn, stack
from torch.optim import Adagrad
from torch.utils.data import DataLoader, TensorDataset
from math import inf, sqrt
from os import listdir, mkdir
from os.path import abspath, exists
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, symbols, offerings, cook_time=inf, verbosity=2):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__()
        torch.autograd.set_detect_anomaly(True)
        self.iota = iota = 1 / 137
        self.phi = phi = 0.618033988749894
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = dev = torch.device(self._device_type_)
        self.to(dev)
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_): mkdir(self._state_path_)
        self.symbols = list(symbols)
        offerings = offerings.to(dev)
        offerings = offerings.transpose(0, 1)
        constants = {}
        constants['cook_time'] = cook_time
        constants['n_sample'] = n_sample = 3
        constants['n_symbols'] = n_symbols = len(self.symbols)
        constants['n_time'] = int(offerings.shape[0])
        constants['n_lin_in'] = n_lin_in = 19
        constants['n_lin_out'] = n_lin_out = 9
        constants['n_output'] = n_symbols * n_sample
        constants['output_dims'] = [n_sample, n_symbols]
        constants['hidden_input'] = n_sample * n_symbols * n_lin_out
        constants['hidden_output'] = 2048
        constants['lr_init'] = iota ** phi
        constants['lr_decay'] = 1 + iota / (phi - 1)
        constants['weight_decay'] = iota / phi
        constants['eps'] = iota * 1e-6
        while constants['n_time'] % n_sample != 0:
            constants['n_time'] -= 1
        offerings = offerings[-constants['n_time']:]
        self._constants_ = dict(constants)
        self._prefix_ = prefix = 'Moirai:'
        tfloat = torch.float
        self.rfft = fft.rfft
        self.candles = offerings.clone().detach()
        self.candles.requires_grad_(True)
        self.targets = offerings[:, :, -1].clone().detach()
        self.targets = (self.targets + iota).softmax(1)
        self.cdl_means = offerings[:, :, :4].mean(2).clone().detach()
        self.candelabrum = TensorDataset(
            self.candles[:-n_sample],
            self.targets[n_sample:],
            self.cdl_means[:-n_sample],
            )
        self.cauldron = DataLoader(
            self.candelabrum,
            batch_size=n_sample,
            )
        self.bilinear = nn.Bilinear(
            in1_features=n_lin_in,
            in2_features=n_lin_in,
            out_features=n_lin_out,
            bias=True,
            device=dev,
            dtype=tfloat,
            )
        self.input_cell = nn.LSTMCell(
            input_size=constants['hidden_input'],
            hidden_size=constants['hidden_output'],
            bias=True,
            device=dev,
            dtype=tfloat,
        )
        self.output_cell = nn.LSTMCell(
            input_size=constants['hidden_output'],
            hidden_size=constants['hidden_output'],
            bias=True,
            device=dev,
            dtype=tfloat,
        )
        self.linear = nn.Linear(
            in_features=constants['hidden_output'],
            out_features=constants['n_output'],
            bias=True,
            device=dev,
            dtype=tfloat,
            )
        self.normalizer = nn.InstanceNorm1d(
            num_features=constants['n_symbols'],
            eps=1e-09,
            momentum=0.1,
            affine=False,
            track_running_stats=False,
            device=dev,
            dtype=tfloat,
            )
        self.loss_fn = nn.HuberLoss(
            reduction='mean',
            delta=1.0,
            )
        self.optimizer = Adagrad(
            self.parameters(),
            lr=phi,
            lr_decay=constants['lr_decay'],
            weight_decay=constants['weight_decay'],
            initial_accumulator_value=constants['lr_init'],
            eps=constants['eps'],
            foreach=True,
            )
        self.metrics = dict(
            accuracy=0,
            avg_gain=0,
            epochs=0,
            rnn_loss=0,
            net_gain=0,
            trades_profit=0,
            trades_total=0,
            )
        self.verbosity = int(verbosity)
        if self.verbosity > 1:
            for key, value in constants.items():
                print(prefix, f'set {key} to {value}')
            print('')
        self.__manage_state__(call_type=0)

    def __manage_state__(self, call_type=0):
        """Handles loading and saving of the RNN state."""
        state_path = f'{self._state_path_}/.norn.state'
        if call_type == 0:
            try:
                state = torch.load(state_path, map_location=self._device_type_)
                if 'metrics' in state:
                    self.metrics = dict(state['metrics'])
                self.load_state_dict(state['moirai'])
                if self.verbosity > 2:
                    print(self._prefix_, 'Loaded RNN state.')
            except FileNotFoundError:
                if self.verbosity > 2:
                    print(self._prefix_, 'No state found, creating default.')
                self.__manage_state__(call_type=1)
            except Exception as details:
                if self.verbosity > 1:
                    print(self._prefix_, *details.args)
        elif call_type == 1:
            torch.save(
                {
                    'metrics': self.metrics,
                    'moirai': self.state_dict(),
                    },
                state_path,
                )
            if self.verbosity > 2:
                print(self._prefix_, 'Saved RNN state.')

    def inscribe_sigil(self, wax):
        """
            Let Clotho mold the candles
            Let Lachesis measure the candles
            Let Atropos seal the candles
            Let Awen contain the wax
        """
        sigil = self.normalizer(wax.transpose(0, 1))
        sigil = self.rfft(sigil)
        sigil = self.bilinear(sigil.real, sigil.imag)
        sigil = sigil.sigmoid().softmax(1).flatten()
        sigil = self.input_cell(sigil)[0]
        sigil = self.output_cell(sigil)[0]
        sigil = self.linear(sigil).view(self._constants_['output_dims'])
        sigil = self.normalizer(sigil).softmax(1)
        return sigil.clone()

    def research(self):
        """Moirai research session, fully stocked with cheese and drinks."""
        prefix = self._prefix_
        print(prefix, 'Research started.')
        constants = self._constants_
        cook_time = constants['cook_time']
        inscribe_sigil = self.inscribe_sigil
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        symbols = self.symbols
        topk = torch.topk
        n_sample = constants['n_sample']
        n_save = 100
        n_time = constants['n_time']
        sample_range = range(n_sample)
        least_loss = inf
        cooking = True
        t_cook = time.time()
        loss_retry = 0
        loss_timeout = 1000
        self.train()
        while cooking:
            net_gain = 0
            rnn_loss = 0
            prev_trade = (None, None)
            trades = {s: list() for s in symbols}
            trades_profit = 0
            trades_total = 0
            for candles, target, cdl_means in iter(self.cauldron):
                optimizer.zero_grad()
                sigil = inscribe_sigil(candles)
                loss = loss_fn(sigil, target)
                loss.backward()
                optimizer.step()
                rnn_loss += loss.item()
                for day in sample_range:
                    trade = topk(sigil[day], 1).indices
                    day_avg = cdl_means[day]
                    prev_sym = prev_trade[0]
                    prev_avg = prev_trade[1]
                    for sym in symbols:
                        trades[sym].append(1 if sym == trade else 0)
                    if prev_sym is not None:
                        gain = 0
                        curr_avg = day_avg[prev_sym]
                        if curr_avg > 0 < prev_avg:
                            gain = (curr_avg - prev_avg) / prev_avg
                        net_gain += gain
                        trades_total += 1
                        if gain > 0:
                            trades_profit += 1
                        trades[symbols[prev_sym]][-1] = -1
                    prev_trade = (trade, day_avg[trade])
            accuracy = 1 + ((trades_profit - trades_total) / trades_total)
            self.metrics['accuracy'] = accuracy
            self.metrics['avg_gain'] = (net_gain / n_time).item()
            self.metrics['net_gain'] = net_gain.item()
            self.metrics['rnn_loss'] = sqrt(rnn_loss / n_time)
            self.metrics['trades_profit'] = trades_profit
            self.metrics['trades_total'] = trades_total
            self.metrics['epochs'] += 1
            if self.metrics['rnn_loss'] < least_loss:
                loss_retry = 0
                least_loss = self.metrics['rnn_loss']
            else:
                loss_retry += 1
                if loss_retry == loss_timeout:
                    cooking = False
            elapsed = time.time() - t_cook
            if elapsed >= cook_time:
                cooking = False
            if self.verbosity > 0:
                for metric_key, metric_value in self.metrics.items():
                    print(f'{prefix} {metric_key} = {metric_value}')
                print('')
            if self.metrics['epochs'] % n_save == 0:
                self.__manage_state__(call_type=1)
        if self.metrics['epochs'] % n_save != 0:
            self.__manage_state__(call_type=1)
        return self.get_predictions()

    def get_predictions(self, n_png=''):
        """Output for the last batch."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        banner = ''.join(['*' for _ in range(80)])
        constants = self._constants_
        n_sample = constants['n_sample']
        prefix = self._prefix_
        symbols = self.symbols
        topk = torch.topk
        self.eval()
        sigil = self.inscribe_sigil(self.candles[-n_sample:])
        print(banner)
        forecast = list()
        for t in range(sigil.shape[0]):
            inscriptions = topk(sigil[t], max_trades)
            for i in range(max_trades):
                s = symbols[inscriptions.indices[i]]
                v = inscriptions.values[i]
                forecast.append((s, v))
                print(f'{prefix} day({t}) {s} {v} prob')
        print(banner)
        sig = sigil.clone().detach().cpu().numpy()
        xticks = range(sig.shape[1])
        plt.clf()
        fig = plt.figure(figsize=(38.40, 5.40))
        ax = fig.add_subplot()
        ax.set_xlabel('Symbol')
        ax.set_ylabel('Prob')
        ax.set_xticks(xticks)
        ax.set_xticklabels(self.symbols, fontweight='light')
        ax.tick_params(axis='x', which='major', labelsize=7, pad=5, rotation=90)
        width_adj = [0.7, 0.5, 0.3]
        colors = [(0.34, 0.34, 1, 1), (0.34, 1, 0.34, 1), (1, 0.34, 0.34, 1)]
        colors_set = 0
        for day in range(sig.shape[0]):
            bar_params = dict(
                width=width_adj[day],
                align='edge',
                aa=True,
                color=colors[day],
                edgecolor=(1, 1, 1, 0.05),
                )
            if colors_set < 3:
                bar_params['label'] = f'Forecast Day {day + 1}'
                colors_set += 1
            ax.bar(xticks, sig[day], **bar_params)
        title = f'Candelabrum probabilities over the next {sig.shape[0]} days'
        fig.suptitle(title, fontsize=18)
        fig.legend(ncol=1, fontsize='xx-large', fancybox=True)
        plt.savefig(f'./resources/candelabrum{n_png}.png')
        plt.clf()
        fig = plt.figure(figsize=(19.20, 10.80))
        ax = fig.add_subplot()
        ax.set_xlabel('Signal Gradient')
        grad = self.signals.grad.clone().detach()
        grad = grad.sum(2).mean(1).sqrt().cpu().numpy()
        ax.plot(grad)
        plt.savefig(f'./resources/signal_gradient{n_png}.png')
        plt.close()
        return forecast
