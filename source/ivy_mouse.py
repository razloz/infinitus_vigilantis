"""Three blind mice to predict the future."""
import json
import hashlib
import io
import time
import torch
import traceback
import numpy as np
import source.ivy_commons as icy
from pandas import read_csv
from torch import bernoulli, fft, nan_to_num, nn, stack, topk
from torch.optim import Adagrad
from torch.utils.data import DataLoader, TensorDataset
from math import inf, sqrt, pi
from os import listdir, mkdir
from os.path import abspath, exists
from source.ivy_cartography import cartography, plot_candelabrum
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class ThreeBlindMice(nn.Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, symbols, offerings, cook_time=inf, trim=34, verbosity=2):
        """Beckon the Norn."""
        super(ThreeBlindMice, self).__init__()
        #Setup
        torch.autograd.set_detect_anomaly(True)
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self.to(self._device_)
        dev = self._device_
        tfloat = torch.float
        self._state_path_ = abspath('./rnn')
        if not exists(self._state_path_):
            mkdir(self._state_path_)
        iota = 1 / 137
        phi = 1.618033988749894
        n_sample = 3
        n_symbols = len(symbols)
        offerings = offerings.to(dev).transpose(0, 1)[trim:]
        n_time = int(offerings.shape[0])
        while n_time % n_sample != 0:
            n_time -= 1
        offerings = offerings[-n_time:]
        n_lin_in = 11
        n_lin_out = 9
        n_output = n_symbols * n_sample
        output_dims = [n_sample, n_symbols]
        hidden_input = n_sample * n_symbols * n_lin_out
        hidden_output = 377 * 9
        hidden_dims = [377, 9]
        #Tensors
        self.candles = offerings.clone().detach()
        self.targets = offerings[:, :, -1].clone().detach().softmax(1)
        self.cdl_means = self.candles[:, :, -2].clone().detach()
        self.trade_array = torch.zeros(
            *self.targets.shape,
            device=dev,
            dtype=tfloat,
            )
        wax = [-phi, pi, phi, -1, 0, 1, phi, -pi, -phi]
        self.wax = list()
        for _ in range(n_symbols):
            for _ in range(n_sample):
                self.wax += wax
        self.wax = torch.tensor(
            self.wax,
            device=dev,
            dtype=tfloat,
            ).view(n_symbols, n_sample, 9)
        self.wax.requires_grad_(True)
        self.candelabrum = TensorDataset(
            self.candles[:-n_sample],
            self.targets[n_sample:],
            )
        self.cauldron = DataLoader(
            self.candelabrum,
            batch_size=n_sample,
            )
        #Functions
        self.bilinear = nn.Bilinear(
            in1_features=n_lin_in,
            in2_features=n_lin_in,
            out_features=n_lin_out,
            bias=True,
            device=dev,
            dtype=tfloat,
            )
        self.input_cell = nn.LSTMCell(
            input_size=hidden_input,
            hidden_size=hidden_output,
            bias=True,
            device=dev,
            dtype=tfloat,
        )
        self.output_cell = nn.LSTM(
            input_size=hidden_dims[1],
            hidden_size=hidden_dims[1],
            num_layers=3,
            bias=True,
            device=dev,
            dtype=tfloat,
        )
        self.normalizer = nn.InstanceNorm1d(
            num_features=n_symbols,
            eps=1e-09,
            momentum=0.1,
            affine=False,
            track_running_stats=False,
            device=dev,
            dtype=tfloat,
            )
        self.optimizer = Adagrad(self.parameters())
        self.loss_fn = nn.functional.binary_cross_entropy
        self.activation = nn.functional.leaky_relu
        self.rfft = fft.rfft
        #Settings
        self.metrics = dict(
            rnn_loss=0,
            n_profit=0,
            n_trades=0,
            accuracy=0,
            net_gain=0,
            avg_gain=0,
            max_gain=0,
            max_loss=0,
            epochs=0,
            )
        self.symbols = symbols
        self.cook_time = cook_time
        self.n_sample = n_sample
        self.n_symbols = n_symbols
        self.n_time = n_time
        self.n_lin_in = n_lin_in
        self.n_lin_out = n_lin_out
        self.n_output = n_output
        self.output_dims = output_dims
        self.hidden_input = hidden_input
        self.hidden_output = hidden_output
        self.hidden_dims = hidden_dims
        self.pi = pi
        self.verbosity = verbosity
        self._prefix_ = prefix = 'Moirai:'
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

    def inscribe_sigil(self, candles):
        """
            Let Clotho mold the candles
            Let Lachesis measure the candles
            Let Atropos seal the candles
            Let Awen contain the wax
        """
        activation = self.activation
        symbols = self.n_symbols
        dims = self.hidden_dims
        output_dims = self.output_dims
        output_size = output_dims[0] * output_dims[1]
        candles = self.normalizer(candles.transpose(0, 1))
        candles = self.rfft(candles)
        candles = self.bilinear(candles.real, candles.imag)
        candle_wax = list()
        for i, batch in enumerate(self.wax):
            for ii, wax in enumerate(batch):
                candle = candles[i, ii].view(3, 3)
                candle_wax.append((wax.view(3, 3) @ candle).flatten() ** -1)
        candle_wax = stack(candle_wax).log_softmax(1).flatten()
        candles = self.input_cell(candle_wax)[0].view(dims)
        candles = self.output_cell(candles)[0].flatten()
        sigil = topk(candles, output_size, sorted=False).indices
        inscription = candles[sigil].view(output_dims).softmax(1)
        return inscription.clone()

    def research(self):
        """Moirai research session, fully stocked with cheese and drinks."""
        prefix = self._prefix_
        verbosity = self.verbosity
        inscribe_sigil = self.inscribe_sigil
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        cdl_means = self.cdl_means
        targets = self.targets
        trade_array = self.trade_array.clone()
        topk = torch.topk
        cook_time = self.cook_time
        n_sample = self.n_sample
        n_time = self.n_time
        least_loss = inf
        loss_retry = 0
        loss_timeout = 1000
        n_save = 3
        trade_range = range(n_sample)
        self.train()
        cooking = True
        print(prefix, 'Research started.\n')
        t_cook = time.time()
        while cooking:
            trade_array *= 0
            trade_index = 0
            rnn_loss = 0
            max_gain = 0
            max_loss = 0
            net_gain = 0
            n_trades = 0
            n_profit = 0
            prev_trades = [None for _ in trade_range]
            for candles, targets in iter(self.cauldron):
                optimizer.zero_grad()
                sigil = inscribe_sigil(candles)
                loss = loss_fn(sigil, targets)
                loss.backward()
                optimizer.step()
                rnn_loss += loss.item()
                for i in trade_range:
                    prob, index = topk(sigil[i], 1)
                    ti = trade_index + i
                    price = cdl_means[ti][index]
                    trade_array[ti][index] = prob.item()
                    if prev_trades[i] is not None:
                        prev_index, prev_price = prev_trades[i]
                        curr_price = cdl_means[ti][prev_index]
                        gain = (curr_price - prev_price) / prev_price
                        net_gain += gain
                        if gain > max_gain:
                            max_gain = gain
                        elif gain < max_loss:
                            max_loss = gain
                        n_trades += 1
                        if gain > 0:
                            n_profit += 1
                    prev_trades[i] = (index, price)
                trade_index += n_sample
                if verbosity > 1:
                    print(prefix, trade_index, 'loss =', rnn_loss / trade_index)
            self.metrics['rnn_loss'] = rnn_loss / n_time
            self.metrics['n_profit'] = n_profit
            self.metrics['n_trades'] = n_trades
            self.metrics['accuracy'] = n_profit / n_trades
            self.metrics['net_gain'] = net_gain.item()
            self.metrics['avg_gain'] = (net_gain / n_trades).item()
            self.metrics['max_gain'] = max_gain.item()
            self.metrics['max_loss'] = max_loss.item()
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
            if verbosity > 0:
                print(prefix, 'loss over time =', self.metrics['rnn_loss'])
            if self.metrics['epochs'] % n_save == 0:
                self.__manage_state__(call_type=1)
                if verbosity > 0:
                    self.update_webview(*self.get_predictions())
        self.trade_array = trade_array.clone().detach()
        if self.metrics['epochs'] % n_save != 0:
            self.__manage_state__(call_type=1)
            if verbosity > 0:
                self.update_webview(*self.get_predictions())
        if verbosity > 0:
            for metric_key, metric_value in self.metrics.items():
                print(f'{prefix} {metric_key} = {metric_value}')
            print(prefix, 'Research complete.\n')
        return self.trade_array.clone().detach()

    def get_predictions(self):
        """Output for the last batch."""
        banner = ''.join(['*' for _ in range(80)])
        n_sample = self.n_sample
        prefix = self._prefix_
        symbols = self.symbols
        topk = torch.topk
        self.eval()
        sigil = self.inscribe_sigil(self.candles[-n_sample:])
        print(banner)
        forecast = list()
        for t in range(sigil.shape[0]):
            prob, sym = topk(sigil[t], 1)
            s = symbols[sym.item()]
            v = prob.item()
            forecast.append((s, v))
            print(f'{prefix} day({t}) {s} {v} prob')
        print(banner)
        return (
            dict(self.metrics),
            forecast,
            sigil.clone().detach().cpu().numpy(),
            )

    def update_webview(self, metrics, forecast, sigil):
        """Commission the Cartographer to plot the forecast."""
        html = """<p style="color:white;text-align:center;font-size:18px"><b>"""
        row_keys = [
            ('accuracy', 'net_gain', 'n_profit', 'n_trades'),
            ('avg_gain', 'max_gain', 'max_loss'),
            ('epochs', 'rnn_loss'),
            ]
        for row, keys in enumerate(row_keys):
            if row > 0:
                html += """<br>"""
            for k in keys:
                v = metrics[k]
                html += """{0}: {1}&emsp;""".format(k, v)
        ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.localtime())
        html += """timestamp: {0}</b></p>""".format(ts)
        with open(abspath('./resources/metrics.html'), 'w+') as f:
            f.write(html)
        plot_candelabrum(sigil, self.symbols)
        for day, probs in enumerate(forecast):
            symbol = probs[0]
            cdl_path = abspath(f'./candelabrum/{symbol}.ivy')
            candles = read_csv(cdl_path, index_col=0, parse_dates=True)
            c_path = abspath(f'./resources/forecast_{day}.png')
            cartography(symbol, candles, chart_path=c_path, chart_size=365)
        print(self._prefix_, 'Webview updated.')
