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
        iota = (1 / 137) ** 9
        phi = 1.618033988749894
        n_symbols = len(symbols)
        offerings = offerings.to(dev).transpose(0, 1)[trim:]
        n_time = int(offerings.shape[0])
        n_lin_in = 11
        n_lin_out = 9
        n_signals = int(n_symbols * 0.118)
        hidden_input = n_symbols * n_lin_out
        hidden_dims = [256, 9]
        hidden_output = hidden_dims[0] * hidden_dims[1]
        #Tensors
        self.candles = offerings.clone().detach()
        self.targets = offerings[:, :, -1].clone().detach().log_softmax(1)
        self.cdl_means = self.candles[:, :, -2].clone().detach()
        self.trade_array = torch.zeros(
            n_signals * 2,
            device=dev,
            dtype=tfloat,
            ).requires_grad_(True)
        self.trade_profit = torch.zeros(
            1,
            device=dev,
            dtype=tfloat,
            ).requires_grad_(True)
        wax = [-phi, pi, phi, -1, 0, 1, phi, -pi, -phi]
        self.wax = list()
        for _ in range(n_symbols):
            self.wax += wax
        self.wax = torch.tensor(
            self.wax,
            device=dev,
            dtype=tfloat,
            ).view(n_symbols, 1, 9)
        self.wax.requires_grad_(True)
        self.candelabrum = TensorDataset(
            self.candles[:-1],
            self.targets[1:],
            )
        self.cauldron = DataLoader(
            self.candelabrum,
            batch_size=1,
            drop_last=True,
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
            num_layers=55,
            bias=True,
            device=dev,
            dtype=tfloat,
            dropout=0.118,
            bidirectional=True,
        )
        self.normalizer = nn.InstanceNorm1d(
            num_features=n_symbols,
            eps=iota,
            momentum=0.1,
            affine=False,
            track_running_stats=False,
            device=dev,
            dtype=tfloat,
            )
        self.cauldron_loss = nn.KLDivLoss(
            reduction='batchmean',
            log_target=True,
            )
        self.cauldron_optimizer = Adagrad(self.parameters())
        self.trade_rnn = nn.GRU(
            input_size=self.trade_array.shape[0],
            hidden_size=self.trade_array.shape[0],
            num_layers=32,
            bias=True,
            batch_first=True,
            dropout=0.5,
            bidirectional=True,
            device=dev,
            )
        self.trade_loss = nn.HuberLoss()
        self.trade_optimizer = Adagrad(self.trade_rnn.parameters())
        self.rfft = fft.rfft
        #Settings
        self.metrics = dict(
            mae=0,
            n_profit=0,
            n_trades=0,
            accuracy=0,
            net_gain=0,
            avg_gain=0,
            max_gain=0,
            min_gain=0,
            epochs=0,
            trade=None,
            exit_trade=False,
            days_trading=0,
            days_mean=0,
            days_max=0,
            days_min=0,
            )
        self.iota = iota
        self.phi = phi
        self.symbols = symbols
        self.cook_time = cook_time
        self.n_forecast = n_symbols * 2
        self.n_symbols = n_symbols
        self.n_time = n_time
        self.n_lin_in = n_lin_in
        self.n_lin_out = n_lin_out
        self.n_signals = n_signals
        self.hidden_input = hidden_input
        self.hidden_output = hidden_output
        self.hidden_dims = hidden_dims
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
                self.cauldron_optimizer.load_state_dict(state['c_optim'])
                self.trade_optimizer.load_state_dict(state['t_optim'])
                self.wax.grad = state['wax']
                self.trade_array.grad = state['trade_array']
                self.trade_profit.grad = state['trade_profit']
                if self.verbosity > 0:
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
                    'c_optim': self.cauldron_optimizer.state_dict(),
                    't_optim': self.trade_optimizer.state_dict(),
                    'trade_array': self.trade_array.grad,
                    'trade_profit': self.trade_profit.grad,
                    'wax': self.wax.grad,
                    },
                state_path,
                )
            if self.verbosity > 0:
                print(self._prefix_, 'Saved RNN state.')

    def cauldron_bubble(self, candles):
        """
            Let Clotho mold the candles
            Let Lachesis measure the candles
            Let Atropos seal the candles
            Let Awen contain the wax
        """
        symbols = self.n_symbols
        dims = self.hidden_dims
        n_forecast = self.n_forecast
        candles = self.normalizer(candles.transpose(0, 1))
        candles = self.rfft(candles)
        candles = self.bilinear(candles.real, candles.imag)
        candle_wax = self.wax.view(symbols, 3, 3) @ candles.view(symbols, 3, 3)
        candle_wax = candle_wax.abs().log_softmax(-1).flatten()
        candles = self.input_cell(candle_wax)[0].view(dims)
        candles = self.output_cell(candles)[0].flatten()
        sigil = candles[-symbols:].view(1, symbols).log_softmax(1)
        return sigil.clone()

    def research(self, n_save=8, loss_timeout=1000):
        """Moirai research session, fully stocked with cheese and drinks."""
        cauldron_bubble = self.cauldron_bubble
        cauldron_loss = self.cauldron_loss
        cook_time = self.cook_time
        least_loss = inf
        loss_retry = 0
        n_time = self.n_time - 1
        optimizer = self.cauldron_optimizer
        prefix = self._prefix_
        verbosity = self.verbosity
        print(prefix, 'Research started.\n')
        cooking = True
        t_cook = time.time()
        while cooking:
            self.train()
            mae = 0
            for candles, targets in iter(self.cauldron):
                optimizer.zero_grad()
                sigil = cauldron_bubble(candles)
                loss = cauldron_loss(sigil, targets)
                loss.backward()
                optimizer.step()
                mae += loss.item()
            mae /= n_time
            self.metrics['epochs'] += 1
            if mae < least_loss:
                loss_retry = 0
                least_loss = mae
            else:
                loss_retry += 1
                if loss_retry == loss_timeout:
                    cooking = False
            elapsed = time.time() - t_cook
            if elapsed >= cook_time:
                cooking = False
            if verbosity > 0:
                print(prefix, 'Mean Absolute Error =', mae)
            self.metrics['mae'] = mae
            if self.metrics['epochs'] % n_save == 0:
                self.__manage_state__(call_type=1)
        if self.metrics['epochs'] % n_save != 0:
            self.__manage_state__(call_type=1)
        if verbosity > 0:
            print(prefix, 'Research complete.\n')
        return self.trade_array

    def trade(self, freeze=False, max_epochs=1000, n_save=13):
        """Trade signals."""
        cauldron_bubble = self.cauldron_bubble
        prefix = self._prefix_
        verbosity = self.verbosity
        cdl_means = self.cdl_means
        targets = self.targets
        tensor = torch.tensor
        dev = self._device_
        tfloat = torch.float
        topk = torch.topk
        iota = self.iota
        phi = self.phi
        n_signals = self.n_signals
        days_min = 3
        days_max = 21
        trade_array = self.trade_array
        trade_rnn = self.trade_rnn
        trade_loss = self.trade_loss
        trade_optimizer = self.trade_optimizer
        trade_shape = trade_array.shape[0]
        loss_target = torch.zeros(1, device=dev, dtype=tfloat)
        trade_profit = self.trade_profit
        cauldron = self.cauldron
        if not freeze:
            self.train()
        else:
            self.eval()
        epochs = 0
        while epochs < max_epochs:
            longest_trade = 0
            trade_symbols = list()
            day_index = 0
            max_gain = 0
            min_gain = 0
            net_gain = 0
            n_trades = 0
            n_profit = 0
            n_loss = 0
            n_doji = 0
            trade = None
            trade_days = list()
            days_trading = 0
            for key in (
                'accuracy',
                'avg_gain',
                'max_gain',
                'min_gain',
                'days_trading',
                'days_mean',
                'days_max',
                'days_min',
                ):
                self.metrics[key] = 0
            exit_trade = False
            trade_symbol = None
            signal_indices = list()
            for candles, _ in iter(cauldron):
                sigil = cauldron_bubble(candles).flatten()
                b_sig = topk(sigil, n_signals, largest=True, sorted=True)
                s_sig = topk(sigil, n_signals, largest=False, sorted=True)
                ndx = b_sig.indices.tolist() + s_sig.indices.tolist()
                probs = b_sig.values.tolist() + s_sig.values.tolist()
                probs = tensor(probs, device=dev, dtype=tfloat)
                trade_probs = (trade_array + probs).view(1, trade_shape)
                trade_probs = trade_rnn(trade_probs)[0].flatten()
                trade_probs = trade_probs[-trade_shape:].softmax(0)
                signal_indices.append(ndx)
                if trade:
                    days_trading += 1
                    if exit_trade:
                        price = cdl_means[day_index][trade_symbol]
                        gain = (price - entry_price) / entry_price
                        net_gain += gain
                        if gain > max_gain:
                            max_gain = gain
                        elif gain < min_gain:
                            min_gain = gain
                        n_trades += 1
                        if gain > 0:
                            n_profit += 1
                        elif gain < 0:
                            n_loss += 1
                        else:
                            n_doji += 1
                        #if not freeze:
                            #loss = sum([loss, (phi / (phi ** gain)) * iota])
                        trade_days.append(days_trading)
                        days_trading = 0
                        trade = None
                        exit_trade = False
                    elif days_trading == 1:
                        trade[1] = cdl_means[day_index][trade[0]]
                    elif days_trading >= days_min:
                        trade_symbol, entry_price = trade
                        exit_trade = any([
                            trade_symbol != sym_index,
                            days_trading == days_max,
                            ])
                if not trade:
                    sym_index = topk(trade_probs, 1, largest=True).indices
                    sym_index = ndx[sym_index.item()]
                    trade = [sym_index, None]
                    days_trading = 0
                day_index += 1
                if not freeze:
                    loss = trade_profit + ((1 - (net_gain / day_index)) * iota)
                    loss = trade_loss(trade_profit, loss_target)
                    loss.backward()
                    trade_optimizer.step()
            epochs += 1
            self.metrics['n_profit'] = n_profit
            self.metrics['n_loss'] = n_loss
            self.metrics['n_doji'] = n_doji
            self.metrics['n_trades'] = n_trades
            self.metrics['net_gain'] = net_gain
            if n_trades != 0:
                self.metrics['accuracy'] = n_profit / n_trades
                self.metrics['avg_gain'] = (net_gain / n_trades)
                self.metrics['max_gain'] = max_gain
                self.metrics['min_gain'] = min_gain
                self.metrics['days_trading'] = days_trading
                self.metrics['days_mean'] = sum(trade_days) / len(trade_days)
                self.metrics['days_max'] = max(trade_days)
                self.metrics['days_min'] = min(trade_days)
            self.metrics['trade'] = trade
            self.metrics['exit_trade'] = exit_trade
            for metric_key, metric_value in self.metrics.items():
                print(f'{prefix} {metric_key} = {metric_value}')
            print('')
            if epochs % n_save == 0:
                self.__manage_state__(call_type=1)
        if epochs % n_save != 0:
            self.__manage_state__(call_type=1)
        self.update_webview(*self.get_predictions())

    def get_predictions(self):
        """Output for the last batch."""
        banner = ''.join(['*' for _ in range(80)])
        prefix = self._prefix_
        symbols = self.symbols
        topk = torch.topk
        self.eval()
        sigil, forecast = self.cauldron_bubble(self.candles[-1:])
        print(banner)
        prob, sym = topk(sigil, 1)
        s = symbols[sym.item()]
        v = prob.exp().item()
        forecast = (s, v)
        print(f'{prefix} forecasted symbol {s} with {v} probability.')
        trade = self.metrics['trade']
        if trade:
            sym_index, price = trade
            days_trading = self.metrics['days_trading']
            print(f'{prefix} active trade is {symbols[sym_index]} @ {price}')
            print(f'{prefix} with {days_trading} days in trade.')
        else:
            print(f'{prefix} no active trades.')
        print(banner)
        return (
            dict(self.metrics),
            forecast,
            sigil.clone().detach().cpu().numpy(),
            )

    def update_webview(self, metrics, forecast, sigil):
        """Commission the Cartographer to plot the forecast."""
        symbols = self.symbols
        html = """<p style="color:white;text-align:center;font-size:18px"><b>"""
        row_keys = [
            ('accuracy', 'net_gain', 'n_trades'),
            ('n_profit', 'n_loss', 'n_doji'),
            ('avg_gain', 'max_gain', 'min_gain'),
            ('epochs', 'mae'),
            ('days_mean', 'days_max', 'days_min'),
            ]
        for row, keys in enumerate(row_keys):
            if row > 0:
                html += """<br>"""
            for k in keys:
                v = metrics[k]
                if type(v) not in (int, float):
                    v = v.item()
                html += """{0}: {1}&emsp;""".format(k, v)
        trade = self.metrics['trade']
        exit_trade = self.metrics['exit_trade']
        if trade:
            sym_index, price = trade
            days_trading = self.metrics['days_trading']
            symbol = symbols[sym_index]
            html += """<br>"""
            html += f"""<a href="{symbol}.png">active trade: {symbol}</a>"""
            html += f""" @ {price}&emsp;({days_trading} days in trade)"""
            html += f""" exit_trade = {exit_trade} """
            cdl_path = abspath(f'./candelabrum/{symbol}.ivy')
            candles = read_csv(cdl_path, index_col=0, parse_dates=True)
            c_path = abspath(f'./resources/{symbol}.png')
            cartography(symbol, candles, chart_path=c_path, chart_size=365)
        ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.localtime())
        html += """timestamp: {0}</b></p>""".format(ts)
        with open(abspath('./resources/metrics.html'), 'w+') as f:
            f.write(html)
        plot_candelabrum(sigil, self.symbols)
        symbol = forecast[0]
        cdl_path = abspath(f'./candelabrum/{symbol}.ivy')
        candles = read_csv(cdl_path, index_col=0, parse_dates=True)
        c_path = abspath(f'./resources/forecast.png')
        cartography(symbol, candles, chart_path=c_path, chart_size=365)
        print(self._prefix_, 'Webview updated.')
