"""Three blind mice to predict the future."""
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2023, Daniel Ward'
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

    def update_webview(self, metrics, forecast, sigil):
        """Commission the Cartographer to plot the forecast."""
        symbols = self.symbols
        html = """<p style="color:white;text-align:center;font-size:18px"><b>"""
        row_keys = [
            ('accuracy', 'net_gain', 'n_trades'),
            ('n_profit', 'n_loss', 'n_doji'),
            ('avg_gain', 'max_gain', 'min_gain'),
            ('epochs', 'confidence'),
            ('days_mean', 'days_max', 'days_min'),
            ]
        percentiles = (
            'confidence',
            'avg_gain',
            'max_gain',
            'min_gain',
            'accuracy',
            'net_gain',
            )
        for row, keys in enumerate(row_keys):
            if row > 0:
                html += """<br>"""
            for k in keys:
                v = metrics[k]
                if hasattr(v, 'item'):
                    v = v.item()
                if k in percentiles:
                    v = f'{round(100 * v, 4)}%'
                html += """{0}: {1}&emsp;""".format(k, v)
        trade = self.metrics['trade']
        exit_trade = self.metrics['exit_trade']
        if trade:
            sym_index, price = trade
            if hasattr(price, 'item'):
                price = price.item()
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
