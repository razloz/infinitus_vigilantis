"""Alpaca Markets Manager for the Infinitus Vigilantis application."""

import requests
import json
import os
import time

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class AlpacaShepherd:
    """Handler for communicating with the Alpaca Markets API."""
    def __init__(self):
        """Load auth keys and create session."""
        ALPACA_ID = os.environ["APCA_API_KEY_ID"]
        if not len(ALPACA_ID) > 0:
            print('AlpacaShepherd: ALPACA_ID required.')
            return None
        ALPACA_SECRET = os.environ["APCA_API_SECRET_KEY"]
        if not len(ALPACA_SECRET) > 0:
            print('AlpacaShepherd: ALPACA_SECRET required.')
            return None
        self._last_query = 0
        self.RATE_LIMIT = 0.33
        self.ALPACA_URL = r'https://paper-api.alpaca.markets/v2/'
        self.DATA_URL = r'https://data.alpaca.markets/v2'
        self.CREDS = {
            "APCA-API-KEY-ID": r'{}'.format(ALPACA_ID),
            "APCA-API-SECRET-KEY": r'{}'.format(ALPACA_SECRET)
            } # self.CREDS

    def __query__(self, url, debugging=False):
        if debugging:
            print(f'AlpacaShepherd: {url}')
        URI = r'{}'.format(url)
        rcode = 0
        elapsed = time.time() - self._last_query
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        with requests.Session() as sess:
            rx = sess.get(URI, headers=self.CREDS)
            self._last_query = time.time()
            rcode = rx.status_code
            if rcode == 200:
                r = json.loads(rx.text)
                if debugging:
                    print(f'AlpacaShepherd: {rcode}\n{r}')
                return r
            else:
                print(f'AlpacaShepherd: Query returned code {rcode}.')
        return rcode

    def calendar(self):
        """Get market calendar for open and close times."""
        return self.__query__(f'{self.ALPACA_URL}/calendar')

    def clock(self):
        """Get market status and hours of operation."""
        return self.__query__(f'{self.ALPACA_URL}/clock')

    def asset(self, symbol):
        """Get details about a specific asset."""
        if not isinstance(symbol, str): return None
        return self.__query__(f'{self.ALPACA_URL}/assets/{symbol}')

    def assets(self):
        """Get an array of assets."""
        return self.__query__(f'{self.ALPACA_URL}/assets')

    def candles(self, symbols, **kwargs):
        """Get historical price data for a symbol or list of symbols."""
        params = 'timeframe=1Min'
        for p in ('limit', 'start', 'end', 'page_token'):
            if p in kwargs:
                params += f'&{p}={kwargs[p]}'
        t = type(symbols)
        if t == list:
            url = f'{self.DATA_URL}/stocks/bars?symbols='
            for s in symbols:
                url += f'{s},'
            url = f'{url[:-1]}&'
        elif t == str and t.isalpha():
            url = f'{self.DATA_URL}/stocks/{symbols}/bars?'
        else:
            return None
        return self.__query__(f'{url}{params}')
