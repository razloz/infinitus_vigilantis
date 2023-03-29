"""Alpaca Markets Manager for the Infinitus Vigilantis application."""
import requests
import json
import os
import time
from requests.exceptions import RequestException
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class AlpacaShepherd:
    """Handler for communicating with the Alpaca Markets API."""
    def __init__(self):
        """Load auth keys and create session."""
        self.PREFIX = 'AlpacaShepherd:'
        ALPACA_ID = os.environ["APCA_API_KEY_ID"]
        if not len(ALPACA_ID) > 0:
            print(f'{self.PREFIX} ALPACA_ID required.')
            return None
        ALPACA_SECRET = os.environ["APCA_API_SECRET_KEY"]
        if not len(ALPACA_SECRET) > 0:
            print(f'{self.PREFIX} ALPACA_SECRET required.')
            return None
        self._last_query = 0
        self.RATE_LIMIT = 0.33
        self.RETRY_WAIT = 90
        self.RETRY_CODES = [408, 429]
        self.ALPACA_URL = r'https://paper-api.alpaca.markets/v2/'
        self.DATA_URL = r'https://data.alpaca.markets/v2'
        self.CREDS = {
            "APCA-API-KEY-ID": r'{}'.format(ALPACA_ID),
            "APCA-API-SECRET-KEY": r'{}'.format(ALPACA_SECRET)
            } # self.CREDS

    def __query__(self, url, just_text=False):
        l = len(url)
        if l > 2048:
            print(f'{self.PREFIX} URL of length {l} exceeds the 2048 limit.')
            return None
        URI = r'{}'.format(url)
        rcode = 0
        elapsed = time.time() - self._last_query
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        try:
            with requests.Session() as sess:
                self._last_query = time.time()
                rx = sess.get(URI, headers=self.CREDS, timeout=5)
                rcode = rx.status_code
        except RequestException as err:
            print(f'{self.PREFIX} Encountered {type(err)}: {err.args}.')
            rcode = 408
        finally:
            if rcode == 200:
                if just_text:
                    return rx.text
                else:
                    return json.loads(rx.text)
            elif rcode in self.RETRY_CODES:
                print(f'{self.PREFIX} Retry in {self.RETRY_WAIT} seconds.')
                time.sleep(self.RETRY_WAIT)
                return self.__query__(url, just_text=just_text)
            else:
                print(f'{self.PREFIX} Query returned code {rcode}.')
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
        params = ''
        for p in ('timeframe', 'start', 'end', 'limit', 'page_token'):
            if p in kwargs:
                params += f'&{p}={kwargs[p]}'
                if p == 'start':
                    params += 'T00:00:00-04:00'
                elif p == 'end':
                    params += 'T23:59:59-04:00'
        if len(params) > 0:
            params = params[1:]
        t = type(symbols)
        if t in (list, tuple):
            url = f'{self.DATA_URL}/stocks/bars?symbols='
            for s in symbols:
                url += f'{s},'
            url = f'{url[:-1]}&'
            return self.__query__(f'{url}{params}', just_text=True)
        elif t == str and symbols.isalpha():
            url = f'{self.DATA_URL}/stocks/{symbols}/bars?{params}'
            return self.__query__(url)
        else:
            return None
