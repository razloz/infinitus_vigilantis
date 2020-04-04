"""Alpaca Markets Manager for the Infinitus Vigilantis application."""

import requests
import json
import os

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2020, Daniel Ward'
__license__ = 'GPL v3'
__version__ = '2020.04'
__codename__ = 'compass'


class AlpacaShepherd:
    """Handler for communicating with the Alpaca Markets API."""
    def __init__(self, timeframe='1Min', api_version='v2'):
        """Load auth keys and create session."""
        ALPACA_ID = os.environ["APCA_API_KEY_ID"]
        if not len(ALPACA_ID) > 0:
            print('AlpacaShepherd: ALPACA_ID required.')
            return None
        ALPACA_SECRET = os.environ["APCA_API_SECRET_KEY"]
        if not len(ALPACA_SECRET) > 0:
            print('AlpacaShepherd: ALPACA_SECRET required.')
            return None
        self.ALPACA_URL = os.environ["APCA_API_BASE_URL"]
        if not len(self.ALPACA_URL) > 0:
            durl = r'https://paper-api.alpaca.markets'
            print(f'AlpacaShepherd: ALPACA_URL missing. Using {durl}')
            self.ALPACA_URL = durl
        self.TIMEFRAME = str(timeframe)
        self.VERSION = str(api_version)
        self.DATA_URL = r'https://data.alpaca.markets/v1'
        self.CREDS = {
            "APCA-API-KEY-ID": r'{}'.format(ALPACA_ID),
            "APCA-API-SECRET-KEY": r'{}'.format(ALPACA_SECRET)
            } # self.CREDS

    def __query_alpaca__(self, url):
        URI = r'{}'.format(url)
        with requests.Session() as sess:
            rx = sess.get(URI, headers=self.CREDS)
            rcode = rx.status_code
            if rcode == 200:
                return json.loads(rx.text)
            else:
                print(f'AlpacaShepherd: Query returned code {rcode}.')
        return None

    def calendar(self):
        """Get market calendar for open and close times."""
        url = r'{}/{}/calendar'.format(self.ALPACA_URL, self.VERSION)
        return self.__query_alpaca__(url)

    def clock(self):
        """Get market status and hours of operation."""
        url = r'{}/{}/clock'.format(self.ALPACA_URL, self.VERSION)
        return self.__query_alpaca__(url)

    def asset(self, symbol):
        """Get details about a specific asset."""
        if not isinstance(symbol, str): return None
        url = r'{}/{}/assets'.format(self.ALPACA_URL, self.VERSION)
        url += r'/{}'.format(symbol)
        return self.__query_alpaca__(url)

    def assets(self):
        """Get an array of assets."""
        url = r'{}/{}/assets'.format(self.ALPACA_URL, self.VERSION)
        return self.__query_alpaca__(url)

    def candles(self, symbols, limit=None, start_date=None, end_date=None):
        """Get price data for symbols."""
        url = f'{self.DATA_URL}/bars/{self.TIMEFRAME}'
        syms = ''
        if isinstance(symbols, list):
            l = len(symbols)
            if l > 1:
                c = 0
                for sym in symbols:
                    c += 1
                    if not c == l:
                        syms += f'{sym},'
                    else:
                        syms += str(sym)
            elif l == 1:
                syms = str(sym)
        elif isinstance(symbols, str):
            syms = symbols
        if len(syms) > 0:
            q = f'{url}?symbols={syms}'
            if limit is not None: q += f'&limit={limit}'
            if start_date is not None: q += f'&start={start_date}'
            if end_date is not None: q += f'&end={end_date}'
            return self.__query_alpaca__(q)
        return None
