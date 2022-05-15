"""Alpaca Markets Manager for the Infinitus Vigilantis application."""

import requests
import json
import os

__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2021, Daniel Ward'
__license__ = 'GPL v3'
__version__ = '2021.05'
__codename__ = 'bling'


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
        self.ALPACA_URL = r'https://paper-api.alpaca.markets/v2/account'
        self.DATA_URL = r'https://data.alpaca.markets/v2'
        self.CREDS = {
            "APCA-API-KEY-ID": r'{}'.format(ALPACA_ID),
            "APCA-API-SECRET-KEY": r'{}'.format(ALPACA_SECRET)
            } # self.CREDS

    def __query__(self, url):
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

    def candles(self, symbol, limit=None, start_date=None, end_date=None):
        """Get historical price data for a specific symbol."""
        if len(symbol) > 0:
            url = f'{self.DATA_URL}/stocks/{symbol}/bars?timeframe=1Min'
            if limit is not None: url += f'&limit={limit}'
            if start_date is not None: url += f'&start={start_date}'
            if end_date is not None: url += f'&end={end_date}'
            return self.__query__(url)
        return None
