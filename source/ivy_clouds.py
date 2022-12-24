""""""
import requests
import pandas as pd
import os
import time
from requests.exceptions import RequestException
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'

def __collect__(file_path):
    """Collect weather data from local storage."""
    collected = pd.read_csv(file_path, infer_datetime_format=True)
    return collected

def __gather__(
    dataset='global-marine',
    startdate='2019-01-01',
    enddate='2019-01-02',
    ):
    """Gather weather data from NCEI.NOAA.GOV"""
    RETRY_CODES = [408, 429]
    RETRY_WAIT = 180
    endpoint = 'https://www.ncei.noaa.gov/access/services/data/v1'
    url = f'{endpoint}?dataset={dataset}'
    url += f'&startDate={startdate}&endDate={enddate}'
    url += '&boundingBox=90,-180,-90,180'
    url += '&includeStationName=1'
    url += '&includeStationLocation=1'
    rcode = 0
    try:
        with requests.Session() as sess:
            rx = sess.get(url, timeout=5)
            rcode = rx.status_code
    except RequestException as err:
        print(f'Encountered {type(err)}: {err.args}.')
        rcode = 408
    finally:
        if rcode == 200:
            pd.save_csv('./weather.data', rx.text)
        elif rcode in RETRY_CODES:
            print(f'Retry in {RETRY_WAIT} seconds.')
            time.sleep(RETRY_WAIT)
            return __gather__(
                dataset=dataset,
                startdate=startdate,
                enddate=enddate,
                )
        else:
            print(f'{self.PREFIX} Query returned code {rcode}.')
            return rcode
