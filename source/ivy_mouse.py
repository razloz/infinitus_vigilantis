#!../.env/bin/python3
"""Three blind mice to predict the future."""
import random
import torch
import traceback
import pandas as pd
import ivy_commons as icy
from os import listdir
from os.path import abspath
from torch.nn import LSTM, Module, HuberLoss
from torch.nn.functional import log_softmax
from torch.optim import Adam
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class ThreeBlindMice(Module):
    def __init__(self, n_features, *args, **kwargs):
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._prefix_ = 'Three Blind Mice:'
        self._state_path_ = abspath(f'../rnn/moirai.state')
        self._tensor_args_ = dict(
            device=self._device_,
            dtype=torch.float,
            requires_grad=True
            )
        self.n_features = int(n_features)
        self.n_hidden = int(self.n_features ** 2)
        self.clotho = LSTM(
            input_size=self.n_features,
            hidden_size=self.n_hidden,
            num_layers=3,
            batch_first=True,
            proj_size=4
            )
        self.lachesis = HuberLoss()
        self.atropos = Adam(self.clotho.parameters())
        self.to(self._device_)
        print(f'Set device to {self._device_type_.upper()}.')

    def __manage_state__(self, call_type=0, verbose=False):
        try:
            p = self._state_path_
            if call_type == 0:
                d = self._device_type_
                self.load_state_dict(torch.load(p, map_location=d))
                if verbose:
                    print(self._prefix_, 'Loaded RNN state.')
            elif call_type == 1:
                torch.save(self.state_dict(), p)
                if verbose:
                    print(self._prefix_, 'Saved RNN state.')
        except Exception as details:
            if verbose:
                print(self._prefix_, 'Encountered an exception.')
                traceback.print_exception(details)

    def study(self, inputs, targets, epochs=1):
        self.__manage_state__(call_type=0)
        t_args = self._tensor_args_
        inputs = torch.tensor(inputs[:-1], **t_args)
        targets = torch.tensor(targets[1:], **t_args)
        split_input = inputs.split(1, dim=0)
        for i in range(epochs):
            self.atropos.zero_grad()
            outputs = list()
            for t in split_input:
                output, hidden = self.clotho(t)
                outputs.append(output)
            outputs = torch.cat(outputs, dim=0)
            loss = self.lachesis(outputs, targets)
            loss.backward()
            self.atropos.step()
        try:
            accuracy = round((1 - loss.item()) * 100, 2)
        except ValueError:
            accuracy = 'NaN'
        print(self._prefix_, f'{accuracy}% accuracy after studying.')
        self.__manage_state__(call_type=1)


if __name__ == '__main__':
    file_name = 'AMD-2019-01-03'
    tz = 'America/New_York'
    p = lambda c: pd.to_datetime(c, utc=True).tz_convert(tz)
    kwargs = dict(
        index_col=0,
        parse_dates=True,
        infer_datetime_format=True,
        date_parser=p
        )
    data_path = abspath('../candelabrum')
    ivy_files = listdir(data_path)
    total_files = len(ivy_files)
    moirai = ThreeBlindMice(30)
    print('Starting study loop...')
    files_studied = 0
    while True:
        file_index = 0
        file_name = random.choice(ivy_files)
        remaining_files = len(ivy_files)
        if remaining_files == 0:
            break
        for i in range(remaining_files):
            if ivy_files[i] == file_name:
                file_index = i
        ivy_files.pop(file_index)
        if file_name[-4:] != '.ivy':
            continue
        print(f'( {files_studied} / {total_files} ) The Moirai are inspecting {file_name}')
        ivy_path = abspath(f'{data_path}/{file_name}')
        with open(ivy_path) as f:
            ivy_data = pd.read_csv(f, **kwargs)
        indicators = icy.get_indicators(ivy_data)
        ivy_data.pop('utc_ts')
        cheese = ivy_data.merge(
            indicators,
            left_index=True,
            right_index=True
            )
        labels = cheese.keys()
        inputs = cheese[labels[4:]].to_numpy()
        targets = cheese[labels[:4]].to_numpy()
        skip_condition = [
            inputs.shape[1] != 30,
            targets.shape[1] != 4,
            inputs.shape[0] < 90,
            targets.shape[0] < 90
            ]
        if any(skip_condition):
            continue
        moirai.study(inputs, targets, epochs=55)
        files_studied += 1
    print('Study complete.')
