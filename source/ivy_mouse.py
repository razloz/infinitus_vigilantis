"""Three blind mice to predict the future."""
import torch
import traceback
import source.ivy_commons as icy
from os.path import abspath
from torch.nn import LSTM, Module, HuberLoss
from torch.optim import Adam
__author__ = 'Daniel Ward'
__copyright__ = 'Copyright 2022, Daniel Ward'
__license__ = 'GPL v3'


class ThreeBlindMice(Module):
    """Let the daughters of necessity shape the candles of the future."""
    def __init__(self, n_features, n_targets, *args, **kwargs):
        super(ThreeBlindMice, self).__init__(*args, **kwargs)
        self._device_type_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self._device_ = torch.device(self._device_type_)
        self._prefix_ = 'Moirai:'
        self._state_path_ = abspath('./rnn/moirai.state')
        self._n_features_ = n_features = int(n_features)
        self._n_targets_ = n_targets = int(n_targets)
        self._n_hidden_ = n_hidden = int(n_features ** 2)
        self._tensor_args_ = {
            'device': self._device_,
            'dtype': torch.float,
            'requires_grad': True
            }
        rnn_params = {
            'input_size': n_features,
            'hidden_size': n_hidden,
            'proj_size': n_targets,
            'batch_first': True,
            'num_layers': n_targets,
            'dropout': 0.34,
            'device': self._device_
            }
        self._moirai_ = LSTM(**rnn_params)
        self._optimizer_ = Adam(self._moirai_.parameters(), lr=0.008)
        self._loss_fn_ = HuberLoss()
        self._accuracy_ = 0
        self.predictions = dict()
        self.to(self._device_)
        print(self._prefix_, f'Set device to {self._device_type_.upper()}.')

    def __manage_state__(self, call_type=0, verbose=False):
        """Handles loading and saving of the RNN state."""
        try:
            p = self._state_path_
            if call_type == 0:
                d = self._device_type_
                self.load_state_dict(torch.load(p, map_location=d))
                print(self._prefix_, self._moirai_.state_dict().items())
                print(self._prefix_, self.predictions.keys())
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

    def study(self, symbol, candles, n_predictions=5, min_size=90):
        if not all((
            len(candles.keys()) == self._n_features_,
            len(candles.index) >= min_size
            )): return False
        self.__manage_state__(call_type=0)
        rnn = self._moirai_
        optimizer = self._optimizer_
        loss_fn = self._loss_fn_
        n_targets = self._n_targets_
        candles = torch.tensor(candles.to_numpy(), **self._tensor_args_)
        candles_index = range(0, candles.size(0), n_predictions)
        last_step = candles_index[-1]
        target_accuracy = 99.95
        timeout=1000
        epochs = 0
        accuracy = 0
        loss = 1
        while accuracy < target_accuracy:
            if epochs > timeout:
                print(self._prefix_, 'Timeout reached. Breaking loop.')
                break
            sealed_candles = None
            for i in candles_index:
                if sealed_candles is not None:
                    continue
                optimizer.zero_grad()
                # Clotho molds the candles
                ii = i + n_predictions
                iii = ii + n_predictions
                if iii <= last_step:
                    wicks = candles[i:ii, :]
                    targets = candles[ii:iii, :n_targets]
                    waxed = torch.cat([rnn(t)[0] for t in wicks.split(1)])
                else:
                    # Atropos seals the candles
                    wicks = candles[n_predictions:, :]
                    targets = None
                    waxed = [rnn(t)[0] for t in wicks.split(1)]
                    sealed_candles = torch.cat(waxed)
                # Lachesis measures the candles
                if targets is not None:
                    loss = loss_fn(waxed, targets)
                    loss.backward()
                    optimizer.step()
            if sealed_candles is not None:
                self.predictions[symbol] = sealed_candles.detach().cpu().numpy()
            else:
                self.predictions[symbol] = None
            try:
                accuracy = round((1 - loss.item()) * 100, 2)
            except ValueError:
                accuracy = 0
            self._accuracy_ = accuracy
            epochs += 1
            epoch_str = 'epoch' if epochs == 1 else 'epochs'
            msg = f'( {symbol}: {epochs} {epoch_str} )    {accuracy}% accuracy'
            print(self._prefix_, msg)
        msg = 'After {} epochs, a prediction accuracy of {}% was realized.'
        print(self._prefix_, msg.format(epochs, accuracy))
        self.__manage_state__(call_type=1)
        return True

