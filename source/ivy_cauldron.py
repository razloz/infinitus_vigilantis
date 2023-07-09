"""Model for predictive feed-forward forecasting of stock movement."""
import torch
import json
from os import path
import time
from torch.utils.data import DataLoader, TensorDataset
torch.autograd.set_detect_anomaly(True)
DEVICE_TYPE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
FLOAT = torch.float
PI = torch.pi
abspath = path.abspath
bernoulli = torch.bernoulli
vstack = torch.vstack


class Cauldron(torch.nn.Module):
    def __init__(self, candelabrum=None, n_batch=7):
        super(Cauldron, self).__init__()
        if candelabrum is None:
            cdl_path = abspath('./candelabrum')
            candelabrum = torch.load(f'{cdl_path}/candelabrum.candles')
            candelabrum.to(DEVICE)
            candelabrum = candelabrum.transpose(0, 1)
        else:
            candelabrum = candelabrum.clone().detach()
        with open(f'{cdl_path}/candelabrum.symbols', 'r') as f:
            self.symbols = json.loads(f.read())['symbols']
        n_trim = candelabrum.shape[0]
        while n_trim % (n_batch * 2) != 0:
            n_trim -= 1
        candelabrum = candelabrum[-n_trim:, :, :]
        n_split = int(candelabrum.shape[0] / 2)
        n_symbols = candelabrum.shape[1]
        n_features = candelabrum.shape[2] - 1
        self.network = torch.nn.Transformer(
            d_model=n_features,
            nhead=5,
            num_encoder_layers=7,
            num_decoder_layers=7,
            dim_feedforward=64,
            dropout=0.1,
            activation=torch.nn.functional.leaky_relu,
            layer_norm_eps=1e-05,
            batch_first=True,
            norm_first=False,
            device=DEVICE,
            dtype=FLOAT,
            )
        self.loss_fn = torch.nn.HuberLoss()
        self.optimizer = torch.optim.Adagrad(self.parameters())
        self.grad = None
        self.training_data = DataLoader(
            TensorDataset(
                candelabrum[:-n_split, :, :-1][:-n_batch].requires_grad_(True),
                candelabrum[:-n_split, :, -1][n_batch:],
                ),
            batch_size=n_batch,
            drop_last=True,
            )
        self.validation_data = DataLoader(
            TensorDataset(
                candelabrum[n_split:, :, :-1][:-n_batch],
                candelabrum[n_split:, :, -1][n_batch:],
                ),
            batch_size=n_batch,
            drop_last=True,
            )
        self.t_bernoulli = torch.full(
            [n_batch, n_symbols, n_features],
            0.382,
            device=DEVICE,
            dtype=FLOAT,
            )
        self.candelabrum = candelabrum
        self.n_split = n_split
        self.metrics = dict(
            epochs=0,
            error=0,
            total_time=0,
            )
        self.state_path = abspath('./rnn/.norn.state')
        self.load_state()

    def load_state(self):
        state_path = self.state_path
        try:
            state = torch.load(state_path, map_location=DEVICE_TYPE)
            if 'metrics' in state:
                self.metrics = dict(state['metrics'])
            self.load_state_dict(state['state'])
            self.optimizer.load_state_dict(state['optimizer'])
        except FileNotFoundError:
            print('No state found, creating default.')
            self.save_state()
        except Exception as details:
            print('Exception', *details.args)

    def save_state(self):
        state_path = self.state_path
        torch.save({
            'metrics': self.metrics,
            'state': self.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            }, state_path,
            )
        print('Saved state.')

    def train_network(self, hours=72, checkpoint=0.25):
        network = self.network
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        training_data = self.training_data
        t_bernoulli = self.t_bernoulli
        epoch = self.metrics['epochs']
        elapsed = 0
        n_split = self.n_split
        self.train()
        print('Training started.')
        start_time = time.time()
        while elapsed < hours:
            epoch += 1
            print('Epoch:', epoch)
            error = 0
            optimizer.zero_grad()
            for inputs, targets in iter(training_data):
                state = network(
                    inputs * bernoulli(t_bernoulli),
                    inputs,
                    ).mean(-1)
                loss = loss_fn(state, targets)
                loss.backward()
                error += loss.item()
            optimizer.step()
            error /= n_split
            elapsed = ((time.time() - start_time) / 60) / 60
            self.metrics['epochs'] = epoch
            self.metrics['error'] = error
            self.metrics['total_time'] += elapsed
            print('\nState:', state, '\nTargets:', targets)
            print('Error:', error)
            print('Elapsed:', elapsed, '\n')
            if elapsed % checkpoint == 0:
                self.save_state()
        if elapsed % checkpoint != 0:
            self.save_state()
        print(f'Training finished after {elapsed} hours.')

