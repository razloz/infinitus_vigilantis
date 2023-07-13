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
            dropout=0.118,
            activation=torch.nn.functional.leaky_relu,
            layer_norm_eps=0.000118,
            batch_first=True,
            norm_first=True,
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
        self.n_batch = n_batch
        self.n_symbols = n_symbols
        self.metrics = dict(
            training_epochs=0,
            training_error=0,
            training_time=0,
            validation_accuracy=0,
            validation_error=0,
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
            self.network.load_state_dict(state['network'])
        except FileNotFoundError:
            print('No state found, creating default.')
            self.save_state()
        except Exception as details:
            print('Exception', *details.args)

    def save_state(self):
        state_path = self.state_path
        torch.save({
            'metrics': self.metrics,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state': self.state_dict(),
            }, state_path,
            )
        print('Saved state.')

    def train_network(self, hours=72, checkpoint=10):
        network = self.network
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        training_data = self.training_data
        t_bernoulli = self.t_bernoulli
        epoch = self.metrics['training_epochs']
        elapsed = 0
        timesteps = self.n_split - self.n_batch
        self.train()
        print('Training started.')
        start_time = time.time()
        while elapsed < hours:
            epoch_time = time.time()
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
            error /= timesteps
            ts = time.time()
            elapsed = ((ts - start_time) / 60) / 60
            epoch_time = ((ts - epoch_time) / 60) / 60
            self.metrics['training_epochs'] = epoch
            self.metrics['training_error'] = error
            self.metrics['training_time'] += epoch_time
            print('\nState:', state, '\nTargets:', targets)
            print('epochs:', self.metrics['training_epochs'])
            print('epoch_time:', epoch_time * 60, 'minutes')
            print('training_error:', self.metrics['training_error'])
            print('training_time:', self.metrics['training_time'], 'hours')
            print('session_time:', elapsed, 'hours')
            if epoch % checkpoint == 0:
                self.validate_network()
                self.save_state()
        if epoch % checkpoint != 0:
            self.validate_network()
            self.save_state()
        print(f'Training finished after {elapsed} hours.')

    def validate_network(self, threshold=0.0005):
        validation_data = self.validation_data
        t_bernoulli = self.t_bernoulli
        timesteps = self.n_split - self.n_batch
        n_error = self.n_symbols * self.n_batch
        network = self.network
        accuracy = 0
        batches = 0
        mae = 0
        self.eval()
        print('Starting validation routine.')
        for inputs, targets in iter(validation_data):
            state = network(inputs, inputs).mean(-1)
            error = (state - targets).abs()
            correct = error[error <= threshold].shape[0]
            accuracy += correct / n_error
            mae += error.flatten().mean(0).item()
            batches += 1
        self.metrics['validation_accuracy'] = accuracy / batches
        self.metrics['validation_error'] = mae / timesteps
        print('validation_accuracy:', self.metrics['validation_accuracy'])
        print('validation_error:', self.metrics['validation_error'])
        print('Validation complete.')
