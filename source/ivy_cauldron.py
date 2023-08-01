"""Model for predictive feed-forward forecasting of stock movement."""
import torch
import json
import time
from os import path, mkdir, environ
from math import isclose
from torch.utils.data import DataLoader, TensorDataset
torch.autograd.set_detect_anomaly(True)
DEVICE_TYPE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
FLOAT = torch.float
PI = torch.pi
abspath = path.abspath
bernoulli = torch.bernoulli
vstack = torch.vstack
leaky_relu = torch.nn.functional.leaky_relu

class Cauldron(torch.nn.Module):
    def __init__(self, candelabrum=None, verbosity=1):
        super(Cauldron, self).__init__()
        if DEVICE_TYPE != 'cpu':
            environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
            if verbosity > 0:
                print('Disabled CUDA memory caching.')
            torch.cuda.empty_cache()
        if candelabrum is None:
            cdl_path = abspath('./candelabrum')
            candelabrum = torch.load(f'{cdl_path}/candelabrum.candles')
            candelabrum.to(DEVICE)
        else:
            candelabrum = candelabrum.clone().detach()
        with open(f'{cdl_path}/candelabrum.symbols', 'r') as f:
            self.symbols = json.loads(f.read())['symbols']
        n_batch = 5
        n_trim = candelabrum.shape[0]
        while n_trim % (n_batch * 2) != 0:
            n_trim -= 1
        candelabrum = candelabrum[-n_trim:, :, :]
        n_split = int(candelabrum.shape[0] / 2)
        n_symbols = candelabrum.shape[1]
        n_features = candelabrum.shape[2] - 1
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
        candelabrum = None
        self.network = torch.nn.Transformer(
            d_model=n_features,
            nhead=n_features,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=256,
            dropout=0.118,
            activation=leaky_relu,
            layer_norm_eps=1.18e-9,
            batch_first=True,
            norm_first=True,
            dtype=FLOAT,
            )
        self.network.to(DEVICE)
        self.loss_fn = torch.nn.HuberLoss()
        self.optimizer = torch.optim.Adagrad(self.parameters())
        self.n_split = n_split
        self.n_batch = n_batch
        self.n_symbols = n_symbols
        self.metrics = dict(
            training_epochs=0,
            training_error=0,
            training_time=0,
            validation_accuracy=0,
            validation_correct=0,
            validation_error=0,
            validation_total=0,
            )
        network_path = abspath('./network')
        if not path.exists(network_path):
                mkdir(network_path)
        self.state_path = f'{network_path}/.norn.state'
        self.verbosity = verbosity
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
            if self.verbosity > 1:
                print('No state found, creating default.')
            self.save_state()
        except Exception as details:
            if self.verbosity > 1:
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
        if self.verbosity > 1:
            print('Saved state.')

    def train_network(self, hours=72, checkpoint=10, validate=50):
        verbosity = self.verbosity
        network = self.network
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        training_data = self.training_data
        t_bernoulli = self.t_bernoulli
        epoch = self.metrics['training_epochs']
        elapsed = 0
        timesteps = self.n_split - self.n_batch
        p_range = range(self.n_symbols)
        p_last = p_range[-1]
        m = ' || {} \t\t {} \t\t || '
        self.train()
        if verbosity > 0:
            print('Training started.')
        start_time = time.time()
        while elapsed < hours:
            epoch_time = time.time()
            epoch += 1
            if verbosity > 0:
                print('Epoch:', epoch)
            error = 0
            optimizer.zero_grad()
            for inputs, targets in iter(training_data):
                state = leaky_relu(
                    network(
                        inputs * bernoulli(t_bernoulli),
                        inputs,
                        ).sum(-1),
                    )
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
            if verbosity > 1:
                print('\n || State \t\t Targets \t\t || ')
                for i in p_range:
                    s = round(state[-1][i].item(), 9)
                    t = round(targets[-1][i].item(), 9)
                    print(m.format(s, t))
            if verbosity > 0:
                print('epochs:', self.metrics['training_epochs'])
                print('epoch_time:', epoch_time * 60, 'minutes')
                print('training_error:', self.metrics['training_error'])
                print('training_time:', self.metrics['training_time'], 'hours')
                print('session_time:', elapsed, 'hours')
            if epoch % validate == 0:
                self.validate_network()
            if epoch % checkpoint == 0:
                self.save_state()
        if epoch % validate != 0:
            self.validate_network()
        if epoch % checkpoint != 0:
            self.validate_network()
            self.save_state()
        if verbosity > 0:
            print(f'Training finished after {elapsed} hours.')

    def validate_network(self, threshold=0.003):
        verbosity = self.verbosity
        validation_data = self.validation_data
        t_bernoulli = self.t_bernoulli
        n_batch = self.n_batch
        n_symbols = self.n_symbols
        timesteps = self.n_split - n_batch
        n_error = n_symbols * n_batch
        network = self.network
        accuracy = 0
        batches = 0
        error = 0
        self.eval()
        n_correct = 0
        n_total = 0
        if verbosity > 0:
            print('Starting validation routine.')
        for inputs, targets in iter(validation_data):
            state = leaky_relu(network(inputs, inputs).sum(-1))
            correct = 0
            for i in range(n_batch):
                for ii in range(n_symbols):
                    prediction = state[i][ii].item()
                    truth = targets[i][ii].item()
                    if isclose(prediction, truth, abs_tol=threshold):
                        correct += 1
            accuracy += correct / n_error
            n_correct += correct
            n_total += n_error
            error += (state - targets).abs().flatten().mean(0).item()
            batches += 1
        self.metrics['validation_accuracy'] = accuracy / batches
        self.metrics['validation_correct'] = n_correct
        self.metrics['validation_error'] = error / timesteps
        self.metrics['validation_total'] = n_total
        if verbosity > 0:
            print('correct predictions:', self.metrics['validation_correct'])
            print('total predictions:', self.metrics['validation_total'])
            print('validation_accuracy:', self.metrics['validation_accuracy'])
            print('validation_error:', self.metrics['validation_error'])
            print('Validation complete.')
