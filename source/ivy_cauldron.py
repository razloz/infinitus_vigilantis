"""Model for predictive forecasting of stock movement."""
import torch
import json
import time
from random import randint
from os import path, mkdir, environ
from math import isclose
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.init import uniform_
torch.autograd.set_detect_anomaly(True)
DEVICE_TYPE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
FLOAT = torch.float
PI = torch.pi
abspath = path.abspath
vstack = torch.vstack
leaky_relu = torch.nn.functional.leaky_relu


class Cauldron(torch.nn.Module):
    def __init__(self, candelabrum=None, verbosity=2, no_caching=True):
        super(Cauldron, self).__init__()
        if DEVICE_TYPE != 'cpu' and no_caching:
            environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
            if verbosity > 1:
                print('Disabled CUDA memory caching.')
            torch.cuda.empty_cache()
        if candelabrum is None:
            cdl_path = abspath('./candelabrum')
            candelabrum = torch.load(f'{cdl_path}/candelabrum.candles')
        else:
            candelabrum = candelabrum.clone().detach()
        candelabrum.to(DEVICE)
        with open(f'{cdl_path}/candelabrum.symbols', 'r') as f:
            self.symbols = json.loads(f.read())['symbols']
        n_batch = 9
        n_slice = n_batch * 2
        n_time, n_symbols, n_data = candelabrum.shape
        self.input_mask = torch.full(
            [1, n_batch],
            0.382,
            device=DEVICE,
            dtype=FLOAT,
            )
        self.network = torch.nn.Transformer(
            d_model=n_batch,
            nhead=n_batch,
            num_encoder_layers=512,
            num_decoder_layers=512,
            dim_feedforward=4096,
            dropout=0.118,
            activation=leaky_relu,
            layer_norm_eps=1.18e-34,
            batch_first=True,
            norm_first=True,
            device=DEVICE,
            dtype=FLOAT,
            )
        self.loss_fn = torch.nn.HuberLoss()
        self.optimizer = torch.optim.Adam(self.parameters())
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
        self.candelabrum = candelabrum
        self.n_batch = n_batch
        self.n_slice = n_slice
        self.n_time = n_time
        self.n_symbol = n_symbols - 1
        self.n_data = n_data
        self.batch_max = n_time - n_slice - 1
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
            iota = (1 / 137) ** 9
            if self.verbosity > 0:
                print('No state found, creating default.')
                print(f'Initializing weights with bounds of {-iota} to {iota}')
            for name, param in self.named_parameters():
                if param.requires_grad:
                    uniform_(param, -iota, iota)
            self.save_state()
        except Exception as details:
            if self.verbosity > 0:
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
        if self.verbosity > 0:
            print('Saved state.')

    def random_batch(self):
        """Returns a random batch of inputs and targets for a random symbol."""
        n_batch = self.n_batch
        batch_start = randint(0, self.batch_max)
        batch_end = batch_start + self.n_slice
        batch_symbol = randint(0, self.n_symbol)
        batch = self.candelabrum[batch_start:batch_end, batch_symbol, -1]
        return (
            batch[:n_batch].view(1, n_batch),
            batch[n_batch:].view(1, n_batch),
            )

    def train_network(self, n_depth=9, hours=72, checkpoint=15, validate=45):
        verbosity = self.verbosity
        network = self.network
        loss_fn = self.loss_fn
        optimizer = self.optimizer
        input_mask = self.input_mask
        random_batch = self.random_batch
        bernoulli = torch.bernoulli
        epoch = self.metrics['training_epochs']
        elapsed = 0
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
            inputs, targets = random_batch()
            depth = 0
            mask = bernoulli(input_mask)
            while depth <= n_depth:
                state = network(inputs * mask, inputs)
                loss = loss_fn(state, targets)
                loss.backward()
                optimizer.step()
                depth += 1
                if verbosity > 1:
                    print('\n', state, '\nstate shape', state.shape)
                    print('\n', targets, '\ntargets shape', targets.shape, '\n')
            error = loss.item()
            ts = time.time()
            elapsed = ((ts - start_time) / 60) / 60
            epoch_time = ((ts - epoch_time) / 60) / 60
            self.metrics['training_epochs'] = epoch
            self.metrics['training_error'] = error
            self.metrics['training_time'] += epoch_time
            if verbosity > 0:
                print('epoch_time:', epoch_time * 60, 'minutes')
                print('training_error:', self.metrics['training_error'])
                print('training_time:', self.metrics['training_time'], 'hours')
                print('session_time:', elapsed, 'hours')
            #if epoch % validate == 0:
            #    self.validate_network()
            if epoch % checkpoint == 0:
                self.save_state()
        #if epoch % validate != 0:
        #    self.validate_network()
        if epoch % checkpoint != 0:
            self.save_state()
        #self.validate_network()
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
