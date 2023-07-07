"""Model for predictive feed-forward forecasting of stock movement."""
import torch
import json
from os import path
from torch.utils.data import DataLoader, TensorDataset
torch.autograd.set_detect_anomaly(True)
DEVICE_TYPE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE_TYPE)
FLOAT = torch.float
PI = torch.pi
abspath = path.abspath
bernoulli = torch.bernoulli
vstack = torch.vstack
candelabrum = torch.load(abspath('./candelabrum/candelabrum.candles'))
candelabrum.to(DEVICE)
candelabrum = candelabrum.transpose(0, 1)
with open(abspath('./candelabrum/candelabrum.symbols'), 'r') as f:
    symbols = json.loads(f.read())['symbols']

print('Candelabrum:', candelabrum.shape)
print('Symbols:', symbols)
n_batch = 14
n_trim = candelabrum.shape[0]
while n_trim % (n_batch * 2) != 0:
    n_trim -= 1

candelabrum = candelabrum[-n_trim:, :, :]
n_split = int(candelabrum.shape[0] / 2)
n_symbols = candelabrum.shape[1]
n_features = candelabrum.shape[2] - 1
print('Candelabrum:', candelabrum.shape)
t_net = torch.nn.Transformer(
    d_model=n_features,
    nhead=5,
    num_encoder_layers=5,
    num_decoder_layers=5,
    dim_feedforward=64,
    dropout=0.1,
    activation=torch.nn.functional.leaky_relu,
    layer_norm_eps=1e-05,
    batch_first=True,
    norm_first=False,
    device=DEVICE,
    dtype=FLOAT,
    )
t_loss = torch.nn.HuberLoss()
t_opt = torch.optim.Adagrad(t_net.parameters())
t_state = None
t_grad = None
t_data = DataLoader(
    TensorDataset(
        candelabrum[:-n_split, :, :-1][:-n_batch].requires_grad_(True),
        candelabrum[:-n_split, :, -1][n_batch:],
        ),
    batch_size=n_batch,
    drop_last=True,
    )
t_valid = DataLoader(
    TensorDataset(
        candelabrum[n_split:, :, :-1][:-n_batch],
        candelabrum[n_split:, :, -1][n_batch:],
        ),
    batch_size=n_batch,
    drop_last=True,
    )
t_bernoulli = torch.full(
    [n_batch, n_symbols, n_features],
    0.382,
    device=DEVICE,
    dtype=FLOAT,
    )
epoch = 0
while True:
    epoch += 1
    print('Epoch:', epoch)
    running_mean = 0
    for inputs, targets in iter(t_data):
        t_state = t_net(
            inputs * bernoulli(t_bernoulli),
            inputs,
            ).mean(-1)
        loss = t_loss(t_state, targets)
        loss.backward()
        t_opt.step()
        running_mean += loss.item()
    running_mean /= n_split
    print(t_state, '\n', targets, '\n\n')
    print('Epoch Loss:', running_mean)


