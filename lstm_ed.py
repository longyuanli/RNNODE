import logging

import numpy as np
import pandas as pd
import torch
import pickle
import torch.nn as nn
from scipy.stats import multivariate_normal, norm
from torch.utils.data import DataLoader
from torch.distributions import Normal, kl_divergence
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import trange
from .model_utils import windowing, EarlyStopping, Detector, flatten_padded_batches

class LSTMED(Detector):
    def __init__(self, name: str='LSTM-ED', num_epochs: int=10, batch_size: int=20, lr: float=1e-3,
                 hidden_size: int=20, sequence_length: int=30, train_gaussian_percentage: float=0.25,
                 n_layers: tuple=(1, 1), use_bias: tuple=(True, True), dropout: tuple=(0, 0),
                 seed: int=None, gpu: int = None, details=True,patience : int=20, verbose=False):
        super(LSTMED, self).__init__(__name__, name, seed, details=details)
        if gpu is not None:
            self.device = torch.device(f"cuda:{gpu}")
        else:
            self.device = torch.device("cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.checkpoint_dir = f"/tmp/{self.name}_{self.seed}.pt"
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.train_gaussian_percentage = train_gaussian_percentage
        self.verbose = verbose
        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.detector = None
        self.mean, self.cov, self.std = None, None, None


    def fit(self, ts:np.array):

        def train():
            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                X = data.to(self.device)
                optimizer.zero_grad()
                output = self.detector(X)
                loss = nn.MSELoss(reduction='mean')(output, X)
                train_loss += loss.item() / len(train_loader.dataset)
                loss.backward()
                optimizer.step()
            return train_loss

        def valid():
            valid_loss = 0
            for batch_idx, data in enumerate(valid_loader):
                with torch.no_grad():
                    X = data.to(self.device)
                    output = self.detector(X)
                    loss = nn.MSELoss(reduction='mean')(output, X)
                    valid_loss += loss.item() / len(valid_loader.dataset)
            return valid_loss

        ts = ts.astype(np.float32)
        self.x_dim = ts.shape[1]
        ts_chunks = windowing(ts, window_size=self.sequence_length)
        num_train = int(ts_chunks.shape[0] * 0.95)
        train_chunks = ts_chunks[:num_train]
        valid_chunks = ts_chunks[num_train:]
        train_loader = DataLoader(train_chunks, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_chunks, batch_size=self.batch_size, shuffle=True)
        self.detector = LSTMEDModule(ts.shape[1], self.hidden_size,
                                     self.n_layers, self.use_bias, self.dropout,
                                     device=self.device).to(self.device)
        self.early_stopping = EarlyStopping(patience=self.patience,
                                            verbose=self.verbose,
                                            checkpoint_dir=self.checkpoint_dir)
        optimizer = torch.optim.Adam(self.detector.parameters(), lr=self.lr)

        for epoch in trange(self.num_epochs):
            self.detector.train()
            train_loss = train()
            #self.detector.eval()
            valid_loss = valid()
            if self.verbose and epoch % 1 == 0:
                    print(f">> epoch: {epoch}, train_loss:{train_loss:.3f}"
                          f" valid_loss: {valid_loss}")

            self.early_stopping(valid_loss, self.detector)
            if self.early_stopping.early_stop:
                self.detector.load_state_dict(pickle.load(open(self.checkpoint_dir, 'rb')))
                break

        error_vectors = []
        for batch_idx, data in enumerate(valid_loader):
            with torch.no_grad():
                X = data.to(self.device)
                output = self.detector(X)
                error = nn.L1Loss(reduction='none')(output, X)
                error_vectors += list(error.view(-1, ts.shape[1]).cpu().numpy())

        self.mean = np.mean(error_vectors, axis=0)
        self.cov = np.cov(error_vectors, rowvar=False)
        self.std = np.std(error_vectors, axis=0)
        #self.mean = np.mean(error_vectors, axis=0)
        #self.cov = np.cov(error_vectors, rowvar=False)
        #self.std = np.std(error_vectors, axis=0)

    def predict(self, ts: np.array, aggregate = True):
        mvnormal = multivariate_normal(self.mean, self.cov, allow_singular=True)
        normal = norm(self.mean, self.std)
        ts = ts.astype(np.float32)
        length_padding = ts.shape[0] % self.sequence_length
        self.x_dim = ts.shape[1]
        ts_ready = windowing(ts, window_size=self.sequence_length, step=self.sequence_length)
        if length_padding > 0:
            ts_last = ts[-self.sequence_length:, :][None, :, :]
            ts_ready = np.concatenate([ts_ready, ts_last], axis=0)
        ts_ready_torch = torch.FloatTensor(ts_ready).to(self.device)
        output = self.detector(ts_ready_torch)
        with torch.no_grad():
            error = nn.L1Loss(reduction='none')(output, ts_ready_torch).cpu().numpy()
        errors= flatten_padded_batches(error, length_padding)
        if aggregate:
            scores = -mvnormal.logpdf(errors)
            return scores[:,None]
        else:
            scores = -normal.logpdf(errors)
            return scores

    def density_estimation(self, ts:np.array):
        ts = ts.astype(np.float32)
        length_padding = ts.shape[0] % self.sequence_length
        self.x_dim = ts.shape[1]
        ts_ready = windowing(ts, window_size=self.sequence_length, step=self.sequence_length)
        if length_padding > 0:
            ts_last = ts[-self.sequence_length:, :][None, :, :]
            ts_ready = np.concatenate([ts_ready, ts_last], axis=0)
        ts_ready_torch = torch.FloatTensor(ts_ready).to(self.device)
        output = self.detector(ts_ready_torch)
        with torch.no_grad():
            output = self.detector(ts_ready_torch)
            output= flatten_padded_batches(output.cpu().numpy(), length_padding)
            rv = norm(output, self.std)
        return rv




class LSTMEDModule(nn.Module):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple,
                 device=None):
        super(LSTMEDModule, self).__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        if device is None:
            self.device = torch.device("cpu")

        else:
            self.device = device
        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0],
                               dropout=self.dropout[0]).to(device)
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1],
                               dropout=self.dropout[1]).to(device)
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features).to(device)

    def _init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers[0], batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.n_layers[0], batch_size, self.hidden_size).to(self.device))

    def forward(self, ts_batch, return_latent: bool=False):
        batch_size = ts_batch.shape[0]

        # 1. Encode the timeseries to make use of the last hidden state.
        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(ts_batch.float(), enc_hidden)  # .float() here or .double() for the model

        # 2. Use hidden state as initialization for our Decoder-LSTM
        dec_hidden = enc_hidden

        # 3. Also, use this hidden state to get the first output aka the last point of the reconstructed timeseries
        # 4. Reconstruct timeseries backwards
        #    * Use true data for training decoder
        #    * Use hidden2output for prediction
        output = torch.zeros(ts_batch.size()).to(self.device)
        for i in reversed(range(ts_batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(ts_batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        return (output, enc_hidden[1][-1]) if return_latent else output
