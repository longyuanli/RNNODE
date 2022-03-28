import torch
from torch import nn, optim
from torch.distributions import Normal, kl_divergence, StudentT
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm_notebook
from .model_utils import windowing, EarlyStopping, Detector, StudentT_converter, Normal_converter,StudentT_Normal_converter
#StudentT.arg_constraints['df'].lower_bound = 1


class RDSSM(Detector):
    def __init__(self, name: str='RDSSM', num_epochs: int=100, batch_size: int=20, lr: float=1e-3,
                 sequence_length: int=30,  step: int = 1,
                 seed: int=None, gpu: int = None, details=True,z_dim: int = 10, h_dim : int=100,
                 patience : int=10, verbose = False, decoder_type='StudentT', beta:float=1, IWAE: int=5, normal_score=False):
        super(RDSSM,self).__init__(__name__, name, seed, details=details)
        if gpu is not None:
            self.device = torch.device(f"cuda:{gpu}")
        else:
            self.device = torch.device("cpu")
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.IWAE=IWAE
        self.lr = lr
        self.sequence_length = sequence_length
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.step = step
        self.patience = patience
        self.verbose = verbose
        self.checkpoint_dir = f"/tmp/{self.name}_{self.seed}.pt"
        self.model_class = RDSSM_module
        if decoder_type == "StudentT":
            self.decoder_class = StudentT_Decoder
        else:
            self.decoder_class = Normal_Decoder
        self.detector = None
        self.x_dim = None
        self.beta = beta
        self.normal_score = normal_score

    def fit(self, ts: np.array):
        """
        RDSMM take a T x N time series as input
        """
        def train():
            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                X = data.to(self.device)
                optimizer.zero_grad()
                kld_loss, nll_loss = self.detector(X)
                loss = self.beta*kld_loss + nll_loss
                train_loss += loss.item() / len(train_loader.dataset)
                loss.backward()
                optimizer.step()
            return train_loss

        def valid():
            valid_loss = 0
            valid_nll = 0
            for batch_idx, data in enumerate(valid_loader):
                with torch.no_grad():
                    X = data.to(self.device)
                    kld_loss, nll_loss = self.detector(X)
                    loss = kld_loss + nll_loss
                    valid_loss += loss.item() / len(valid_loader.dataset)
                    valid_nll += nll_loss.item() / len(valid_loader.dataset)
            return valid_nll, valid_loss

        ts = ts.astype(np.float32)
        self.x_dim = ts.shape[1]
        ts_chunks = windowing(ts, window_size=self.sequence_length)
        num_train = int(ts_chunks.shape[0] * 0.95)
        train_chunks = ts_chunks[:num_train]
        valid_chunks = ts_chunks[num_train:]
        train_loader = DataLoader(train_chunks, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_chunks, batch_size=self.batch_size, shuffle=True)
        self.detector = self.model_class(self.x_dim, self.h_dim, self.z_dim, device=self.device,
                                         decoder=self.decoder_class, IWAE=self.IWAE).to(self.device)
        self.early_stopping = EarlyStopping(patience=self.patience,
                                            verbose=self.verbose,
                                            checkpoint_dir=self.checkpoint_dir)
        optimizer = torch.optim.Adam(self.detector.parameters(), lr=self.lr)

        for epoch in tqdm_notebook(range(self.num_epochs)):
            train_loss = train()
            valid_nll, valid_loss = valid()
            if self.verbose:
                if epoch % 1 == 0:
                    print(f">> epoch: {epoch}, train_loss:{train_loss:.3f}"
                          f" valid_loss: {valid_loss}, valid_nll:{valid_nll}")
            self.early_stopping(valid_nll, self.detector)
            if self.early_stopping.early_stop:
                self.detector.load_state_dict(pickle.load(open(self.checkpoint_dir, 'rb')))
                break

    def predict(self, ts: np.array, aggregate=True):
        ts = ts.astype(np.float32)
        length_padding = ts.shape[0] % self.sequence_length
        self.x_dim = ts.shape[1]
        ts_ready = windowing(ts, window_size=self.sequence_length, step=self.sequence_length)
        if length_padding > 0:
            ts_last = ts[-self.sequence_length:, :][None, :, :]
            ts_ready = np.concatenate([ts_ready, ts_last], axis=0)
        ts_ready_torch = torch.FloatTensor(ts_ready).to(self.device)
        scores = self.detector.scoring(ts_ready_torch,self.normal_score).data.cpu().numpy()
        del ts_ready_torch
        scores = np.concatenate([scores[:-1].reshape(-1, self.x_dim), scores[-1][-length_padding:]], axis=0)
        if aggregate:
            return scores.sum(axis=-1)[:,None]
        else:
            return scores

    def density_estimation(self, ts: np.array):
        ts = ts.astype(np.float32)
        length_padding = ts.shape[0] % self.sequence_length
        self.x_dim = ts.shape[1]
        ts_ready = windowing(ts, window_size=self.sequence_length, step=self.sequence_length)
        if length_padding > 0:
            ts_last = ts[-self.sequence_length:, :][None, :, :]
            ts_ready = np.concatenate([ts_ready, ts_last], axis=0)
        ts_ready_torch = torch.FloatTensor(ts_ready).to(self.device)
        x_rvs = self.detector.density_estimation(ts_ready_torch)
        if self.normal_score:
            x_rvs = StudentT_Normal_converter(x_rvs, length_padding)
            return x_rvs
        x_rvs = self.detector.decoder.converter(x_rvs, length_padding)
        del ts_ready_torch
        return x_rvs

class Encoder(nn.Module):
    def __init__(self, h_dim, z_dim, device=None):
        super(Encoder, self).__init__()
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.enc = nn.Sequential(
            nn.Linear(h_dim * 3, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(nn.Linear(h_dim, z_dim),
                                     nn.Softplus())

    def forward(self, xh,zh, h):
        zh = self.enc(torch.cat([xh, zh, h], -1))
        z_mu = self.enc_mean(zh)
        z_std = self.enc_std(zh)
        return Normal(z_mu, z_std)


class Normal_Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, bound):
        super(Normal_Decoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim))
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        self.bound = bound

    def forward(self, zh, h):
        xh = self.dec(torch.cat([zh, h], -1))
        x_mu = self.dec_mean(xh)
        x_std = self.dec_std(xh) + self.bound
        return Normal(x_mu, x_std)


class StudentT_Decoder(nn.Module):
    def __init__(self, x_dim, h_dim, bound):
        super(StudentT_Decoder, self).__init__()
        self.dec = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())
        self.dec_mean = nn.Sequential(
            nn.Linear(h_dim, x_dim))
        self.dec_std = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        self.dec_df = nn.Sequential(
            nn.Linear(h_dim, x_dim),
            nn.Softplus())
        self.bound = bound
        self.converter = StudentT_converter

    def forward(self, zh, h, normal=False):
        StudentT.arg_constraints['df'].lower_bound = 1
        xh = self.dec(torch.cat([zh, h], -1))
        x_mu = self.dec_mean(xh)
        x_std = self.dec_std(xh)
        x_df = self.dec_df(xh) + self.bound
        if normal:
            return Normal(x_mu, x_std)
        return StudentT(x_df, x_mu, x_std)




class Recurrence(nn.Module):
    def __init__(self, h_dim):
        super(Recurrence, self).__init__()
        self.h_dim = h_dim
        self.rnn = nn.GRUCell(h_dim, h_dim)

    def forward(self, xh, h):
        # warp the RNN to support multiple sampling objective
        hshape = h.shape
        h = self.rnn(xh.view(-1, self.h_dim), h.view(-1, self.h_dim))
        return h.view(hshape)


class Prior(nn.Module):
    def __init__(self, h_dim, z_dim):
        super(Prior, self).__init__()
        self.prior = nn.Sequential(
            nn.Linear(h_dim*2, h_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.Softplus())

    def forward(self, zh, h):
        ph = self.prior(torch.cat([zh, h], -1))
        p_mu = self.prior_mean(ph)
        p_std = self.prior_std(ph)
        return Normal(p_mu, p_std)


class RDSSM_module(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim,encoder=Encoder, decoder=Normal_Decoder, IWAE=5, device=None):
        super(RDSSM_module, self).__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.IWAE = IWAE
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.bound = torch.FloatTensor([1]).to(device)
        self.encoder = encoder(h_dim, z_dim).to(device)
        self.decoder = decoder(x_dim, h_dim,self.bound).to(device)
        self.prior = Prior(h_dim, z_dim)
        self.recurrence = Recurrence(h_dim)
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU())
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ReLU())

    def initial_states(self, X):
        # initialize model,
        # return h1, z0

        x = X[:, 0, :].unsqueeze(1).repeat(1, self.IWAE, 1)
        z = torch.zeros(X.shape[0], self.IWAE, self.z_dim).to(self.device)
        h = torch.zeros(X.shape[0], self.IWAE, self.h_dim).to(self.device)
        xh = self.phi_x(x)

        # (x0, h0) -> (h1)
        h = self.recurrence(xh, h)
        return h, z

    def infer_latents(self, X):
        """
        X : B x T x N
        1. extent to B T x IWAE x N matrix
        hs : B x IWAE x H matrix
        zs :
        reutrn :
        """
        h, z = self.initial_states(X)
        zs, hs, kls = [], [], []
        for t in range(X.shape[1]):
            hs.append(h)
            x = X[:, t, :].unsqueeze(1).repeat(1, self.IWAE, 1)
            xh = self.phi_x(x)
            zh = self.phi_z(z) # this is z0
            pz_rv = self.prior(zh, h)       # p(z_t|z_t-1, h_t)
            qz_rv = self.encoder(xh, zh, h) # q(z_t|x_t, z_t-1, h_t)
            z = qz_rv.rsample()
            zs.append(z)
            kl = kl_divergence(qz_rv, pz_rv)
            kls.append(kl)
            h = self.recurrence(xh, h) # p(h_t+1|x_t, h_t)

        hs = torch.stack(hs, dim = 1)
        zs = torch.stack(zs, dim = 1)
        kls = torch.stack(kls, dim=1)
        return zs, hs, kls

    def generation(self, zs, hs, X, normal=False):
        X = X.unsqueeze(2).repeat(1, 1, self.IWAE, 1)
        zhs = self.phi_z(zs)
        x_rvs = self.decoder(zhs, hs, normal=normal)
        log_probs = x_rvs.log_prob(X)
        return x_rvs, log_probs


    def forward(self, X, normal=False):
        """
        X: Batch_size x T x N dimension
        """
        self.X_shape = X.shape
        self.batch_size = X.shape[0]
        zs, hs, kls = self.infer_latents(X)
        x_rvs, log_probs = self.generation(zs, hs, X, normal)
        nll_loss = -log_probs.sum() / self.batch_size / self.IWAE
        kld_loss = kls.sum() / self.batch_size / self.IWAE
        return kld_loss, nll_loss

    def scoring(self, X, normal=False):
        zs, hs, kls = self.infer_latents(X)
        x_rvs, log_probs = self.generation(zs, hs, X, normal=normal)
        scores = -log_probs.mean(dim=2)
        return scores

    def density_estimation(self, X):
        with torch.no_grad():
            zs, hs, kls = self.infer_latents(X)
            x_rvs, log_probs = self.generation(zs, hs, X)
        return x_rvs
