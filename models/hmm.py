import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy as sp
import torch.distributions as tD
from .nets import MLP, View, kaiming_init, reparametrize

def sticky_transitions(num_states, stickiness=0.95):
    P = stickiness * torch.eye(num_states) 
    P += (1 - stickiness) / (num_states - 1) * (1 - torch.eye(num_states))

    return P

class HMM(nn.Module):
    def __init__(self, n_class, lags,x_dim, hidden_dim, mode="mle_scaled:H",num_layers=3) -> None:
        super().__init__()
        self.mode,self.feat = mode.split(":")
        if self.mode == "em":
            self.register_buffer('log_A',torch.randn(n_class, n_class))
            self.register_buffer('log_pi',torch.randn(n_class))
        elif self.mode == "mle_scaled" or self.mode == "mle":
            self.log_A = nn.Parameter(torch.randn(n_class, n_class))
            
            self.log_pi = nn.Parameter(torch.randn(n_class))
        else:
            raise ValueError("mode must be em or mle_scaled or mle, but got {}".format(self.mode))
        self.n_class = n_class
        self.x_dim = x_dim
        self.lags = lags
        if self.feat == "Ht":
            self.trans = MLP(input_dim=(lags+1)*x_dim, hidden_dim=hidden_dim,
                            output_dim=n_class*2*x_dim, num_layers=num_layers)
        elif self.feat == "H":
            self.trans = MLP(input_dim=(lags)*x_dim, hidden_dim=hidden_dim,
                            output_dim=n_class*2*x_dim, num_layers=num_layers)
        else:
            raise ValueError("feat must be Ht or H")

    def forward_log(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        # length = lags_and_length - self.lags
        log_alpha = torch.zeros(batch_size, length, self.n_class,device=logp_x_c.device)
        log_A = torch.log_softmax(self.log_A,dim=1)
        log_pi = torch.log_softmax(self.log_pi,dim=0)
        for t in range(length):
            if t == 0:
                log_alpha_t = logp_x_c[:, t] + log_pi
            else:
                log_alpha_t = logp_x_c[:, t] + torch.logsumexp(
                    log_alpha[:, t-1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1)
            log_alpha[:, t] = log_alpha_t
        logp_x = torch.logsumexp(log_alpha[:, -1],dim=-1)
        # logp_x = torch.sum(log_scalers, dim=-1)
        return logp_x
    
    def forward_backward_log(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        # length = lags_and_length - self.lags
        log_alpha = torch.zeros(batch_size, length, self.n_class,device=logp_x_c.device)
        log_beta = torch.zeros(batch_size, length, self.n_class,device=logp_x_c.device)
        log_scalers = torch.zeros(batch_size, length,device=logp_x_c.device)
        log_A = torch.log_softmax(self.log_A,dim=1)
        log_pi = torch.log_softmax(self.log_pi,dim=0)
        for t in range(length):
            if t == 0:
                log_alpha_t = logp_x_c[:, t] + log_pi
            else:
                log_alpha_t = logp_x_c[:, t] + torch.logsumexp(
                    log_alpha[:, t-1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1)
            log_scalers[:, t] = torch.logsumexp(log_alpha_t, dim=-1)
            log_alpha[:, t] = log_alpha_t - log_scalers[:, t].unsqueeze(-1)
        log_beta[:, -1] = torch.zeros(batch_size, self.n_class,device=logp_x_c.device)
        for t in range(length-2,-1,-1):
            log_beta_t = torch.logsumexp(log_beta[:, t+1].unsqueeze(-1) + log_A.unsqueeze(0) + logp_x_c[:, t+1].unsqueeze(1), dim=-1)
            log_beta[:, t] = log_beta_t - log_scalers[:, t].unsqueeze(-1)
        log_gamma = log_alpha + log_beta
        # logp_x = torch.logsumexp(log_alpha[:, -1],dim=-1)
        logp_x = torch.sum(log_scalers, dim=-1)
        return log_alpha, log_beta, log_scalers, log_gamma, logp_x

    def viterbi_algm(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        log_delta = torch.zeros(batch_size, length, self.n_class,device=logp_x_c.device)
        psi = torch.zeros(batch_size, length, self.n_class, dtype=torch.long,device=logp_x_c.device)
        
        log_A = torch.log_softmax(self.log_A,dim=1)
        log_pi = torch.log_softmax(self.log_pi,dim=0)
        # log_A = torch.log_softmax(self.log_A,dim=1)
        # log_pi = torch.log_softmax(self.log_pi,dim=0)
        for t in range(length):
            if t == 0:
                log_delta[:, t] = logp_x_c[:, t] + log_pi
            else:
                max_val, max_arg = torch.max(
                    log_delta[:, t-1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1)
                log_delta[:, t] = max_val + logp_x_c[:, t]
                psi[:, t] = max_arg
        # logp_x = torch.max(log_delta[:, -1])
        c = torch.zeros(batch_size, length, dtype=torch.long,device=logp_x_c.device)
        c[:, -1] = torch.argmax(log_delta[:, -1], dim=-1)
        for t in range(length-2, -1, -1):
            c[:, t] = psi[:,t+1].gather(1, c[:, t+1].unsqueeze(1)).squeeze()
        return c #, logp_x

    def forward(self, x):
        batch_size, lags_and_length, _ = x.shape
        length = lags_and_length - self.lags
        # x_H = (batch_size, length, (lags) * x_dim)
        x_H = x.unfold(dimension=1, size=self.lags+1, step=1).transpose(-2, -1)
        if self.feat == "H":
            x_H = x_H[...,:self.lags,:].reshape(batch_size, length, -1)
        elif self.feat == "Ht":
            x_H = x_H.reshape(batch_size, length, -1)

        # (batch_size, length, n_class, x_dim)
        out = self.trans(x_H).reshape(batch_size, length, self.n_class, 2 * self.x_dim)
        mus, logvars = out[...,:self.x_dim], out[..., self.x_dim:]
        dist = tD.Normal(mus, torch.exp(logvars / 2))
        logp_x_c = dist.log_prob(x[:, self.lags:].unsqueeze(2)).sum(-1)  # (batch_size, length, n_class)
        if self.mode == "em" or self.mode == "mle_scaled":
            log_alpha, log_beta, log_scalers, log_gamma, logp_x = self.forward_backward_log(logp_x_c)
            if self.mode == "em":
                batch_normalizing_factor = torch.log(torch.tensor(batch_size,device=logp_x_c.device))
                expected_log_pi = log_gamma[:,0, :] - log_gamma[:,0, :].logsumexp(dim=-1).unsqueeze(-1)
                expected_log_pi = expected_log_pi.logsumexp(dim=0) - batch_normalizing_factor
                log_A = torch.log_softmax(self.log_A,dim=1)
                log_xi = torch.zeros(batch_size, length-1, self.n_class, self.n_class,device=logp_x_c.device)
                for t in range(length-1): # B,Ct,1 B,1,Ct+1 1,Ct,Ct+1 B,1,Ct+1,  
                    log_xi_t = log_alpha[:, t].unsqueeze(-1) + log_beta[:, t+1].unsqueeze(1) + log_A.unsqueeze(0) + logp_x_c[:, t+1].unsqueeze(1)
                    log_xi_scalers = torch.logsumexp(log_xi_t, dim=(1,2),keepdim=True)
                    log_xi[:, t] = log_xi_t - log_xi_scalers
                expected_log_A = torch.logsumexp(log_xi, dim=1) - torch.logsumexp(log_xi, dim=(1,3)).unsqueeze(-1)
                expected_log_A = expected_log_A.logsumexp(dim=0) - batch_normalizing_factor
                self.log_A = expected_log_A.detach()
                self.log_pi = expected_log_pi.detach()
        elif self.mode == "mle":
            logp_x = self.forward_log(logp_x_c)
        
        c_est = self.viterbi_algm(logp_x_c)
        return logp_x, c_est


class AbsHMM(nn.Module):
    def __init__(self, n_class) -> None:
        super().__init__()
        # A_init = torch.exp(torch.randn(n_class, n_class)) + n_class*torch.eye(n_class)
        # A_init = A_init / A_init.sum(dim=1, keepdim=True)
        # A = sticky_transitions(n_class,0.9)
        # self.log_A = nn.Parameter(torch.log(A))
        
        # pi = torch.ones(n_class) / n_class
        # self.log_pi = nn.Parameter(torch.log(pi))
        self.log_A = nn.Parameter(torch.randn(n_class, n_class))
        self.log_pi = nn.Parameter(torch.randn(n_class))
        self.n_class = n_class

    def forward_log(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        # length = lags_and_length - self.lags
        log_alpha = torch.zeros(batch_size, length, self.n_class,device=logp_x_c.device)
        log_A = torch.log_softmax(self.log_A,dim=1)
        log_pi = torch.log_softmax(self.log_pi,dim=0)
        for t in range(length):
            if t == 0:
                log_alpha_t = logp_x_c[:, t] + log_pi
            else:
                log_alpha_t = logp_x_c[:, t] + torch.logsumexp(
                    log_alpha[:, t-1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1)
            log_alpha[:, t] = log_alpha_t
        logp_x = torch.logsumexp(log_alpha[:, -1],dim=-1)
        # logp_x = torch.sum(log_scalers, dim=-1)
        return logp_x
    
    def forward_backward_log(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        # length = lags_and_length - self.lags
        log_alpha = torch.zeros(batch_size, length, self.n_class,device=logp_x_c.device)
        log_beta = torch.zeros(batch_size, length, self.n_class,device=logp_x_c.device)
        log_scalers = torch.zeros(batch_size, length,device=logp_x_c.device)
        log_A = torch.log_softmax(self.log_A,dim=1)
        log_pi = torch.log_softmax(self.log_pi,dim=0)
        for t in range(length):
            if t == 0:
                log_alpha_t = logp_x_c[:, t] + log_pi
            else:
                log_alpha_t = logp_x_c[:, t] + torch.logsumexp(
                    log_alpha[:, t-1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1)
            log_scalers[:, t] = torch.logsumexp(log_alpha_t, dim=-1)
            log_alpha[:, t] = log_alpha_t - log_scalers[:, t].unsqueeze(-1)
        log_beta[:, -1] = torch.zeros(batch_size, self.n_class,device=logp_x_c.device)
        for t in range(length-2,-1,-1):
            log_beta_t = torch.logsumexp(log_beta[:, t+1].unsqueeze(-1) + log_A.unsqueeze(0) + logp_x_c[:, t+1].unsqueeze(1), dim=-1)
            log_beta[:, t] = log_beta_t - log_scalers[:, t].unsqueeze(-1)
        log_gamma = log_alpha + log_beta
        # logp_x = torch.logsumexp(log_alpha[:, -1],dim=-1)
        logp_x = torch.sum(log_scalers, dim=-1)
        return log_alpha, log_beta, log_scalers, log_gamma, logp_x

    def viterbi_algm(self, logp_x_c):
        batch_size, length, n_class = logp_x_c.shape
        log_delta = torch.zeros(batch_size, length, self.n_class,device=logp_x_c.device)
        psi = torch.zeros(batch_size, length, self.n_class, dtype=torch.long,device=logp_x_c.device)
        
        log_A = torch.log_softmax(self.log_A,dim=1)
        log_pi = torch.log_softmax(self.log_pi,dim=0)
        # log_A = torch.log_softmax(self.log_A,dim=1)
        # log_pi = torch.log_softmax(self.log_pi,dim=0)
        for t in range(length):
            if t == 0:
                log_delta[:, t] = logp_x_c[:, t] + log_pi
            else:
                max_val, max_arg = torch.max(
                    log_delta[:, t-1].unsqueeze(-1) + log_A.unsqueeze(0), dim=1)
                log_delta[:, t] = max_val + logp_x_c[:, t]
                psi[:, t] = max_arg
        # logp_x = torch.max(log_delta[:, -1])
        c = torch.zeros(batch_size, length, dtype=torch.long,device=logp_x_c.device)
        c[:, -1] = torch.argmax(log_delta[:, -1], dim=-1)
        for t in range(length-2, -1, -1):
            c[:, t] = psi[:,t+1].gather(1, c[:, t+1].unsqueeze(1)).squeeze()
        return c #, logp_x

class CartPoleHMMz(AbsHMM):
    def __init__(self, n_class, lags,z_dim, hidden_dim):
        super().__init__(n_class)
        self.lags = lags
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.action_num = 2
        self.action_dim = 2
        self.action_case = 1
        num_layers = 1
        if self.action_case == 0:
            # (1) use mask to control the action
            self.trans =nn.ModuleList(
                MLP(input_dim=z_dim*self.lags, 
                    hidden_dim=hidden_dim, 
                    output_dim=n_class*z_dim*2,
                    num_layers=num_layers) 
                for _ in range(self.action_num))
        elif self.action_case == 1:
            # (2) use embedding to control the action
            self.a_embeddings = nn.Embedding(self.action_num, self.action_dim)
            self.trans =MLP(input_dim=(z_dim + self.action_dim)*self.lags, 
                    hidden_dim=hidden_dim, 
                    output_dim=n_class*z_dim*2,
                    num_layers=num_layers)
        elif self.action_case == 2:
            # (3) ignore action
            self.trans =MLP(input_dim=z_dim*self.lags, 
                    hidden_dim=hidden_dim, 
                    output_dim=n_class*z_dim*2,
                    num_layers=num_layers)
        
        # self.rnn = nn.LSTM(z_dim, z_dim, batch_first=True,bidirectional=True, num_layers=2)
    def forward(self, z, a):
        batch_size, lags_and_length, _ = z.shape # batch_size, lags_and_length, z_dim
        
        length = lags_and_length - self.lags
        
        # z_rnn = self.rnn(z)[0]
        z_rnn = z

        
        if self.action_case == 0:
            # (1) use mask to control the action
            z_H = z_rnn.unfold(dimension=1, size=self.lags+1, step=1).transpose(-2, -1)
            z_H = z_H[...,:self.lags,:].reshape(batch_size, length, -1)
            z_mask = F.one_hot(a[...,self.lags-1:-1], num_classes=self.action_num)
            out_action = [self.trans[i](z_H).reshape(batch_size, -1, self.n_class, 2 * self.z_dim) for i in range(self.action_num)]
            out = torch.stack(out_action, dim=2)
            out = (out * z_mask.unsqueeze(-1).unsqueeze(-1)).sum(2)
        elif self.action_case == 1:
            # (2) use embedding to control the action
            a_emb = self.a_embeddings(a)
            # a_emb = a_emb[...,self.lags-1:-1,:].reshape(batch_size, length, -1)
            z_a = torch.cat([z_rnn, a_emb], dim=-1)
            z_a_H = z_a.unfold(dimension=1, size=self.lags+1, step=1).transpose(-2, -1) # batch_size, length, lags+1, z_dim
            z_a_H = z_a_H[...,:self.lags,:].reshape(batch_size, length, -1) # batch_size, length, z_dim*(lags)
            
            out = self.trans(z_a_H).reshape(batch_size, -1, self.n_class, 2 * self.z_dim)
        elif self.action_case == 2:
            # (3) ignore action
            z_H = z_rnn.unfold(dimension=1, size=self.lags+1, step=1).transpose(-2, -1)
            z_H = z_H[...,:self.lags,:].reshape(batch_size, length, -1)
            out = self.trans(z_H).reshape(batch_size, -1, self.n_class, 2 * self.z_dim)
        
        mus, logvars = out[...,:self.z_dim], out[..., self.z_dim:]
        dist = tD.Normal(mus, torch.exp(logvars / 2))
        logp_x_c = dist.log_prob(z[:, self.lags:,:].unsqueeze(2)).sum(-1)  # (batch_size, length, n_class)
        p_x_H_dist = tD.Normal(torch.zeros_like(z[:, :self.lags]), torch.ones_like(z[:, :self.lags]))
        logp_x_H = p_x_H_dist.log_prob(z[:, :self.lags]).sum(dim=[-1, -2])
        log_alpha, log_beta, log_scalers, log_gamma, logp_x = self.forward_backward_log(logp_x_c)
        # logp_x = self.forward_log(logp_x_c)
        c_est = self.viterbi_algm(logp_x_c)
        return logp_x+logp_x_H, c_est


class CartPoleHMMx(AbsHMM):
    def __init__(self, n_class, lags, z_dim, hidden_dim, action_num=2,action_embedding_dim=2,nc=1,rnn_bidirectional=False):
        super().__init__(n_class)
        self.action_num = action_num
        self.action_embedding_dim = action_embedding_dim
        self.lags = lags
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.nc = nc
        self.encoder=nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 64, 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, hidden_dim, 4, 1),          # B, x_dim,  1,  1
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),
            View((-1, hidden_dim)),             # B, hidden_dim
            nn.Linear(hidden_dim, hidden_dim*2),        # B, x_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),               # B, hidden_dim
            View((-1, hidden_dim, 1, 1)),               # B, hidden_dim,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(hidden_dim, 64, 4),      # B,  64,  4,  4
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 64, 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 128, 128
        )
        input_dim = (lags)*hidden_dim*2 + action_embedding_dim if rnn_bidirectional else (lags)*hidden_dim + action_embedding_dim
        self.trans = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=n_class*2*hidden_dim, num_layers=3)
        self.a_embedding = nn.Embedding(action_num, action_embedding_dim)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True,bidirectional=rnn_bidirectional)
        self.weight_init()
        

    def weight_init(self):
        for block in self._modules:
            if block in ['encoder','decoder']:
                for m in self._modules[block]:
                    kaiming_init(m)
            elif isinstance(block,nn.Module):
                kaiming_init(block)
    def forward(self, x, a):
        batch_size, lags_and_length, nc, h, w = x.shape
        length = lags_and_length - self.lags
        
        x_flat = x.view(-1, nc, h, w)
        distributions = self.encoder(x_flat)
        mu = distributions[..., :self.hidden_dim]
        logvar = distributions[..., self.hidden_dim:]
        x_feats = reparametrize(mu, logvar)
        x_recon = self.decoder(x_feats)
        x_recon = x_recon.view(batch_size, lags_and_length, nc, h, w)
        x_feats = x_feats.view(batch_size, lags_and_length, -1)
        x_rnn = self.rnn(x_feats)[0]
        # x_H = (batch_size, length, lags+1, x_dim)
        x_H = x_rnn.unfold(dimension=1, size=self.lags+1, step=1).transpose(-2, -1)
        x_H = x_H[...,:self.lags,:].reshape(batch_size, length, -1) # (batch_size, length, lags * x_dim)

        
        # mask based on action
        ## (batch_size, length, n_class, x_dim)
        # out_action = [self.decoder[i](x_H).reshape(batch_size, length, self.n_class, 2 * self.x_dim) for i in range(self.action_num)]
        # a_unfold = a[:,self.lags-1:-1] # (batch_size, length)
        # mask = F.one_hot(a_unfold, num_classes=2)
        # out = torch.stack(out_action, dim=2) # (batch_size, length, action_num, n_class, 2 * x_dim)
        # out = (out * mask.unsqueeze(-1).unsqueeze(-1)).sum(2) # (batch_size, length, n_class, 2 * x_dim)
        
        # feat based on action
        a_unfold = a[:,self.lags:] # (batch_size, length)
        a_emb = self.a_embedding(a_unfold) # (batch_size, length, 1)
        x_H_a = torch.cat([x_H, a_emb], dim=-1)
        out = self.trans(x_H_a).reshape(batch_size, length, self.n_class, 2 * self.hidden_dim)
        
        mus, logvars = out[...,:self.hidden_dim], out[..., self.hidden_dim:]
        dist = tD.Normal(mus, torch.exp(logvars / 2))
        logp_x_c = dist.log_prob(x_feats[:, self.lags:].unsqueeze(2)).sum(-1)  # (batch_size, length, n_class)
        log_alpha, log_beta, log_scalers, log_gamma, logp_x = self.forward_backward_log(logp_x_c)
        c_est = self.viterbi_algm(logp_x_c)
        return logp_x, c_est, x_recon


class MoSeqHMMz(AbsHMM):
    def __init__(self, n_class, lags,z_dim, hidden_dim):
        super().__init__(n_class)
        self.lags = lags
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        num_layers = 1
        self.trans =MLP(input_dim=z_dim*self.lags, 
                    hidden_dim=hidden_dim, 
                    output_dim=n_class*z_dim*2,
                    num_layers=num_layers)
        
        # self.rnn = nn.LSTM(z_dim, z_dim, batch_first=True,bidirectional=True, num_layers=2)
    def forward(self, z):
        batch_size, lags_and_length, _ = z.shape # batch_size, lags_and_length, z_dim
        
        length = lags_and_length - self.lags
        
        # z_rnn = self.rnn(z)[0]
        z_rnn = z

        z_H = z_rnn.unfold(dimension=1, size=self.lags+1, step=1).transpose(-2, -1)
        z_H = z_H[...,:self.lags,:].reshape(batch_size, length, -1)
        out = self.trans(z_H).reshape(batch_size, -1, self.n_class, 2 * self.z_dim)
        
        mus, logvars = out[...,:self.z_dim], out[..., self.z_dim:]
        dist = tD.Normal(mus, torch.exp(logvars / 2))
        logp_x_c = dist.log_prob(z[:, self.lags:,:].unsqueeze(2)).sum(-1)  # (batch_size, length, n_class)
        log_alpha, log_beta, log_scalers, log_gamma, logp_x = self.forward_backward_log(logp_x_c)
        # logp_x = self.forward_log(logp_x_c)
        c_est = self.viterbi_algm(logp_x_c)
        return logp_x, c_est
    