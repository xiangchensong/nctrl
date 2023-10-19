import torch
import pytorch_lightning as pl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from .nets import MLP, BetaVAE_MLP, NPTransitionPrior, NPChangeTransitionPrior
from .hmm import HMM
from .metrics.correlation import compute_mcc, compute_acc


class TDRL(pl.LightningModule):
    def __init__(
            self,
            x_dim,
            z_dim,
            lags,
            n_class,
            hidden_dim=128,
            lr=1e-4,
            beta=0.0025,
            gamma=0.0075,
            correlation='Pearson'):
        '''Nonlinear ICA for nonparametric stationary processes'''
        super().__init__()
        self.save_hyperparameters()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.lags = lags
        self.n_class = n_class
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.correlation = correlation
        self.net = BetaVAE_MLP(input_dim=x_dim,
                               z_dim=z_dim,
                               hidden_dim=hidden_dim)

        # Initialize transition prior
        self.transition_prior = NPTransitionPrior(lags=lags,
                                                  latent_size=z_dim,
                                                  num_layers=3,
                                                  hidden_dim=hidden_dim)
        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))
        self.best_mcc = -np.inf
        self.validation_step_outputs = []

    @property
    def base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def reconstruction_loss(self, x, x_recon, distribution='gaussian'):
        batch_size = x.shape[0]
        assert batch_size != 0

        if distribution == 'bernoulli':
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'gaussian':
            recon_loss = F.mse_loss(
                x_recon, x, size_average=False).div(batch_size)

        elif distribution == 'sigmoid_gaussian':
            x_recon = F.sigmoid(x_recon)
            recon_loss = F.mse_loss(
                x_recon, x, size_average=False).div(batch_size)

        return recon_loss

    def training_step(self, batch, batch_idx):
        # (batch_size, lags+length, x_dim) (batch_size, lags+length, z_dim) (batch_size, lags+length)
        x, _, _ = batch
        batch_size, lags_and_length, _ = x.shape
        x_recon, mus, logvars, z_est = self.net(x)
        # recon_loss = self.reconstruction_loss(x, x_recon)
        recon_loss = self.reconstruction_loss(x[:, :self.lags], x_recon[:, :self.lags]) + \
            (self.reconstruction_loss(
                x[:, self.lags:], x_recon[:, self.lags:]))/(lags_and_length-self.lags)

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lags:]
        residuals, logabsdet = self.transition_prior(z_est)
        log_pz_laplace = torch.sum(self.base_dist.log_prob(
            residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (
            torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (lags_and_length - self.lags)
        kld_laplace = kld_laplace.mean()

        # VAE training
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace

        self.log_dict({'train/elbo_loss': loss,
                       'train/recon_loss': recon_loss,
                       'train/kld_normal': kld_normal,
                       'train/kld_laplace': kld_laplace})
        return loss

    def valid(self, batch, batch_idx):
        # (batch_size, lags+length, x_dim) (batch_size, lags+length, z_dim) (batch_size, lags+length)
        x, z, _ = batch
        batch_size, lags_and_length, _ = x.shape
        x_recon, mus, logvars, z_est = self.net(x)
        # recon_loss = self.reconstruction_loss(x, x_recon)
        recon_loss = self.reconstruction_loss(
            x[:, :self.lags], x_recon[:, :self.lags]) + self.reconstruction_loss(x[:, self.lags:], x_recon[:, self.lags:])

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lags:]
        residuals, logabsdet = self.transition_prior(z_est)
        log_pz_laplace = torch.sum(self.base_dist.log_prob(
            residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (
            torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace) / (lags_and_length - self.lags)
        kld_laplace = kld_laplace.mean()

        # VAE training
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace

        # Compute Mean Correlation Coefficient (MCC)
        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = z.view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        return loss, mcc, recon_loss, kld_normal, kld_laplace

    def validation_step(self, batch, batch_idx):
        loss, mcc, recon_loss, kld_normal, kld_laplace = self.valid(
            batch, batch_idx)
        self.validation_step_outputs.append(
            {'loss': loss, 'mcc': mcc, 'recon_loss': recon_loss, 'kld_normal': kld_normal, 'kld_laplace': kld_laplace})

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mcc = np.stack([x['mcc'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['recon_loss'] for x in outputs]).mean()
        avg_kld_normal = torch.stack([x['kld_normal'] for x in outputs]).mean()
        avg_kld_laplace = torch.stack(
            [x['kld_laplace'] for x in outputs]).mean()
        if avg_mcc > self.best_mcc:
            self.best_mcc = avg_mcc
        self.log_dict({'val/mcc': avg_mcc,
                       'val/best_mcc': self.best_mcc}, prog_bar=True)
        self.log_dict({'val/loss': avg_loss,
                       'val/recon_loss': avg_recon_loss,
                       'val/kld_normal': avg_kld_normal,
                       'val/kld_laplace': avg_kld_laplace})
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters(
        )), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []


class CTDRL(pl.LightningModule):
    def __init__(
            self,
            x_dim,
            z_dim,
            lags,
            n_class,
            hidden_dim=128,
            embedding_dim=8,
            lr=1e-4,
            beta=0.0025,
            gamma=0.0075,
            correlation='Pearson',
            **kwargs):
        """
        Reference implementation of TDRL model with changing causal dynamic.
        Upper bound of the NCTRL.
        """
        super().__init__()
        self.save_hyperparameters()
        self.A = None
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.lags = lags
        self.n_class = n_class
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.beta = beta
        self.gamma = gamma
        self.correlation = correlation

        self.c_embeddings = nn.Embedding(n_class, embedding_dim)
        self.net = BetaVAE_MLP(
            input_dim=x_dim, z_dim=z_dim, hidden_dim=hidden_dim)
        self.transition_prior = NPChangeTransitionPrior(lags=lags,
                                                        latent_size=z_dim,
                                                        embedding_dim=embedding_dim,
                                                        num_layers=3,
                                                        hidden_dim=hidden_dim)

        # base distribution for calculation of log prob under the model
        self.register_buffer('base_dist_mean', torch.zeros(self.z_dim))
        self.register_buffer('base_dist_var', torch.eye(self.z_dim))
        self.best_mcc = -np.inf
        self.validation_step_outputs = []

    @property
    def base_dist(self):
        # Noise density function
        return D.MultivariateNormal(self.base_dist_mean, self.base_dist_var)

    def reconstruction_loss(self, x, x_recon):
        batch_size = x.shape[0]
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
        return recon_loss

    def training_step(self, batch, batch_idx):
        x, _, c = batch
        _, lags_and_length, _ = x.shape

        x_recon, mus, logvars, z_est = self.net(x)
        # (batch_size, lags+length, embedding_dim)
        embeddings = self.c_embeddings(c)

        # recon_loss = self.reconstruction_loss(x, x_recon)
        recon_loss = self.reconstruction_loss(x[:, :self.lags], x_recon[:, :self.lags]) + \
            (self.reconstruction_loss(
                x[:, self.lags:], x_recon[:, self.lags:]))/(lags_and_length-self.lags)

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lags:]
        residuals, logabsdet = self.transition_prior(z_est, embeddings)
        log_pz_laplace = torch.sum(
            self.base_dist.log_prob(residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1),
                       dim=-1) - log_pz_laplace) / (lags_and_length-self.lags)
        kld_laplace = kld_laplace.mean()

        # VAE training
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace

        self.log_dict({"train/elbo_loss": loss, "train/recon_loss": recon_loss,
                      "train/kld_normal": kld_normal, "train/kld_laplace": kld_laplace})
        return loss

    def validation_step(self, batch, batch_idx):
        x, z, c = batch
        _, lags_and_length, _ = x.shape

        x_recon, mus, logvars, z_est = self.net(x)
        # (batch_size, lags+length, embedding_dim)
        embeddings = self.c_embeddings(c)

        # recon_loss = self.reconstruction_loss(x, x_recon)
        recon_loss = self.reconstruction_loss(x[:, :self.lags], x_recon[:, :self.lags]) + \
            (self.reconstruction_loss(
                x[:, self.lags:], x_recon[:, self.lags:]))/(lags_and_length-self.lags)

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lags:]
        residuals, logabsdet = self.transition_prior(z_est, embeddings)
        log_pz_laplace = torch.sum(
            self.base_dist.log_prob(residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1),
                       dim=-1) - log_pz_laplace) / (lags_and_length-self.lags)
        kld_laplace = kld_laplace.mean()

        # VAE training
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace

        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = z.view(-1, self.z_dim).T.detach().cpu().numpy()
        mcc = compute_mcc(zt_recon, zt_true, self.correlation)
        acc, _ = compute_acc(c.cpu().numpy(), c.cpu().numpy(),
                             C=self.n_class)
        self.validation_step_outputs.append(
            {'loss': loss, 'mcc': mcc, 'acc': acc, 'recon_loss': recon_loss, 'kld_normal': kld_normal, 'kld_laplace': kld_laplace})

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_mcc = np.stack([x['mcc'] for x in outputs]).mean()
        avg_acc = np.stack([x['acc'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['recon_loss'] for x in outputs]).mean()
        avg_kld_normal = torch.stack([x['kld_normal'] for x in outputs]).mean()
        avg_kld_laplace = torch.stack(
            [x['kld_laplace'] for x in outputs]).mean()
        if avg_mcc > self.best_mcc:
            self.best_mcc = avg_mcc
        self.log_dict({'val/mcc': avg_mcc,
                       'val/acc': avg_acc,
                       'val/best_mcc': self.best_mcc}, prog_bar=True)
        self.log_dict({'val/loss': avg_loss,
                       'val/recon_loss': avg_recon_loss,
                       'val/kld_normal': avg_kld_normal,
                       'val/kld_laplace': avg_kld_laplace})
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        opt_v = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters(
        )), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        return [opt_v], []


class NCTRL(CTDRL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hmm = HMM(n_class=self.n_class, lags=self.lags,
                       x_dim=self.z_dim, hidden_dim=self.hidden_dim, mode=kwargs['hmm_mode'])
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):

        model_opt, hmm_opt = self.optimizers()
        x, _, _ = batch
        _, lags_and_length, _ = x.shape
        # x_flat = x.view(-1, x_dim)

        x_recon, mus, logvars, z_est = self.net(x)
        E_logp_x, c_est = self.hmm(x)
        # * (self.alpha_factor**self.current_epoch) if self.current_epoch < 5 else -E_logp_x.mean() * 0.0
        hmm_loss = -E_logp_x.mean()

        hmm_opt.zero_grad()
        self.manual_backward(hmm_loss)
        hmm_opt.step()

        # (batch_size, lags+length, embedding_dim)
        embeddings = self.c_embeddings(c_est)

        # recon_loss = self.reconstruction_loss(x, x_recon)
        recon_loss = self.reconstruction_loss(x[:, :self.lags], x_recon[:, :self.lags]) + \
            (self.reconstruction_loss(
                x[:, self.lags:], x_recon[:, self.lags:]))/(lags_and_length-self.lags)

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lags:]
        residuals, logabsdet = self.transition_prior(z_est, embeddings)
        log_pz_laplace = torch.sum(
            self.base_dist.log_prob(residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1),
                       dim=-1) - log_pz_laplace) / (lags_and_length-self.lags)
        kld_laplace = kld_laplace.mean()

        # VAE training
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace

        model_opt.zero_grad()
        self.manual_backward(loss)
        model_opt.step()

        self.log_dict({"train/elbo_loss": loss,
                       "train/recon_loss": recon_loss,
                       "train/hmm_loss": hmm_loss,
                       "train/kld_normal": kld_normal,
                       "train/kld_laplace": kld_laplace})

    def validation_step(self, batch, batch_idx):
        x, z, c = batch
        _, lags_and_length, _ = x.shape
        # x_flat = x.view(-1, x_dim)

        x_recon, mus, logvars, z_est = self.net(x)
        E_logp_x, c_est = self.hmm(x)
        hmm_loss = -E_logp_x.mean()
        # (batch_size, lags+length, embedding_dim)
        embeddings = self.c_embeddings(c_est)

        # recon_loss = self.reconstruction_loss(x, x_recon)
        recon_loss = self.reconstruction_loss(x[:, :self.lags], x_recon[:, :self.lags]) + \
            (self.reconstruction_loss(
                x[:, self.lags:], x_recon[:, self.lags:]))/(lags_and_length-self.lags)

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lags:]
        residuals, logabsdet = self.transition_prior(z_est, embeddings)
        log_pz_laplace = torch.sum(
            self.base_dist.log_prob(residuals), dim=1) + logabsdet.sum(dim=1)
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1),
                       dim=-1) - log_pz_laplace) / (lags_and_length-self.lags)
        kld_laplace = kld_laplace.mean()

        # VAE training
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace

        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = z.view(-1, self.z_dim).T.detach().cpu().numpy()
        ct_est = c_est.cpu().flatten().numpy()
        ct_true = c[:, self.lags:].cpu().flatten().numpy()
        # mcc,_,_ = compute_mcc(zt_recon, zt_true, self.correlation)
        self.validation_step_outputs.append({'loss': loss ,'recon_loss': recon_loss, 'hmm_loss': hmm_loss, 'kld_normal': kld_normal, 'kld_laplace': kld_laplace,
                                             'raw': {'z': zt_true, 'z_est': zt_recon, 'c': ct_true, 'c_est': ct_est}})

    def on_validation_epoch_end(self) -> None:
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # mcc = torch.stack([x['mcc'] for x in outputs]).mean()
        avg_recon_loss = torch.stack([x['recon_loss'] for x in outputs]).mean()
        avg_hmm_loss = torch.stack([x['hmm_loss'] for x in outputs]).mean()
        avg_kld_normal = torch.stack([x['kld_normal'] for x in outputs]).mean()
        avg_kld_laplace = torch.stack(
            [x['kld_laplace'] for x in outputs]).mean()
        # check A and acc
        A_est = torch.log_softmax(
            self.hmm.log_A.detach().cpu(), dim=1).exp().numpy()
        A = self.A.detach().cpu().numpy()
        c = np.concatenate([x['raw']['c'] for x in outputs], axis=0)
        c_est = np.concatenate([x['raw']['c_est'] for x in outputs], axis=0)
        z = np.concatenate([x['raw']['z'] for x in outputs], axis=1)
        z_est = np.concatenate([x['raw']['z_est'] for x in outputs], axis=1)
        acc, matchidx = compute_acc(c, c_est, C=self.n_class)
        A_permuted = A[matchidx, :][:, matchidx]
        A_err = np.abs(A_permuted - A_est).mean()
        mcc = compute_mcc(z_est, z, self.correlation)
        if mcc > self.best_mcc:
            self.best_mcc = mcc
        self.log_dict({'val/mcc': mcc,
                       'val/A_err': A_err,
                       'val/acc': acc,
                       'val/hmm_loss': avg_hmm_loss,
                       'val/best_mcc': self.best_mcc}, prog_bar=True)
        self.log_dict({'val/loss': avg_loss,
                       'val/recon_loss': avg_recon_loss,
                       'val/kld_normal': avg_kld_normal,
                       'val/kld_laplace': avg_kld_laplace})
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # params = list(self.named_parameters())
        # require_grad = [(name,param) for name, param in params if param.requires_grad]
        model_params = [param for name, param in self.named_parameters(
        ) if not name.startswith('hmm.')]
        hmm_params = [param for name, param in self.named_parameters(
        ) if name.startswith('hmm.')]
        model_opt = torch.optim.AdamW(model_params, lr=self.lr, betas=(0.9, 0.999), weight_decay=0.0001)
        hmm_opt = torch.optim.Adam(hmm_params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0001)
        return model_opt, hmm_opt


class NCTRLz(NCTRL):
    def training_step(self, batch, batch_idx):

        model_opt, hmm_opt = self.optimizers()
        x, _, _ = batch
        _, lags_and_length, _ = x.shape
        # x_flat = x.view(-1, x_dim)

        x_recon, mus, logvars, z_est = self.net(x)
        E_logp_x, c_est = self.hmm(z_est.detach())
        # * (self.alpha_factor**self.current_epoch) if self.current_epoch < 5 else -E_logp_x.mean() * 0.0
        hmm_loss = -E_logp_x.mean()

        hmm_opt.zero_grad()
        self.manual_backward(hmm_loss)
        hmm_opt.step()

        # (batch_size, lags+length, embedding_dim)
        embeddings = self.c_embeddings(c_est)

        # recon_loss = self.reconstruction_loss(x, x_recon)
        recon_loss = self.reconstruction_loss(x[:, :self.lags], x_recon[:, :self.lags]) + \
            (self.reconstruction_loss(
                x[:, self.lags:], x_recon[:, self.lags:]))/(lags_and_length-self.lags)

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lags:]
        residuals, logabsdet = self.transition_prior(z_est, embeddings)
        log_pz_laplace = torch.sum(
            self.base_dist.log_prob(residuals), dim=1) + logabsdet.sum(1)
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1),
                       dim=-1) - log_pz_laplace) / (lags_and_length-self.lags)
        kld_laplace = kld_laplace.mean()

        # VAE training
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace

        model_opt.zero_grad()
        self.manual_backward(loss)
        model_opt.step()

        self.log_dict({"train/elbo_loss": loss,
                       "train/recon_loss": recon_loss,
                       "train/hmm_loss": hmm_loss,
                       "train/kld_normal": kld_normal,
                       "train/kld_laplace": kld_laplace})

    def validation_step(self, batch, batch_idx):
        x, z, c = batch
        _, lags_and_length, _ = x.shape
        # x_flat = x.view(-1, x_dim)

        x_recon, mus, logvars, z_est = self.net(x)
        E_logp_x, c_est = self.hmm(z_est.detach())
        hmm_loss = -E_logp_x.mean()
        # (batch_size, lags+length, embedding_dim)
        embeddings = self.c_embeddings(c_est)

        # recon_loss = self.reconstruction_loss(x, x_recon)
        recon_loss = self.reconstruction_loss(x[:, :self.lags], x_recon[:, :self.lags]) + \
            (self.reconstruction_loss(
                x[:, self.lags:], x_recon[:, self.lags:]))/(lags_and_length-self.lags)

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(
            mus[:, :self.lags]), torch.ones_like(logvars[:, :self.lags]))
        log_pz_normal = torch.sum(
            torch.sum(p_dist.log_prob(z_est[:, :self.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(
            torch.sum(log_qz[:, :self.lags], dim=-1), dim=-1)
        kld_normal = log_qz_normal - log_pz_normal
        kld_normal = kld_normal.mean()
        # Future KLD
        log_qz_laplace = log_qz[:, self.lags:]
        residuals, logabsdet = self.transition_prior(z_est, embeddings)
        log_pz_laplace = torch.sum(
            self.base_dist.log_prob(residuals), dim=1) + logabsdet.sum(1)
        kld_laplace = (torch.sum(torch.sum(log_qz_laplace, dim=-1),
                       dim=-1) - log_pz_laplace) / (lags_and_length-self.lags)
        kld_laplace = kld_laplace.mean()

        # VAE training
        loss = recon_loss + self.beta * kld_normal + self.gamma * kld_laplace

        zt_recon = mus.view(-1, self.z_dim).T.detach().cpu().numpy()
        zt_true = z.view(-1, self.z_dim).T.detach().cpu().numpy()
        ct_est = c_est.cpu().flatten().numpy()
        ct_true = c[:, self.lags:].cpu().flatten().numpy()
        # mcc,_,_ = compute_mcc(zt_recon, zt_true, self.correlation)
        self.validation_step_outputs.append({'loss': loss, 'recon_loss': recon_loss, 'hmm_loss': hmm_loss, 'kld_normal': kld_normal, 'kld_laplace': kld_laplace,
                                             'raw': {'z': zt_true, 'z_est': zt_recon, 'c': ct_true, 'c_est': ct_est}})
