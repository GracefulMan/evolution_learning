""" Recurrent model training """
from functools import partial
from os import mkdir
from os.path import join, exists

import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils.loaders import RolloutSequenceDataset
from models.mdrnn import MDRNN, gmm_loss
from utils.learning import EarlyStopping
## WARNING : THIS SHOULD BE REPLACED WITH PYTORCH 0.5
from utils.learning import ReduceLROnPlateau
from utils.misc import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE
from utils.misc import save_checkpoint
from models.vae import VAE

class MDRNNTrainer:
    def __init__(self, batch_size=32, seq_len=32, logdir='exp2',data_path='datasets/carracing',noreload=False, include_reward=False,
                 device=None):
        self.logdir = logdir
        self.noreload = noreload
        self.include_reward = include_reward
        self.BSIZE = batch_size
        self.SEQ_LEN = seq_len
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.train_loader, self.test_loader = self.data_loader()
        self.mdrnn = MDRNN(LSIZE, ASIZE, RSIZE, 5).to(self.device)
        self.vae = VAE(3, LSIZE).to(self.device)
        self.optimizer = torch.optim.RMSprop(self.mdrnn.parameters(), lr=1e-3, alpha=.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5)
        self.earlystopping = EarlyStopping('min', patience=30)
        self.load_state_dict()
        self.train = partial(self.data_pass, train=True, include_reward=self.include_reward)
        self.test = partial(self.data_pass, train=False, include_reward=self.include_reward)

    def data_loader(self):
        # Data Loading
        transform = transforms.Lambda(
            lambda x: np.transpose(x, (0, 3, 1, 2)) / 255)
        train_loader = DataLoader(
            RolloutSequenceDataset(self.data_path, self.SEQ_LEN, transform, buffer_size=30),
            batch_size=self.BSIZE, num_workers=8, shuffle=True, drop_last=True)
        test_loader = DataLoader(
            RolloutSequenceDataset(self.data_path, self.SEQ_LEN, transform, train=False, buffer_size=10),
            batch_size=self.BSIZE, num_workers=8, drop_last=True)
        return train_loader, test_loader

    def load_state_dict(self):
        vae_file = join(self.logdir, 'vae', 'best.tar')
        assert exists(vae_file), "No trained VAE in the logdir..."
        state = torch.load(vae_file)
        print("Loading VAE at epoch {} "
              "with test error {}".format(
            state['epoch'], state['precision']))
        self.vae.load_state_dict(state['state_dict'])
        # Loading model
        self.rnn_dir = join(self.logdir, 'mdrnn')
        self.rnn_file = join(self.rnn_dir, 'best.tar')
        if not exists(self.rnn_dir):
            mkdir(self.rnn_dir)

        if exists(self.rnn_file) and not self.noreload:
            rnn_state = torch.load(self.rnn_file)
            print("Loading MDRNN at epoch {} "
                  "with test error {}".format(
                rnn_state["epoch"], rnn_state["precision"]))
            self.mdrnn.load_state_dict(rnn_state["state_dict"])
            self.optimizer.load_state_dict(rnn_state["optimizer"])
            self.scheduler.load_state_dict(rnn_state['scheduler'])
            self.earlystopping.load_state_dict(rnn_state['earlystopping'])

    def to_latent(self, obs, next_obs):
        """ Transform observations to latent space.

        :args obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)
        :args next_obs: 5D torch tensor (BSIZE, SEQ_LEN, ASIZE, SIZE, SIZE)

        :returns: (latent_obs, latent_next_obs)
            - latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
            - next_latent_obs: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        """
        with torch.no_grad():
            obs, next_obs = [
                f.upsample(x.view(-1, 3, SIZE, SIZE), size=RED_SIZE,
                           mode='bilinear', align_corners=True)
                for x in (obs, next_obs)]

            (obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma) = [
                self.vae(x)[1:] for x in (obs, next_obs)]

            latent_obs, latent_next_obs = [
                (x_mu + x_logsigma.exp() * torch.randn_like(x_mu)).view(self.BSIZE, self.SEQ_LEN, LSIZE)
                for x_mu, x_logsigma in
                [(obs_mu, obs_logsigma), (next_obs_mu, next_obs_logsigma)]]
        return latent_obs, latent_next_obs

    def get_loss(self, latent_obs, action, reward, terminal, latent_next_obs, include_reward: bool):
        """ Compute losses.

        The loss that is computed is:
        (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
             BCE(terminal, logit_terminal)) / (LSIZE + 2)
        The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
        approximately linearily with LSIZE. All losses are averaged both on the
        batch and the sequence dimensions (the two first dimensions).

        :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args reward: (BSIZE, SEQ_LEN) torch tensor
        :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """
        latent_obs, action, \
        reward, terminal, \
        latent_next_obs = [arr.transpose(1, 0)
                           for arr in [latent_obs, action,
                                       reward, terminal,
                                       latent_next_obs]]
        mus, sigmas, logpi, rs, ds = self.mdrnn(action, latent_obs)
        gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
        bce = f.binary_cross_entropy_with_logits(ds, terminal)
        if include_reward:
            mse = f.mse_loss(rs, reward)
            scale = LSIZE + 2
        else:
            mse = 0
            scale = LSIZE + 1
        loss = (gmm + bce + mse) / scale
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)

    def data_pass(self, epoch, train, include_reward):  # pylint: disable=too-many-locals
        """ One pass through the data """
        if train:
            self.mdrnn.train()
            loader = self.train_loader
        else:
            self.mdrnn.eval()
            loader = self.test_loader

        loader.dataset.load_next_buffer()

        cum_loss = 0
        cum_gmm = 0
        cum_bce = 0
        cum_mse = 0

        pbar = tqdm(total=len(loader.dataset), desc="Epoch {}".format(epoch))
        for i, data in enumerate(loader):
            obs, action, reward, terminal, next_obs = [arr.to(self.device) for arr in data]

            # transform obs
            latent_obs, latent_next_obs = self.to_latent(obs, next_obs)

            if train:
                losses = self.get_loss(latent_obs, action, reward,
                                       terminal, latent_next_obs, include_reward)

                self.optimizer.zero_grad()
                losses['loss'].backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    losses = self.get_loss(latent_obs, action, reward,
                                           terminal, latent_next_obs, include_reward)

            cum_loss += losses['loss'].item()
            cum_gmm += losses['gmm'].item()
            cum_bce += losses['bce'].item()
            cum_mse += losses['mse'].item() if hasattr(losses['mse'], 'item') else \
                losses['mse']

            pbar.set_postfix_str("loss={loss:10.6f} bce={bce:10.6f} "
                                 "gmm={gmm:10.6f} mse={mse:10.6f}".format(
                loss=cum_loss / (i + 1), bce=cum_bce / (i + 1),
                gmm=cum_gmm / LSIZE / (i + 1), mse=cum_mse / (i + 1)))
            pbar.update(self.BSIZE)
        pbar.close()
        return cum_loss * self.BSIZE / len(loader.dataset)

    def train_mdrnn(self, epochs, cur_best=None):
        for e in range(epochs):
            self.train(e)
            test_loss = self.test(e)
            self.scheduler.step(test_loss)
            self.earlystopping.step(test_loss)

            is_best = not cur_best or test_loss < cur_best
            if is_best:
                cur_best = test_loss
            checkpoint_fname = join(self.rnn_dir, 'checkpoint.tar')
            save_checkpoint({
                "state_dict": self.mdrnn.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'earlystopping': self.earlystopping.state_dict(),
                "precision": test_loss,
                "epoch": e}, is_best, checkpoint_fname,
                self.rnn_file)

            if self.earlystopping.stop:
                print("End of Training because of early stopping at epoch {}".format(e))
                break
        return cur_best

if __name__ == '__main__':
    mdrnn = MDRNNTrainer()
    cur_best = mdrnn.train_mdrnn(10)
    cur_best=mdrnn.train_mdrnn(30, cur_best)
