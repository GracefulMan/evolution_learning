""" Training VAE """
from os import mkdir
from os.path import join, exists

import torch
import torch.utils.data
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import save_image

from utils.loaders import RolloutObservationDataset
from models.vae import VAE
from utils.learning import EarlyStopping
from utils.learning import ReduceLROnPlateau
from utils.misc import save_checkpoint
from config import LSIZE, RED_SIZE
torch.manual_seed(123)
# Fix numeric divergence due to bug in Cudnn
torch.backends.cudnn.benchmark = True


class VaeTrainer:
    def __init__(self, batch_size=64, logdir='exp', data_path='datasets/carracing', noreload=False, nosamples=False, device=None):
        self.batch_size = batch_size
        self.logdir = logdir
        self.noreload = noreload
        self.nosamples = nosamples
        self.data_path = data_path
        self.dataset_train, self.dataset_test, self.train_loader, self.test_loader = self.data_loader()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.model = VAE(3, LSIZE).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5)
        self.earlystopping = EarlyStopping('min', patience=30)
        self.load_state_dict()

    def data_loader(self):
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((RED_SIZE, RED_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((RED_SIZE, RED_SIZE)),
            transforms.ToTensor(),
        ])
        dataset_train = RolloutObservationDataset(self.data_path,
                                                  transform_train, train=True)
        dataset_test = RolloutObservationDataset(self.data_path,
                                                 transform_test, train=False)
        train_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=2)
        test_loader = torch.utils.data.DataLoader(
            dataset_test, batch_size=self.batch_size, shuffle=True, num_workers=2)
        return dataset_train, dataset_test, train_loader, test_loader

    @staticmethod
    def loss_function(recon_x, x, mu, logsigma):
        """ VAE loss function """
        BCE = F.mse_loss(recon_x, x, size_average=False)
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + 2 * logsigma - mu.pow(2) - (2 * logsigma).exp())
        return BCE + KLD

    def train(self, epoch):
        """ One training epoch """
        self.model.train()
        self.dataset_train.load_next_buffer()
        train_loss = 0
        for batch_idx, data in enumerate(self.train_loader):
            data = data.to(self.device)
            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            if batch_idx % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader),
                           loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(self.train_loader.dataset)))

    def load_state_dict(self):
        # check vae dir exists, if not, create it
        self.vae_dir = join(self.logdir, 'vae')
        if not exists(self.vae_dir):
            mkdir(self.vae_dir)
            mkdir(join(self.vae_dir, 'samples'))
        reload_file = join(self.vae_dir, 'best.tar')
        if not self.noreload and exists(reload_file):
            state = torch.load(reload_file)
            print("Reloading model at epoch {}"
                  ", with test error {}".format(
                state['epoch'],
                state['precision']))
            self.model.load_state_dict(state['state_dict'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.earlystopping.load_state_dict(state['earlystopping'])

    def test(self):
        """ One test epoch """
        self.model.eval()
        self.dataset_test.load_next_buffer()
        test_loss = 0
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                recon_batch, mu, logvar = self.model(data)
                test_loss += self.loss_function(recon_batch, data, mu, logvar).item()

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        return test_loss

    def train_vae(self, epochs, cur_best=None):
        for epoch in range(1, epochs + 1):
            self.train(epoch)
            test_loss = self.test()
            self.scheduler.step(test_loss)
            self.earlystopping.step(test_loss)

            # checkpointing
            best_filename = join(self.vae_dir, 'best.tar')
            filename = join(self.vae_dir, 'checkpoint.tar')
            is_best = not cur_best or test_loss < cur_best
            if is_best:
                cur_best = test_loss
            save_checkpoint({
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'precision': test_loss,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'earlystopping': self.earlystopping.state_dict()
            }, is_best, filename, best_filename)

            if not self.nosamples:
                with torch.no_grad():
                    sample = torch.randn(RED_SIZE, LSIZE).to(self.device)
                    sample = self.model.decoder(sample).cpu()
                    save_image(sample.view(64, 3, RED_SIZE, RED_SIZE),
                               join(self.vae_dir, 'samples/sample_' + str(epoch) + '.png'))

            if self.earlystopping.stop:
                print("End of Training because of early stopping at epoch {}".format(epoch))
                break
        return cur_best

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='VAE Module')
    parser.add_argument('--batch-size', type=int, help='train batch size',default=256)
    parser.add_argument('--retrain', type=bool, help='retrain the model', default=False)
    parser.add_argument('--epochs', type=int, help='train epochs', default=1000)
    args = parser.parse_args()
    print(args)
    Vae = VaeTrainer(batch_size=args.batch_size, noreload=args.retrain)
    Vae.train_vae(args.epochs)