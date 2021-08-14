from vae import VaeTrainer
from mdrnn import MDRNNTrainer
from ppo_controller import ControllerTrainer
from generate_datasets import DataGenerator
import torch
class EvolutionTrainer:
    def __init__(self):
        self.vae_epochs = 30
        self.mdrnn_epochs = 30
        self.controller_epochs = 1e6

        self.vae_batch_size = 64
        self.mdrnn_batch_size = 32
        self.controller_batch_size = 128

        self.logdir = 'exp'
        self.data_generator_rollout = 1024
        self.data_path = 'datasets/carracing'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_generator = DataGenerator(self.data_path, self.logdir, self.device)
        self.data_generator.generate_random_data(1024)
        self.vae_trainer = VaeTrainer(self.vae_batch_size, logdir=self.logdir, data_path=self.data_path, device=self.device)
        self.mdrnn_trainer = MDRNNTrainer(self.mdrnn_batch_size, logdir=self.logdir, data_path=self.data_path, device=self.device)
        self.control_trainer = ControllerTrainer(self.logdir, device=self.device)
        self.vae_best = None
        self.mdrnn_best = None


    def train(self, evolutions=5):
        for _ in range(evolutions):
            print('-'*30, 'epoch ', _, '-'*30)
            self.vae_trainer.load_state_dict()
            self.vae_best = self.vae_trainer.train_vae(self.vae_epochs, self.vae_best)

            self.mdrnn_trainer.load_state_dict()
            self.mdrnn_best = self.mdrnn_trainer.train_mdrnn(self.mdrnn_epochs, self.mdrnn_best)

            self.control_trainer.load_state_dict()
            self.control_trainer.train_epochs(self.controller_epochs)
            self.data_generator.generate_data(self.control_trainer.args.agent)
            print(f'vae best:{self.vae_best}, mdrnn best:{self.mdrnn_best}')
            print('\n'*10)


if __name__ == "__main__":
    evo = EvolutionTrainer()
    evo.train()