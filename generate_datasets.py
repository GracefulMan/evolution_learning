"""
Generating data from the CarRacing gym environment.
!!! DOES NOT WORK ON TITANIC, DO IT AT HOME, THEN SCP !!!
"""
import os.path
from os.path import join, exists

import gym
import numpy as np
import torch
from torchvision import transforms

from models.mdrnn import MDRNNCell
from models.vae import VAE
from config import LSIZE, ASIZE, RSIZE, RED_SIZE
from utils.misc import sample_continuous_policy

gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64


class DataGenerator:
    def __init__(self, data_dir='datasets', logdir='exp', device=None):
        self.mdir = logdir
        self.device = device
        self.vae = None
        self.mdrnn = None
        self.data_dir = os.path.join(data_dir, 'carracing')
        self.start_index = 0
        self.max_index = 2048

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((RED_SIZE, RED_SIZE)),
            transforms.ToTensor()
        ])
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load_model_and_params(self):
        # Loading world model and vae
        vae_file, rnn_file, ctrl_file = \
            [join(self.mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]

        assert exists(vae_file) and exists(rnn_file), \
            "Either vae or mdrnn is untrained."
        vae_state, rnn_state = [
            torch.load(fname, map_location='cuda:0')
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(self.device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(self.device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

    def generate_data(self, agent, rollouts=512):
        self.load_model_and_params()
        env = gym.make("CarRacing-v0", verbose=False)
        for i in range(rollouts):
            s = env.reset()
            hidden = [
                torch.zeros(1, RSIZE).to(self.device)
                for _ in range(2)]
            env.env.viewer.window.dispatch_events()
            a_rollout = []
            s_rollout = []
            r_rollout = []
            d_rollout = []

            obs = self.transform(s).unsqueeze(0).to(self.device)
            _, latent_mu, _ = self.vae(obs)
            self.hidden = [
                torch.zeros(1, RSIZE).to(self.device)
                for _ in range(2)]
            obs = torch.cat((latent_mu, hidden[0]), dim=1).detach().cpu().numpy().ravel()

            t = 0
            while True:
                t += 1
                action, _ = agent.select_action(obs)
                s, r, done, _ = env.step(action)

                s_rollout += [s]
                r_rollout += [r]
                d_rollout += [done]
                a_rollout += [action]

                obs = self.transform(s).unsqueeze(0).to(self.device)
                _, latent_mu, _ = self.vae(obs)
                action = torch.as_tensor((action,), dtype=torch.float32, device=self.device)
                _, _, _, _, _, hidden = self.mdrnn(action, latent_mu, hidden)
                obs = torch.cat((latent_mu, hidden[0]), dim=1).detach().cpu().numpy().ravel()
                env.env.viewer.window.dispatch_events()

                if done:
                    print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                    np.savez(join(join(self.data_dir, 'thread_0'),
                                  'rollout_{}'.format((i + self.start_index) % self.max_index)),
                             observations=np.array(s_rollout),
                             rewards=np.array(r_rollout),
                             actions=np.array(a_rollout),
                             terminals=np.array(d_rollout))
                    break
        self.start_index += rollouts

    def generate_random_data(self, rollouts=1024, noise_type='brown', dir_name='thread_1'):  # pylint: disable=R0914
        env = gym.make("CarRacing-v0", verbose=False)
        seq_len = 1000
        save_dir = join(self.data_dir, str(dir_name))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(rollouts):
            env.reset()
            env.env.viewer.window.dispatch_events()
            if noise_type == 'white':
                a_rollout = [env.action_space.sample() for _ in range(seq_len)]
            elif noise_type == 'brown':
                a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

            s_rollout = []
            r_rollout = []
            d_rollout = []

            t = 0
            while True:
                action = a_rollout[t]
                t += 1
                s, r, done, _ = env.step(action)
                env.env.viewer.window.dispatch_events()
                s_rollout += [s]
                r_rollout += [r]
                d_rollout += [done]
                if done:
                    print(">worker:{} End of rollout {}, {} frames...".format(dir_name, i, len(s_rollout)))
                    np.savez(join(save_dir, 'rollout_{}'.format((self.start_index + i) % self.max_index)),
                             observations=np.array(s_rollout),
                             rewards=np.array(r_rollout),
                             actions=np.array(a_rollout),
                             terminals=np.array(d_rollout))
                    break
        self.start_index += rollouts

    def multiworker(self, workers, rollouts, noise_type='brown'):
        from multiprocessing import Pool
        pool = Pool(workers)
        for i in range(workers):
            pool.apply_async(self.generate_random_data, args=(rollouts, noise_type, i))
        pool.close()
        pool.join()


if __name__ == '__main__':
    device = torch.device('cuda')
    import argparse
    parser = argparse.ArgumentParser(description='generate datasets.')
    parser.add_argument('--data-dir', type=str, help='datasets dirs', default='datasets')
    parser.add_argument('--logdir', type=str, help='model saved dir', default='exp')
    parser.add_argument('--policy', type=str, help='the random or ppo policy', default='random')
    parser.add_argument('--rollouts', type=int, help='the num of rollouts to generate', default=512)
    parser.add_argument('--worker', type=int, default=8)
    args = parser.parse_args()
    print(args)
    datagenerator = DataGenerator(data_dir=args.data_dir, logdir=args.logdir, device=device)
    if args.policy == 'random':
        datagenerator.multiworker(args.worker, args.rollouts)
    else:
        from ppo_controller import ControllerTrainer

        ppo = ControllerTrainer(device=device)
        datagenerator.generate_data(ppo.args.agent, args.rollouts)
