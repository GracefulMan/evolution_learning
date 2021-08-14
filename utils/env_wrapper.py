import gym
import torch
from torchvision import transforms
from models.vae import VAE
from models.mdrnn import MDRNNCell
from os.path import join, exists
from utils.misc import LSIZE, ASIZE, RSIZE, RED_SIZE


class CarRacingEnv(gym.Wrapper):
    def __init__(self, env, mdir, device):
        super(CarRacingEnv, self).__init__(env)
        self.mdir = mdir
        self.device = device
        self.vae = None
        self.mdrnn = None
        self.load_model_and_params()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((RED_SIZE, RED_SIZE)),
            transforms.ToTensor()
        ])
        self.observation_space.shape = (LSIZE + RSIZE,)

    def reset(self, **kwargs):
        obs = self.env.reset()
        obs = self.transform(obs).unsqueeze(0).to(self.device)
        _, latent_mu, _ = self.vae(obs)
        self.hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]
        obs = torch.cat((latent_mu, self.hidden[0]), dim=1)
        # return obs.detach().cpu().numpy().ravel()
        return obs.detach().cpu().numpy().ravel()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.transform(obs).unsqueeze(0).to(self.device)
        _, latent_mu, _ = self.vae(obs)
        action = torch.as_tensor((action, ), dtype=torch.float32, device=self.device)
        _, _, _, _, _, self.hidden = self.mdrnn(action, latent_mu, self.hidden)
        obs = torch.cat((latent_mu, self.hidden[0]), dim=1)
        return obs.detach().cpu().numpy().ravel(), reward, done, info

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


if __name__ == '__main__':
    mdir = 'F:\科研与实验\evolution_learning\exp'
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    env = gym.make('CarRacing-v0')
    env = CarRacingEnv(env, mdir, device)
    print(env.observation_space)
    obs = env.reset()
    print(obs)
    index = 1
    import time
    t =time.time()
    while True:
        # env.render()
        action = env.action_space.sample()
        obs, reward, done, info=env.step(action)
        print(index)
        index+=1
        if done:
            break
    print(time.time() - t)