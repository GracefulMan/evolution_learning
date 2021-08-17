from models.ppo import AgentPPO,Arguments,PreprocessEnv, train_and_evaluate
import gym
from utils.env_wrapper import CarRacingEnv
import torch

class ControllerTrainer:
    def __init__(self, mdir='exp',device=None):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
        self.args.agent = AgentPPO()
        self.args.agent.cri_target = True  # True
        "TotalStep: 4e5, TargetReward: -200, UsedTime: 400s"
        env = gym.make('CarRacing-v0')
        env = CarRacingEnv(env, mdir, device)
        self.args.env = PreprocessEnv(env=env)
        self.args.env.target_return = 950  # set target_reward manually for env 'CarRacing-v0'
        self.args.net_dim = 2 ** 7
        self.args.batch_size = self.args.net_dim * 2
        self.args.target_step = self.args.env.max_step * 16
        self.args.break_step = int(5e7)
        self.args.random_seed = 1423
        self.args.eval_gap = 2 ** 7
        self.args.eval_times1 = 2
        self.args.eval_times2 = 4
        self.args.if_per_or_gae = True
        self.args.gamma = 0.97
        self.args.device = device
        self.args.gpu_id = 0

    def train_epochs(self, epochs):
        self.args.break_step = epochs
        train_and_evaluate(self.args)

    def load_state_dict(self):
        self.args.env.load_model_and_params()



if __name__ == '__main__':
    ppo = ControllerTrainer()
    ppo.train_epochs(5e7)



