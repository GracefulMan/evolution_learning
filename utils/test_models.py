from models.ppo import PreprocessEnv, AgentPPO
from utils.env_wrapper import CarRacingEnv
import gym
import torch
import imageio
def test_car_racing(save_video=False):
    agent = AgentPPO()
    env = gym.make('CarRacing-v0')
    mdir = 'F:\科研与实验\evolution_learning\exp'
    device = torch.device('cuda')
    env = CarRacingEnv(env, mdir, device)
    agent.init(2 ** 7, env.observation_space.shape[0], env.action_space.shape[0], 1e-4)
    agent.act.load_state_dict(torch.load('F:/科研与实验/evolution_learning/CarRacing-v0_AgentPPO/actor.pth'))
    avg = 0
    if save_video:
        writer = imageio.get_writer('car.mp4', fps=30)
    for _ in range(1):
        episode_return = 0
        done = False
        obs = env.reset()
        while not done:
            if save_video:
                img=env.render('rgb_array')
                writer.append_data(img)
            env.render()
            action,_ = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_return += reward
            if done:
                print("episode return:", episode_return)

        avg+= episode_return
    if save_video: writer.close()
    print('avg score:', avg/1)


if __name__ == '__main__':
    test_car_racing(save_video=True)