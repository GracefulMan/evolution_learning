"""
Training a linear controller on latent + recurrent state
with CMAES.

This is a bit complex. num_workers slave threads are launched
to process a queue filled with parameters to be evaluated.
"""
import sys
from os import mkdir, unlink, listdir, getpid
from os.path import join, exists
from time import sleep

import cma
import numpy as np
import torch
from torch.multiprocessing import Process, Queue
from tqdm import tqdm

from models import Controller
from utils.misc import RolloutGenerator, ASIZE, RSIZE, LSIZE
from utils.misc import flatten_parameters
from utils.misc import load_parameters


class ControllerTrainer:
    def __init__(self, logdir='exp2', n_samples=4, pop_size=4, num_workers=8, target_return=950, display=True):
        self.logdir = logdir
        self.n_samples = n_samples
        self.pop_size = pop_size
        self.display = display
        self.num_workers = min(self.n_samples * self.pop_size, num_workers)
        self.target_return = target_return
        self.create_dirs()
        self.p_queue = Queue()
        self.r_queue = Queue()
        self.e_queue = Queue()
        self.time_limit = 1000
        self.create_multiprocess()
        self.controller = Controller(LSIZE, RSIZE, ASIZE)  # dummy instance
        self.load_state_dict()
        self.parameters = self.controller.parameters()
        self.es = cma.CMAEvolutionStrategy(flatten_parameters(self.parameters), 0.1,
                                           {'popsize': self.pop_size})

    def create_multiprocess(self):
        for p_index in range(self.num_workers):
            Process(target=self.slave_routine, args=(self.p_queue, self.r_queue, self.e_queue, p_index)).start()

    def create_dirs(self):
        # create tmp dir if non existent and clean it if existent
        tmp_dir = join(self.logdir, 'tmp')
        if not exists(tmp_dir):
            mkdir(tmp_dir)
        else:
            for fname in listdir(tmp_dir):
                unlink(join(tmp_dir, fname))

        # create ctrl dir if non exitent
        ctrl_dir = join(self.logdir, 'ctrl')
        if not exists(ctrl_dir):
            mkdir(ctrl_dir)

    def slave_routine(self, p_queue, r_queue, e_queue, p_index):
        """ Thread routine.

        Threads interact with p_queue, the parameters queue, r_queue, the result
        queue and e_queue the end queue. They pull parameters from p_queue, execute
        the corresponding rollout, then place the result in r_queue.

        Each parameter has its own unique id. Parameters are pulled as tuples
        (s_id, params) and results are pushed as (s_id, result).  The same
        parameter can appear multiple times in p_queue, displaying the same id
        each time.

        As soon as e_queue is non empty, the thread terminate.

        When multiple gpus are involved, the assigned gpu is determined by the
        process index p_index (gpu = p_index % n_gpus).

        :args p_queue: queue containing couples (s_id, parameters) to evaluate
        :args r_queue: where to place results (s_id, results)
        :args e_queue: as soon as not empty, terminate
        :args p_index: the process index
        """
        # init routine
        gpu = p_index % torch.cuda.device_count()
        device = torch.device('cuda'.format(gpu) if torch.cuda.is_available() else 'cpu')

        # redirect streams
        tmp_dir = join(self.logdir, 'tmp')
        sys.stdout = open(join(tmp_dir, str(getpid()) + '.out'), 'a')
        sys.stderr = open(join(tmp_dir, str(getpid()) + '.err'), 'a')

        with torch.no_grad():
            r_gen = RolloutGenerator(self.logdir, device, self.time_limit)

            while e_queue.empty():
                if p_queue.empty():
                    sleep(.1)
                else:
                    s_id, params = p_queue.get()
                    r_queue.put((s_id, r_gen.rollout(params)))

    def evaluate(self, solutions, results, rollouts=100):
        """ Give current controller evaluation.

        Evaluation is minus the cumulated reward averaged over rollout runs.

        :args solutions: CMA set of solutions
        :args results: corresponding results
        :args rollouts: number of rollouts

        :returns: minus averaged cumulated reward
        """
        index_min = np.argmin(results)
        best_guess = solutions[index_min]
        restimates = []

        for s_id in range(rollouts):
            self.p_queue.put((s_id, best_guess))

        print("Evaluating...")
        for _ in tqdm(range(rollouts)):
            while self.r_queue.empty():
                sleep(.1)
            restimates.append(self.r_queue.get()[1])

        return best_guess, np.mean(restimates), np.std(restimates)

    def load_state_dict(self):
        # define current best and load parameters
        cur_best = None
        ctrl_dir = join(self.logdir, 'ctrl')
        ctrl_file = join(ctrl_dir, 'best.tar')
        print("Attempting to load previous best...")
        if exists(ctrl_file):
            state = torch.load(ctrl_file, map_location={'cuda:0': 'cpu'})
            cur_best = - state['reward']
            self.controller.load_state_dict(state['state_dict'])
            print("Previous best was {}...".format(-cur_best))

    def train_controller(self, cur_best=None):
        epoch = 0
        log_step = 3
        while not self.es.stop():
            if cur_best is not None and - cur_best > self.target_return:
                print("Already better than target, breaking...")
                break

            r_list = [0] * self.pop_size  # result list
            solutions = self.es.ask()

            # push parameters to queue
            for s_id, s in enumerate(solutions):
                for _ in range(self.n_samples):
                    self.p_queue.put((s_id, s))

            # retrieve results
            if self.display:
                pbar = tqdm(total=self.pop_size * self.n_samples)
            for _ in range(self.pop_size * self.n_samples):
                while self.r_queue.empty():
                    sleep(.1)
                r_s_id, r = self.r_queue.get()
                r_list[r_s_id] += r / self.n_samples
                if self.display:
                    pbar.update(1)
            if self.display:
                pbar.close()

            self.es.tell(solutions, r_list)
            self.es.disp()

            # evaluation and saving
            if epoch % log_step == log_step - 1:
                best_params, best, std_best = self.evaluate(solutions, r_list)
                print("Current evaluation: {}".format(best))
                if not cur_best or cur_best > best:
                    cur_best = best
                    print("Saving new best with value {}+-{}...".format(-cur_best, std_best))
                    load_parameters(best_params, self.controller)
                    torch.save(
                        {'epoch': epoch,
                         'reward': - cur_best,
                         'state_dict': self.controller.state_dict()},
                        join(join(self.logdir, 'ctrl'), 'best.tar'))
                if - best > self.target_return:
                    print("Terminating controller training with value {}...".format(best))
                    break
            epoch += 1
        self.es.result_pretty()
        self.e_queue.put('EOP')
        return cur_best


if __name__ == '__main__':
    controller = ControllerTrainer()
    controller.train_controller()