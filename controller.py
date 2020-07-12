import numpy as np
import random
import os
import time
import json
from matplotlib import pyplot as plt
from tqdm import tqdm

from q_table import DiscreteQTable
from rbf_function import RBF_function

# Constant parameters
m = 0.055
g = 9.81
l = 0.042
J = 1.91e-4
b = 3e-6
K = 0.0536
R = 9.5
T = 0.005
voltage = [-3, 0, 3]
EPS = 1E-10

# Get reward with given status and action
def getReward(status, action):
    loc, speed = status
    u = voltage[action]
    return -5*loc*loc-0.1*speed*speed-1*u*u

# Constrain status in a certain range
def valid_range(status):
    if status[0] < -np.pi:
        status[0] += 2 * np.pi
    elif status[0] >= np.pi:
        status[0] -= 2 * np.pi
    if status[1] < -15 * np.pi:
        status[1] = -15 * np.pi
    elif status[1] > 15 * np.pi:
        status[1] = 15 * np.pi - EPS
    return status

def getNextStatus(status, action, n=1):
    loc, speed = status
    u = voltage[action]
    for _ in range(n):
        acceleration = (m*g*l*np.sin(loc) - b*speed - K*K*speed/R + K*u/R)/J
        #new_status = np.asarray([loc + speed*T, speed + acceleration*T])
        loc, speed = loc + speed*T, speed + acceleration*T
        new_status = np.asarray([loc, speed])
    new_status = valid_range(new_status)
    return new_status

class Controller(object):
    def __init__(self, args):
        self.is_train = True
        if args.function_type =="Table":
            self.q_function = DiscreteQTable(args.function)
        elif args.function_type =="RBF":
            self.q_function = RBF_function(args.function)
        else:
            raise ValueError("%s is not a valid function!"%(args.function_type))
        self.save_dir = args.save_dir
        self.save_epoch = args.save_epoch
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        args.save(os.path.join(self.save_dir, "config.json"))

    def take_action_random(self):
        return random.randint(0,2)

    def take_action_greedy(self, s):
        return self.q_function.greedy_action(s)

    def take_action(self, s):
        if self.is_train is True and random.random() < self.epsilon:
            return self.take_action_random()
        else:
            return self.take_action_greedy(s)
    
    def train_one(self, max_step, step_size, s = None):
        if s is None:
            s = np.asarray([random.uniform(-np.pi, np.pi), random.uniform(-15*np.pi, 15*np.pi)])
        a = self.take_action(s)
        self.is_train = True
        self.q_function.reset_et()
        trace = []

        for i in range(max_step):
            trace.append(s)
            s_1 = getNextStatus(s, a, step_size)
            r = getReward(s_1, a)
            #self.q_function.train_q_learning(s, a, r, s_1)
            a_1 = self.take_action(s_1)
            self.q_function.train_sarsa(s, a, r, s_1, a_1)
            s, a = s_1, a_1
        return np.stack(trace)
    
    def train(self, train_arg):
        max_step, step_size = train_arg["max_step"], train_arg["step_size"]
        base_lr, lr_step = train_arg["base_lr"], train_arg["lr_step"]
        self.epsilon = train_arg["random_rate"]
        start = np.asarray([-np.pi, 0]) if train_arg["fix_start"] else None
        f = open(os.path.join(self.save_dir, "log.txt"), "w+")
        start_time = time.time()
        trace = []
        for i in tqdm(range(train_arg["num_trace"]), ascii=True):
            if i % lr_step == 0:
                self.q_function.set_lr(base_lr * 0.1 ** (i // lr_step))
            trace_i = self.train_one(max_step, step_size, s=start)
            trace.append(trace_i)
            #self.epsilon -= (0.8-0.2)/max_time
            if i % self.save_epoch == 0:
                self.save_weight("weight_%05d.npy"%(i))
            f.write("Trace %4d: Delta: %f\n"%(i, self.q_function.get_delta()))
        self.save_weight("weight_latest.npy")
        self.save_weight("weight_%05d.npy"%(i))
        f.write("Total Time: %f s"%(time.time() - start_time))
        f.close()
        np.save(os.path.join(self.save_dir, "train_trace.npy"), np.stack(trace))

    def test(self, s=np.asarray([-np.pi, 0]), max_time=5000):
        self.is_train = False
        trace = []
        actions = []

        for i in range(max_time):
            a = self.take_action(s)
            s_1 = getNextStatus(s, a)
            trace.append(s[0])
            actions.append(a)
            s = s_1
        return trace, actions

    def save_weight(self, filename):
        weight = self.q_function.get_weight()
        save_name = os.path.join(self.save_dir, filename)
        np.save(save_name, weight)

    def load_weight(self, filename):
        self.q_function.load_weight(filename)