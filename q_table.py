import numpy as np

class DiscreteQTable(object):
    def __init__(self, table_args):
        grid = table_args["grid"]
        self.table = np.zeros((*grid, 3))
        self.lr = table_args["lr"]
        self.lambda_t = table_args["lambda"]
        self.gamma = 0.98
        self.step_size = [np.pi * 2 / grid[0], np.pi * 30 / grid[1]]

        self.count, self.sum_delta = 0, 0.0
    
    def getIndex(self, s):
        return (int((s[0] + np.pi) // self.step_size[0]), int((s[1] + 15 * np.pi) // self.step_size[1]))

    def greedy_action(self, s):
        index = self.getIndex(s)
        #print(index)
        return np.argmax(self.table[index])

    def train_sarsa(self, s, a, r, s_1, a_1):
        index_0 = (*self.getIndex(s), a)
        index_1 = (*self.getIndex(s_1), a_1)
        delta = r + self.gamma*self.table[index_1] - self.table[index_0]
        self.sum_delta += delta ** 2
        self.count += 1
        if self.lambda_t:
            self.et = self.gamma * self.lambda_t *self.et
            self.et[index_0] += 1
            self.table += self.lr * delta * self.et
            return
        self.table[index_0] += self.lr * delta
    
    def reset_et(self):
        self.et = np.zeros(self.table.shape)

    def get_weight(self):
        return self.table

    def load_weight(self, filename):
        self.table = np.load(filename)

    def get_delta(self):
        avg_delta = self.sum_delta/self.count
        self.count, self.sum_delta = 0, 0.
        return avg_delta

    def set_lr(self, lr):
        self.lr = lr