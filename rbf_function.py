import numpy as np

EPS = 1E-9

class RBF_function(object):
    def __init__(self, rbf_args):
        bins = rbf_args['bins']
        self.w = np.random.randn(3, bins**2)
        self.lr = rbf_args['lr']#0.001
        self.var = rbf_args['var']# 0.1
        self.lambda_t = rbf_args["lambda"]
        self.gamma = 0.98
        
        xl, yl = np.arange(-1, 1-EPS, 2/bins), np.arange(-1, 1, 2/bins)
        xl, yl = np.meshgrid(xl, yl)
        self.xl, self.yl = xl.flatten(), yl.flatten()

        self.count, self.sum_delta = 0, 0.0
    
    def __call__(self, s, a):
        s = self._getFeature(s)
        return np.dot(self.w[a], s)

    def _normalize(self, s):
        s[0] /= np.pi
        s[1] /= 15 * np.pi
        return s
    
    def _rbffunction(self, s):
        s = self._normalize(s)
        #for i, (x, y) in enumerate(zip(xl, yl)):
        x_dis, y_dis = self.xl - s[0], self.yl - s[1]
        x_dis = x_dis + 2 * (x_dis < -1) - 2 * (x_dis > 1)
        y_dis = y_dis + 2 * (y_dis < -1) - 2 * (y_dis > 1)
        feature = np.exp(-x_dis**2/2/self.var-y_dis**2/2/self.var)
        # feature = np.concatenate([np.exp(-x_dis**2/1), np.exp(-(self.yl-s[1])**2/1)])
        #feature = feature / np.sum(feature)
        #feature = feature - 0.5
        return feature
    
    def _getFeature(self, s):
        return self._rbffunction(s.copy())
    
    def greedy_action(self, s):
        s = self._getFeature(s)
        r = np.dot(self.w, s)
        return np.argmax(r)
    
    def reset_et(self):
        self.et = np.zeros(self.w.shape)

    def train_sarsa(self, s, a, r, s_1, a_1):
        q, q_1 = self(s, a), self(s_1, a_1)
        delta = r + self.gamma * q_1 - q
        self.sum_delta += delta ** 2
        self.count += 1

        s = self._getFeature(s)
        if self.lambda_t:
            self.et = self.gamma * self.lambda_t *self.et
            self.et[a] += s
            self.w += self.lr * delta * self.et
            return
        self.w[a] += self.lr * delta * s
    
    def train_q_learning(self, s, a, r, s_1):
        s_1 = self._getFeature(s_1)
        q_1 = np.max(np.dot(self.w, s_1))
        s = self._getFeature(s)
        q = np.dot(self.w[a], s)
        delta = r + self.gamma * q_1 - q
        self.w[a] += self.lr * delta * s

    def get_weight(self):
        return self.w

    def load_weight(self, filename):
        self.w = np.load(filename)

    def get_delta(self):
        avg_delta = self.sum_delta/self.count
        self.count, self.sum_delta = 0, 0.
        return avg_delta

    def set_lr(self, lr):
        self.lr = lr