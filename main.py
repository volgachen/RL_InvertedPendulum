import numpy as np
import random
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

from controller import Controller
from arguments import Argument

if __name__ == "__main__":

    for item in os.listdir('./config/'):
        print('Processing', item)
        args = Argument('config/'+item)
        controller = Controller(args)
        controller.train(args.train)
        trace, actions = controller.test()