import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import random
import cv2
from tqdm import tqdm

from arguments import Argument
from controller import Controller, T

def drawTrace(save_dir):
    filename = os.path.join(save_dir, 'train_trace.npy')
    train_trace = np.load(filename)
    num_trace, len_trace, _ = train_trace.shape
    idx = np.arange(0, num_trace).reshape((-1, 1))
    idx = np.repeat(idx, len_trace, axis = 1)
    pts = np.stack([idx, train_trace[:, :, 0]], axis = 2)
    plt.figure(0)
    plt.title('Training Trace (Red: Start, Blue: End)')
    plt.xlabel('training iter')
    plt.ylabel('angle(rad)')
    for i in range(0 ,len_trace, 100):
        print(255 - int(256.0* i/len_trace), int(256.0* i/len_trace))
        plt.scatter(pts[:,  i, 0], pts[:,  i, 1], c = '#%02X00%02X'%(255 - int(256.0* i/len_trace), int(256.0* i/len_trace)))
    plt.show()

def drawLoss(save_dir):
    filename = os.path.join(save_dir, 'log.txt')
    with open(filename, 'r') as f:
        lines = f.readlines()[:-1]
    lines = [float(l.replace('\n', '').split(' ')[-1]) for l in lines]
    #plt.figure(0)
    #plt.plot(np.arange(len(lines)), lines)
    #plt.show()
    return lines

def drawTest(save_dir, start = [-np.pi, 0]):
    args = Argument(os.path.join(save_dir, 'config.json'))
    controller = Controller(args)
    controller.load_weight(os.path.join(save_dir, 'weight_latest.npy'))
    trace, actions = controller.test(start, max_time=1000)
    #plt.plot(-np.abs(trace))
    #plt.show()
    return trace

def drawAction(save_dir, grid = (500, 500)):
    args = Argument(os.path.join(save_dir, 'config.json'))
    controller = Controller(args)
    controller.load_weight(os.path.join(save_dir, 'weight_latest.npy'))
    func = controller.q_function
    
    x1 = np.arange(-np.pi, np.pi, 2*np.pi/grid[0])
    y1 = np.arange(-15*np.pi, 15*np.pi, 30*np.pi/grid[1])
    x, y = np.meshgrid(x1, y1)
    status = np.stack([x,y], axis=2).reshape((-1, 2))
    features = []
    for i in status:
        features.append(func._getFeature(i))
    #print(features)
    features = np.stack(features, axis=1)
    action = np.argmax(np.dot(func.w, features).transpose(), axis=1)
    cmap = ["black", "grey", "white"]
    #plt.Rectangle((0,0), 6, 6, fill=colors[1], edgecolor=colors[1])
    colors = [cmap[i] for i in action]
    plt.scatter(x, y, c=colors)
    #for istatus, iaction in zip(status, action):
    #    print(istatus)
    #    plt.Rectangle(istatus, 2*np.pi/grid[0], 30*np.pi/grid[1], fill=colors[iaction], edgecolor=colors[iaction])


def to_percent(temp, position):
    return '%2.0f'%(100*temp) + '%'


if __name__ == "__main__":
    '''
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 40.0, (1280,720), True)
    trace = drawTest("./output/0502rbfrand01_ep2")[::5]
    for ang in tqdm(trace, ascii=True):
        frame = np.zeros((720, 1280, 3),dtype=np.uint8) + 255
        new_p = (int(640+200*np.sin(ang)), int(360-200*np.cos(ang)))
        cv2.line(frame, (640, 360), new_p, (0, 0, 0))
        cv2.circle(frame, (640, 360), 15, (0, 0, 0), -1)
        cv2.circle(frame, new_p, 5, (0, 0, 0), -1)
        out.write(frame)
    out.release()
    '''

    #table_analyse()
    #rbf_analyse()
    plt.figure(4)
    table_grid = drawTest('output/final_fixgrid10_005')
    table_rbf = drawTest('output/final_rbfrand01_ep2')
    x = np.arange(0,len(table_grid) * T, T)
    plt.plot(x, -np.abs(table_grid))
    plt.plot(x, -np.abs(table_rbf))
    plt.legend([
        "fix+grid, epsilon = 0.05",
        "rand+rbf, epsilon = 0.2"
        ], loc = 'lower right')
    plt.xlabel('time(s)')
    plt.ylabel('angle(rad)')
    plt.title('Train with fix start point')
    plt.show()
    