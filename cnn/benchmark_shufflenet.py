import torch

import os, sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import time
from cnn.shufflenet_imagenet import ShuffleNet

def benchmark(nets, runs):
    # generate data in advance
    x_all =  torch.randn(runs, 1, 3, 224, 224)
    start = time.perf_counter()
    for i in range(runs):
        x = x_all[i]
        net = nets[i]
        y = net(x)
    total_time = time.perf_counter() - start
    print('Time: {:3.4f}s\n'.format(total_time))
    return total_time

def main():
    # make single threaded
    torch.set_num_threads(1)
    runs = 100

    print('Using basic ShuffleNet')
    # create a bunch of random models in advance
    nets = [ShuffleNet() for i in range(runs)]
    benchmark(nets, runs)

    print('Using perm. ShuffleNet')
    nets = [ShuffleNet(shuffle='P') for i in range(runs)]
    benchmark(nets, runs)

    print('Using BBT ShuffleNet')
    nets = [ShuffleNet(shuffle='regular_1') for i in range(runs)]
    benchmark(nets, runs)

    print('Using BBT ShuffleNet with eval')
    nets = [nets[i].eval() for i in range(runs)]
    benchmark(nets, runs)

if __name__ == "__main__":
    main()