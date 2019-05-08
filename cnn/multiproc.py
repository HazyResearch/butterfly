# modified from https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/multiproc.py

import argparse
import torch
import sys
import subprocess
from pathlib import Path

import random
argslist = list(sys.argv)[1:]
world_size = torch.cuda.device_count()

if '--world-size' in argslist:
    argslist[argslist.index('--world-size')+1] = str(world_size)
else:
    argslist.append('--world-size')
    argslist.append(str(world_size))

save_dir = Path.cwd()
if '--save-dir' in argslist:
    save_dir = argslist[argslist.index('--save-dir')+1]

workers = []

port = random.randint(10000, 20000)
argslist.append('--distributed-init-method')
argslist.append('tcp://localhost:{port}'.format(port=port))
for i in range(world_size):
    if '--rank' in argslist:
        argslist[argslist.index('--rank')+1] = str(i)
    else:
        argslist.append('--rank')
        argslist.append(str(i))
    stdout = None if i == 0 else open(f'{save_dir}/GPU_{i}.log', "w")
    p = subprocess.Popen([str(sys.executable)]+argslist, stdout=stdout)
    workers.append(p)

for p in workers: p.wait()

