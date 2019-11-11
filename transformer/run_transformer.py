import os
import sys

num_trials = 1
for i in range(num_trials):
  os.system(f'python no_ray.py ' + ' '.join(sys.argv[1:]))
