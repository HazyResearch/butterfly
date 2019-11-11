import os
import sys

for i in range(3):
  os.system(f'python no_ray.py ' + ' '.join(sys.argv[1:]))
