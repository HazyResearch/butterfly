import pickle
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.colors import LinearSegmentedColormap

with open('rmse.pkl', 'rb') as f:
    data = pickle.load(f)
transform_names = data['names']
rmse = np.array(data['rmse'])

# For LaTeX
print(" \\\\\n".join([" & ".join(map('{0:.1e}'.format, line)) for line in rmse]))

# Red-green colormap
cdict = {'red':  ((0.0, 0.0, 0.0),
                 (1/6., 0.0, 0.0),
                 (1/2., 0.8, 1.0),
                 (5/6., 1.0, 1.0),
                 (1.0, 0.4, 1.0)),

             'green':  ((0.0, 0.0, 0.4),
                 (1/6., 1.0, 1.0),
                 (1/2., 1.0, 0.8),
                 (5/6., 0.0, 0.0),
                 (1.0, 0.0, 0.0)),

             'blue': ((0.0, 0.0, 0.0),
                 (1/6., 0.0, 0.0),
                 (1/2., 0.9, 0.9),
                 (5/6., 0.0, 0.0),
                 (1.0, 0.0, 0.0))

        }
cmap=LinearSegmentedColormap('rg',cdict, N=256)

plt.figure()
# im = plt.imshow(np.log10(rmse), interpolation='none', vmin=-4, vmax=0, aspect='equal');
# im = plt.imshow(np.log10(rmse), interpolation='none', vmin=-4, vmax=0, cmap='Accent');
im = plt.imshow(np.log10(rmse), interpolation='none', vmin=-4, vmax=0, cmap=cmap);
ax = plt.gca();
ax = plt.gca();

# Major ticks
ax.set_xticks(range(8));
ax.set_yticks(range(9));

# Labels for major ticks
ax.set_xticklabels(['N=8', '16', '32', '64', '128', '256', '512', '1024']);
ax.set_yticklabels(['DFT', 'DCT', 'DST', 'Conv', 'Hadamard', 'Hartley', 'Legendre', 'Hilbert', 'Randn']);

# Minor ticks
ax.set_xticks(np.arange(-.5, 8, 1), minor=True);
ax.set_yticks(np.arange(-.5, 9, 1), minor=True);

# Gridlines based on minor ticks
ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

cbar = plt.colorbar(orientation='horizontal', ticks=[-4, -3, -2, -1, 0])
cbar.ax.set_xticklabels(['1e-4', '1e-3', '1e-2', '1e-1', '1e0'])

plt.savefig('heatmap.png', bbox_inches='tight')
