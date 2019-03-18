import pickle
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from matplotlib.colors import LinearSegmentedColormap

with open('rmse.pkl', 'rb') as f:
    data = pickle.load(f)
transform_names = data['names']
our_rmse = np.array(data['rmse'])
our_rmse = np.delete(our_rmse, -2, axis=0)

with open('sparse_rmse.pkl', 'rb') as f:
    sparse_rmse = pickle.load(f)
sparse_rmse = np.delete(sparse_rmse, -2, axis=0)
with open('lr_rmse.pkl', 'rb') as f:
    lr_rmse = pickle.load(f)
lr_rmse = np.delete(lr_rmse, -2, axis=0)

# with open('mse_robust_pca.pkl', 'rb') as f:
#     sparse_lr_rmse = pickle.load(f)
# It's always an option (depending on parameter) to get just one sparse matrix, or just one low rank, so we take minimum here
# sparse_lr_rmse = np.minimum(sparse_lr_rmse, sparse_rmse)
# sparse_lr_rmse = np.minimum(sparse_lr_rmse, lr_rmse)
sparse_lr_rmse = np.minimum(sparse_rmse, lr_rmse)

# For LaTeX
print(" \\\\\n".join([" & ".join(map('{0:.1e}'.format, line)) for line in our_rmse]))

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

rmses = [our_rmse, sparse_rmse, lr_rmse, sparse_lr_rmse]
titles = ['Butterfly', 'Sparse', 'Low rank', 'Sparse + Low rank']

first = True
fig, axes = plt.subplots(nrows=1, ncols=4)
for rmse, ax, title in zip(rmses, axes.flat, titles):
    # im = ax.imshow(np.log10(rmse), interpolation='none', vmin=-4, vmax=0, cmap=cmap);
    im = ax.imshow(np.log10(rmse), interpolation='none', vmin=-4, vmax=0, cmap='bwr');
    ax.set_title(title, fontsize=10)
    # curr_ax = fig.gca();
    # curr_ax = fig.gca();

    # Major ticks
    ax.set_xticks(range(8));
    ax.set_yticks(range(8));

    # Labels for major ticks
    ax.set_xticklabels(['N=8', '16', '32', '64', '128', '256', '512', '1024'], rotation=270, fontsize=8);
    if first:
        # ax.set_yticklabels(['DFT', 'DCT', 'DST', 'Conv', 'Hadamard', 'Hartley', 'Legendre', 'Hilbert', 'Randn']);
        ax.set_yticklabels(['DFT', 'DCT', 'DST', 'Conv', 'Hadamard', 'Hartley', 'Legendre', 'Randn'], fontsize=8);
        first = False
    else:
        ax.set_yticklabels([]);

    # Minor ticks
    ax.set_xticks(np.arange(-.5, 8, 1), minor=True);
    ax.set_yticks(np.arange(-.5, 8, 1), minor=True);

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

# plt.figure()
# # im = plt.imshow(np.log10(rmse), interpolation='none', vmin=-4, vmax=0, aspect='equal');
# # im = plt.imshow(np.log10(rmse), interpolation='none', vmin=-4, vmax=0, cmap='Accent');
# im = plt.imshow(np.log10(rmse), interpolation='none', vmin=-4, vmax=0, cmap=cmap);
# ax = plt.gca();
# ax = plt.gca();

# # Major ticks
# ax.set_xticks(range(8));
# ax.set_yticks(range(9));

# # Labels for major ticks
# ax.set_xticklabels(['N=8', '16', '32', '64', '128', '256', '512', '1024']);
# ax.set_yticklabels(['DFT', 'DCT', 'DST', 'Conv', 'Hadamard', 'Hartley', 'Legendre', 'Hilbert', 'Randn']);

# # Minor ticks
# ax.set_xticks(np.arange(-.5, 8, 1), minor=True);
# ax.set_yticks(np.arange(-.5, 9, 1), minor=True);

# # Gridlines based on minor ticks
# ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

# cbar = plt.colorbar(orientation='horizontal', ticks=[-4, -3, -2, -1, 0])
fig.subplots_adjust(bottom=-0.2, wspace=0.5)
cbar_ax = fig.add_axes([0.25, 0.11, 0.5, 0.02])
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', ticks=[-4, -3, -2, -1, 0])
cbar.ax.set_xticklabels(['1e-4', '1e-3', '1e-2', '1e-1', '1e0'], fontsize=8)

plt.savefig('heatmap.pdf', bbox_inches='tight')
