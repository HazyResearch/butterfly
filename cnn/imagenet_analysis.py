import pickle
import json
from pathlib import Path
import numpy as np

# butterfly_acc = [56.4, 65.0, 70.1, 71.2]
# butterfly_param = [0.70, 1.54, 3.62, 4.04]
# Butterfly w/ channel pooling
butterfly_smpool_acc = [54.027, 62.840, 68.418]
butterfly_smpool_param = np.array([439688, 1024808, 2597480])  / 1e6
# Butterfly w/ compressed softmax
# butterfly_smstruct_acc = [59.0, 66.1, 70.5]
# butterfly_smstruct_param = [0.60, 1.77, 5.69]
# width_acc = [51.6, 63.9, 70.9]
# width_param = [0.47, 1.33, 4.23]
width_acc = [51.971, 61.770, 67.799]
width_param = np.array([470072, 1019304, 2585560]) / 1e6
lowrank_acc = [47.248, 56.064, 62.727]
lowrank_param = np.array([426376, 997160, 2540136]) / 1e6
sparse_ind_acc = [56.145]
sparse_ind_param = np.array([439688+84992]) / 1e6

import matplotlib.pyplot as plt
plt.switch_backend('agg')

markers = ['o', 'v', 'D', 'p', 's', '>']
colors = ['red', 'orange', 'green', 'blue', 'black']

# plt.plot(butterfly_param, butterfly_acc, marker=markers[0], color=colors[0], label='Butterfly')
plt.plot(width_param, width_acc, marker=markers[2], color=colors[2], label='Reducing width')
plt.plot(butterfly_smpool_param, butterfly_smpool_acc, marker=markers[0], color=colors[0], label='Kaleidoscope')
plt.plot(lowrank_param, lowrank_acc, marker=markers[1], color=colors[1], label='Low-rank')
plt.plot(sparse_ind_param, sparse_ind_acc, marker=markers[3], color=colors[3], label='Sparse (values+indices)')
# plt.plot(butterfly_smstruct_param, butterfly_smstruct_acc, marker=markers[3], color=colors[3], label='Butterfly w/ smstruct')
# plt.plot(mobilenet_numparams / toeplitzlike_numparams[1:], all_accuracy[2, 1:], marker=markers[2], color=colors[2], label='Toeplitz-like')
# plt.plot(mobilenet_numparams / resnet_numparams, resnet_accuracy, marker=markers[3], color=colors[3], label='Resnet18')
# plt.plot(1, butterfly_acc, marker=markers[4], color=colors[4], label='MobileNet')
# plt.axhline(butterfly_acc, color='black', linewidth=3)
ax = plt.gca()
# ax.text(0.55, butterfly_acc + 0.005, 'MobileNet 3.2M', color=colors[4])
# ax.text(0.55 * mobilenet_numparams / resnet_numparams, resnet_accuracy - 0.0075, 'ResNet18 11M', color=colors[3])
# ax.text(0.55 * mobilenet_numparams / all_params[6], all_accuracy[0, 6] + 0.005, 'Structured 0.6M', color=colors[0])
# plt.xscale('log')
plt.xlabel('Number of parameters (millions)', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)
# plt.xticks([0.3, 1, 2, 5, 10, 25], [0.3, 1, 2, 5, 10, 25])
# plt.xticks([1, 2, 5, 10, 25], [1, 2, 5, 10, 25])

# plt.legend(['Butterfly', 'Circulant', 'Toeplitz-like'])
# plt.legend(['Butterfly', 'Reducing width'])
plt.legend()
plt.savefig('imagenet_compression.pdf', bbox_inches='tight')
