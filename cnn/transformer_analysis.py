import pickle
import json
from pathlib import Path
import numpy as np

butterfly_bleu = [32.99, 33.8, 34.32, 34.3, 34.23, 34.1]
lr_bleu = [30.05, 32.71, 33.6, 33.08, 34.15, 34.3]
sparse_bleu = [34.08, 34.31, 34.39, 34.49, 34.586667, 34.3]
param_fraction = np.array([9, 18, 36, 72, 108, 128]) / 128

import matplotlib.pyplot as plt
plt.switch_backend('agg')

markers = ['o', 'v', 'D', 'p', 's', '>']
colors = ['red', 'orange', 'green', 'blue', 'black']

plt.plot(param_fraction, butterfly_bleu, marker=markers[0], color=colors[0], label='Butterfly')
plt.plot(param_fraction, lr_bleu, marker=markers[1], color=colors[1], label='Low-rank')
plt.plot(np.concatenate(([4.375/128], param_fraction)), np.concatenate(([32.84666666667], sparse_bleu)), marker=markers[3], color=colors[3], linestyle=':', label='Sparse (values only)')
sparse_indices = np.concatenate(([9/128], param_fraction[:-2]*2 + 1/512))
sparse_indices_bleu = np.concatenate(([32.84666666667], sparse_bleu[:-2]))
plt.plot(sparse_indices, sparse_indices_bleu, marker=markers[3], color='purple', label='Sparse (values+indices)')  # TODO make sure correct. Note last point omitted for aesthetics
# plt.plot(mobilenet_numparams / toeplitzlike_numparams[1:], all_accuracy[2, 1:], marker=markers[2], color=colors[2], label='Toeplitz-like')
# plt.plot(mobilenet_numparams / resnet_numparams, resnet_accuracy, marker=markers[3], color=colors[3], label='Resnet18')
# plt.plot(1, butterfly_bleu, marker=markers[4], color=colors[4], label='MobileNet')
# plt.axhline(butterfly_bleu, color='black', linewidth=3)
ax = plt.gca()
# ax.text(0.55, butterfly_bleu + 0.005, 'MobileNet 3.2M', color=colors[4])
# ax.text(0.55 * mobilenet_numparams / resnet_numparams, resnet_accuracy - 0.0075, 'ResNet18 11M', color=colors[3])
# ax.text(0.55 * mobilenet_numparams / all_params[6], all_accuracy[0, 6] + 0.005, 'Structured 0.6M', color=colors[0])
# plt.xscale('log')
plt.xlabel('Fraction query/key projection params vs standard attention', fontsize=14)
plt.ylabel('Test BLEU', fontsize=14)
# plt.xticks([0.3, 1, 2, 5, 10, 25], [0.3, 1, 2, 5, 10, 25])
# plt.xticks([1, 2, 5, 10, 25], [1, 2, 5, 10, 25])
plt.xlim([0, 1.19])
plt.ylim([29.75, 35])

# plt.legend(['Butterfly', 'Circulant', 'Toeplitz-like'])
plt.legend(['Kaleidoscope', 'Low-rank', 'Sparse (values only)', 'Sparse (values+indices)'])
plt.savefig('transformer_compression.pdf', bbox_inches='tight')
