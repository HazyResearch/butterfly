import pickle
import json
from pathlib import Path
import numpy as np

butterfly_acc = [56.4, 65.0, 71.1]
width_acc = [51.6, 63.9, 70.9]
butterfly_param = [0.7, 1.4, 4.0]
width_param = [0.5, 1.4, 4.2]

import matplotlib.pyplot as plt
plt.switch_backend('agg')

markers = ['o', 'v', 'D', 'p', 's', '>']
colors = ['red', 'orange', 'green', 'blue', 'black']

plt.plot(butterfly_param, butterfly_acc, marker=markers[0], color=colors[0], label='Butterfly')
plt.plot(width_param, width_acc, marker=markers[2], color=colors[2], label='Reducing width')
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
plt.legend(['Butterfly', 'Reducing width'])
plt.savefig('imagenet_compression.pdf', bbox_inches='tight')
