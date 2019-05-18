import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys

dataset = 'cifar10' if len(sys.argv) < 2 else sys.argv[1]
svds_file = 'svds_{}.pkl'.format(dataset)

if os.path.exists(svds_file):
    with open(svds_file, 'rb') as pkl: spectra = pickle.load(pkl)
else:
    names = []
    tensors = []
    name_patterns = ['conv', 'shortcut', 'downsample']
    if dataset == 'cifar10':
        model = torch.load('model_optimizer.pth')['model']
        for n in model.keys():
            if 'weight' in n and any([x in n for x in name_patterns]) and not 'shortcut.1.weight' in n:
                # shortcut.1.* corresponds to the batch norm in the shortcut
                names.append(n)
                tensors.append(model[n].detach().cpu().numpy())
    elif dataset == 'imagenet':
        import torchvision.models as models
        resnet50 = False
        if resnet50:
            model = models.resnet50(pretrained=True)
        else:
            model = models.resnet18(pretrained=True)
        for n, p in model.named_parameters():
            if 'weight' in n and any([x in n for x in name_patterns]) and not 'downsample.1.weight' in n:
                names.append(n)
                tensors.append(p.detach().cpu().numpy())
    elif dataset == 'timit':
        model = torch.load('final_architecture1.pkl')['model_par']
        for n in model.keys():
            if 'ln' not in n and 'bn' not in n and 'bias' not in n:
                names.append(n)
                # Add two dummy dimensions at the end just to simplify later code
                tensors.append(model[n].detach().cpu().unsqueeze_(-1).unsqueeze_(-1).numpy())
    elif dataset == 'transformer':
        model = torch.load('checkpoint_last.pt')['model']
        name_patterns = ['in_proj', 'out_proj']
        for n in model.keys():
            if any([x in n for x in name_patterns]) and 'bias' not in n:
                names.append(n)
                tensors.append(model[n].detach().cpu())
        names_old = names
        tensors_old = tensors
        names = []
        tensors = []
        num_heads = 4
        for n, p in zip(names_old, tensors_old):
            if 'in_proj' in n:
                q_dim = p.size(0) // 3
                q_proj = p[:q_dim]
                k_proj = p[q_dim:2*q_dim]
                q_heads = q_proj.chunk(num_heads)
                k_heads = k_proj.chunk(num_heads)
                q_heads = [x.numpy().T for x in q_heads]
                k_heads = [x.numpy() for x in k_heads]
                prods = [qh @ kh for qh, kh in zip(q_heads, k_heads)]
                output = np.zeros((prods[0].shape[0], prods[0].shape[1], num_heads, 1))
                for i in range(num_heads):
                    output[:,:,i,0] = prods[i]  # put all heads on the same plot
                names.append(n)
                tensors.append(output)
    else:
        raise ValueError('Unknown dataset')
    spectra = {}
    for n, t in zip(names, tensors):
        t_spectra = []
        print('Processing {}...'.format(n))
        for a in range(t.shape[-2]):
            for b in range(t.shape[-1]):
                f = t[:,:,a,b]
                s = np.linalg.svd(f, compute_uv=False)
                t_spectra.append(s)
        spectra[n] = t_spectra
    with open(svds_file, 'wb') as pkl: pickle.dump(spectra, pkl)

plot_names = sorted(list(spectra.keys()))
if dataset == 'timit':
    plot_names.sort(key = lambda s: int("".join(filter(str.isdigit, s))))
print(plot_names)
subplot_grid_dim = int(np.ceil(np.sqrt(len(plot_names))))

figsize = 30 if dataset == 'imagenet' else 20
fig, ax = plt.subplots(int(np.ceil(len(plot_names) / subplot_grid_dim)), subplot_grid_dim, figsize=(28, 28))

for i, n in enumerate(plot_names):
    for j, s in enumerate(spectra[n]):
        xvals = np.arange(len(s)) + 1
        subplot = ax[i // subplot_grid_dim, i % subplot_grid_dim]
        subplot.plot(xvals, s, label=n + '_filter{}'.format(j))
        if len(s) <= 5:
            subplot.set_xticks(xvals)
        elif dataset != 'transformer':
            subplot.set_xticks([1, len(s)//2, len(s)])
        else:
            subplot.set_xticks([1, len(s)//4, len(s)//2, 3*len(s)//4, len(s)])
        subplot.tick_params(axis='x', labelsize=6)
        subplot.tick_params(axis='y', labelsize=6)
        subplot.set_xlabel('Index', fontsize=7)
        subplot.set_ylabel('Singular value', fontsize=7)
        subplot.set_title(n, fontsize=10)
fig.suptitle(dataset.capitalize(), fontsize=24)
fig.savefig('spectra_{}.png'.format(dataset), dpi=250)
