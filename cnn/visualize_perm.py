import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os, sys

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# Add to $PYTHONPATH in addition to sys.path so that ray workers can see
os.environ['PYTHONPATH'] = project_root + ":" + os.environ.get('PYTHONPATH', '')
import dataset_utils
import model_utils

# perm_path = 'sinkhorn.samples_40.epochs_400'
# perm_path = 'sinkhorn.was237'
# method = 'sinkhorn'
# perm_path = 'butterfly-samples16-anneal.63-temp.06-lr.0008'
perm_path = 'T5.2'
method = 'butterfly'

if __name__ == '__main__':
    original_dataset = {'name': 'PPCIFAR10', 'batch': 8, 'transform': 'original'}
    permuted_dataset = {'name': 'PPCIFAR10', 'batch': 8, 'transform': 'permute'}
    normalize_dataset = {'name': 'PPCIFAR10', 'batch': 8, 'transform': 'normalize'}
    training_dataset = {'name': 'PPCIFAR10', 'batch': 8}
    torch.manual_seed(0)
    orig_train_loader, orig_test_loader = dataset_utils.get_dataset(original_dataset)
    torch.manual_seed(0)
    perm_train_loader, perm_test_loader = dataset_utils.get_dataset(permuted_dataset)
    torch.manual_seed(0)
    norm_train_loader, norm_test_loader = dataset_utils.get_dataset(normalize_dataset)
    torch.manual_seed(0)
    train_train_loader, train_test_loader = dataset_utils.get_dataset(training_dataset)

    def imshow(img, name):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        # plt.show()
        plt.savefig(name, bbox_inches='tight')

    # get some random training images
    torch.manual_seed(0)
    dataiter = iter(orig_train_loader)
    orig_images, labels = dataiter.next()
    torch.manual_seed(0)
    dataiter = iter(perm_train_loader)
    perm_images, _ = dataiter.next()
    torch.manual_seed(0)
    dataiter = iter(norm_train_loader)
    norm_images, _ = dataiter.next()
    torch.manual_seed(0)
    dataiter = iter(train_train_loader)
    train_images, _ = dataiter.next()

    # show images
    # imshow(torchvision.utils.make_grid(images), 'examples')

    perm_path = 'saved_perms/' + perm_path

    # model = model_utils.get_model({'name': 'Permutation', 'args': {'method': method, 'stochastic':True, 'param': 'logit', 'temp': 0.1}})
    # # model = model_utils.get_model({'name': 'Permutation', 'args': {'method': method}})
    # model.load_state_dict(torch.load(perm_path))

    # New version:
    saved_model = torch.load(perm_path)
    model = model_utils.get_model(saved_model['args'])
    model.load_state_dict(saved_model['state'])


    # TODO: nsamples should be able to be passed into forward pass
    for p in model.permute:
        p.samples = 1
    # breakpoint()
    # print(model)
    model.eval()
    with torch.no_grad():
        # breakpoint()
        sample_output = model(perm_images, perm='sample')
        mean_output = model(perm_images, perm='mean')
        mle_output = model(perm_images, perm='mle')
    # imshow(torchvision.utils.make_grid(mean_output), 'examples_mean')
    # imshow(torchvision.utils.make_grid(mle_output), 'examples_mle')
    all_images = torch.cat([orig_images, perm_images, norm_images, train_images, sample_output, mean_output, mle_output], dim=0)
    imshow(torchvision.utils.make_grid(all_images), 'all_examples')

