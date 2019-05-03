import torch
import numpy
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import argh

import my_sorting_model
import my_sinkhorn_ops


dir_path = os.path.dirname(os.path.realpath(__file__))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_random_batch(batch_size, n_numbers, prob_inc, samples_per_num):
    train_ordered, train_random, train_hard_perms = my_sinkhorn_ops.my_sample_uniform_and_order(batch_size, n_numbers, prob_inc)
    train_ordered = train_ordered.to(device)
    train_random = train_random.to(device)
    train_hard_perms = train_hard_perms.to(device)

    # tiled variables, to compare to many permutations
    train_ordered_tiled = train_ordered.repeat(samples_per_num, 1).unsqueeze(-1)
    train_random_tiled = train_random.repeat(samples_per_num, 1).unsqueeze(-1)

    return train_ordered, train_random, train_hard_perms, train_ordered_tiled, train_random_tiled



def inv_soft_perms_flattened(soft_perms_inf):
    n_numbers = soft_perms_inf.size(-1)
    inv_soft_perms = torch.transpose(soft_perms_inf, 2, 3)
    inv_soft_perms = torch.transpose(inv_soft_perms, 0, 1)

    inv_soft_perms_flat = inv_soft_perms.view(-1, n_numbers, n_numbers)
    return inv_soft_perms_flat

def build_l2s_loss(ordered_tiled, random_tiled, soft_perms_inf, n_numbers):
    """Builds loss tensor with soft permutations, for training."""
    '''Am not using htis function explicitly in the training, decided to incorporate it inside the training code itself.
    Keeping for reference'''

    print("soft_perms_inf size", soft_perms_inf.size())
    inv_soft_perms = torch.transpose(soft_perms_inf, 2, 3)
    inv_soft_perms = torch.transpose(inv_soft_perms, 0, 1)

    inv_soft_perms_flat = inv_soft_perms.view(-1, n_numbers, n_numbers)

    ordered_tiled = ordered_tiled.view(-1, n_numbers, 1)

    random_tiled = random_tiled.view(-1, n_numbers, 1)

    # squared l2 loss
    diff = ordered_tiled - torch.matmul(inv_soft_perms_flat, random_tiled)
    l2s_diff = torch.mean(torch.mul(diff, diff))

    print("l2s_diff", l2s_diff)

def train_model(n_numbers       = 50,
                lr              = 0.1,
                temperature     = 1.0,
                batch_size      = 500,
                prob_inc        = 1.0,
                samples_per_num = 5,
                n_iter_sinkhorn = 20,
                n_units         = 32,
                noise_factor    = 1.0,
                optimizer       = 'adam',
                keep_prob       = 1.,
                num_iters       = 500,
                n_epochs        = 500,
                fixed_data      = True):

    # Create the neural network
    model = my_sorting_model.Sinkhorn_Net(latent_dim= n_units, output_dim= n_numbers, dropout_prob = 1. - keep_prob)
    model.to(device)
    model.train()

    # count number of parameters
    n_params = 0
    for p in model.parameters():
        n_params += numpy.prod(p.size())
    print('# of parameters: {}'.format(n_params))

    # We use mean square error loss here.
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8)

    # Start training (old train_model function)
    train_ordered, train_random, train_hard_perms, train_ordered_tiled, train_random_tiled = make_random_batch(batch_size, n_numbers, prob_inc, samples_per_num)

    loss_history = []
    epoch_history = []

    for epoch in range(n_epochs):
        if not fixed_data:
            train_ordered, train_random, train_hard_perms, train_ordered_tiled, train_random_tiled = make_random_batch(batch_size, n_numbers, prob_inc)


        optimizer.zero_grad()
        #obtain log alpha
        log_alpha = model(train_random)
        #apply the gumbel sinkhorn on log alpha
        soft_perms_inf, log_alpha_w_noise = my_sinkhorn_ops.my_gumbel_sinkhorn(log_alpha, temperature, samples_per_num, noise_factor,  n_iter_sinkhorn, squeeze=False)

        inv_soft_perms_flat = inv_soft_perms_flattened(soft_perms_inf)

        loss= criterion(train_ordered_tiled, torch.matmul(inv_soft_perms_flat, train_random_tiled))

        loss.backward()
        optimizer.step()

        epoch_history.append(epoch)
        loss_history.append(loss.item())

        # Update the progress bar.
        print("Epoch {0:03d}: l2 loss={1:.4f}".format(epoch + 1, loss_history[-1]))
    #save the model for evaluation
    torch.save(model.state_dict(), os.path.join(dir_path, 'trained_model'))
    print('Training completed')
    # return loss_history, epoch_history

    ###### Done training


    plt.plot(epoch_history, loss_history)

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig("training_loss.png")
    plt.show()


if __name__ == '__main__':
    _parser = argh.ArghParser()
    _parser.set_default_command(train_model)
    _parser.dispatch()
