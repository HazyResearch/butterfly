import torch
import my_sorting_model
import numpy
import torch.nn as nn
import my_sinkhorn_ops
import os

dir_path = os.path.dirname(os.path.realpath(__file__))

n_numbers = 50
temperature = 1.0
batch_size = 2
prob_inc = 1.0
samples_per_num = 5
n_iter_sinkhorn = 10
n_units = 32
noise_factor = 0.0
keep_prob = 1.
dropout_prob = 1. - keep_prob

# Test process
def test_model(model, batch_size, n_numbers, prob_inc):
    # generate test set
    # validation variables
    test_ordered, test_random, test_hard_perms = my_sinkhorn_ops.my_sample_uniform_and_order(batch_size, n_numbers,
                                                                                             prob_inc)
    # tiled variables, to compare to many permutations
    test_ordered_tiled = test_ordered.repeat(samples_per_num, 1)
    test_random_tiled = test_random.repeat(samples_per_num, 1)

    test_ordered_tiled = test_ordered_tiled.view(-1, n_numbers, 1)
    test_random_tiled = test_random_tiled.view(-1, n_numbers, 1)

    # Testing phase
    model.eval()

    x_in, perms = test_random, test_hard_perms
    y_in = test_ordered

    if is_cuda_available:
        x_in, y_in = x_in.cuda(), y_in.cuda()
        test_ordered_tiled = test_ordered_tiled.cuda()
        perms = perms.cuda()

    #obtain log alpha
    log_alpha = model(x_in)
    #apply the gumbel sinkhorn on log alpha
    soft_perms_inf, log_alpha_w_noise = my_sinkhorn_ops.my_gumbel_sinkhorn(log_alpha, temperature, samples_per_num, noise_factor,  n_iter_sinkhorn, squeeze=False)

    #n_correct_pred += compute_acc(vecmat2perm2x2(outputs), perms, False).data[0]
    l1_loss, l2_loss, prop_wrong, prop_any_wrong, kendall_tau = build_hard_losses(log_alpha_w_noise, test_random_tiled, test_ordered_tiled, perms, n_numbers)
    print("samples_per_num", samples_per_num)
    print("l1 loss", l1_loss)
    print("l2 loss",l2_loss)
    print("prop_wrong", prop_wrong)
    print("prop any wrong", prop_any_wrong)
    print("Kendall's tau", kendall_tau)
    print('Test completed')


def build_hard_losses(log_alpha_w_noise, random_tiled, ordered_tiled, hard_perms, n_numbers):
    """Losses based on hard reconstruction. Only for evaluation.
    Doubly stochastic matrices are rounded with
    the matching function.
    """
    log_alpha_w_noise_flat = torch.transpose(log_alpha_w_noise, 0, 1)
    log_alpha_w_noise_flat = log_alpha_w_noise_flat.view(-1, n_numbers, n_numbers)

    hard_perms_inf = my_sinkhorn_ops.my_matching(log_alpha_w_noise_flat)
    inverse_hard_perms_inf = my_sinkhorn_ops.my_invert_listperm(hard_perms_inf)
    hard_perms_tiled = hard_perms.repeat(samples_per_num, 1)

    # The 3D output of permute_batch_split must be squeezed
    ordered_inf_tiled = my_sinkhorn_ops.my_permute_batch_split(random_tiled, inverse_hard_perms_inf)
    ordered_inf_tiled = ordered_inf_tiled.view(-1, n_numbers)

    #my addition
    ordered_tiled = ordered_tiled.view(-1, n_numbers)

    l_diff = ordered_tiled - ordered_inf_tiled
    l1_diff = torch.mean(torch.abs(l_diff))
    l2_diff = torch.mean(torch.mul(l_diff, l_diff))

    diff_perms = torch.abs(hard_perms_tiled - inverse_hard_perms_inf)
    diff_perms = diff_perms.type(torch.float32)

    prop_wrong = -torch.mean(torch.sign(-diff_perms))
    prop_any_wrong = -torch.mean(torch.sign(-torch.sum(diff_perms, dim = 1)))
    kendall_tau = torch.mean(my_sinkhorn_ops.my_kendall_tau(hard_perms_tiled, inverse_hard_perms_inf))

    return l1_diff, l2_diff, prop_wrong, prop_any_wrong, kendall_tau


#load the trained model
model = my_sorting_model.Sinkhorn_Net(latent_dim= n_units, output_dim= n_numbers, dropout_prob = dropout_prob)
model.load_state_dict(torch.load(os.path.join(dir_path, 'trained_model')))
is_cuda_available = torch.cuda.is_available();
if is_cuda_available:
    model.cuda()

#test the model
test_model(model, batch_size, n_numbers, prob_inc)