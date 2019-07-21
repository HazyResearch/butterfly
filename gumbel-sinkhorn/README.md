# Learning-Gumbel-Sinkhorn-Permutations
LEARNING LATENT PERMUTATIONS WITH GUMBEL SINKHORN NETWORKS IMPLEMENTATION WITH PYTORCH

The algorithm is based on the paper LEARNING LATENT PERMUTATIONS WITH GUMBEL-SINKHORN NETWORKS [https://arxiv.org/pdf/1802.08665.pdf] and their reference tensorflow implementation [https://github.com/google/gumbel_sinkhorn]

Sinkhorn network is a supervised method for learning to reconstruct a scrambled object X˜ (input)
given several training examples (X, X˜). By applying some non-linear transformations, a Sinkhorn network richly parameterizes the mapping between X˜ and the permutation P that once applied to X˜, will allow to reconstruct the original object as Xrec = P<sup>-1</sup>X.
The high level architecture is depicted below:
![architecture](https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch/blob/master/sinkhorn%20gumbel%20network%20architecture.png)

## How to run this code:
### Evaluate
If you want to only evaluate it, you may use the already trained model 'trained_model' and run my_sinkhorn_eval.py
You may change the number of sampled numbers by changing samples_per_num. (The paper tested for samples_per_num 5,10,15,80,100,120).
The following metrics will be calculated:
1. L1 Loss: mean absolution difference between sorted input and the reconstructed input based on hard learned permutations.
2. L2 Loss: mean squared difference between sorted input and the reconstructed input based on hard learned permutations.
3. Prop. wrong: the proportion errors in sorting
4. Prop. any wrong: the proportion of sequences where there was at least one error
5. Kendall's tau: In short, a measure of the similarity between the correct ordering of the input and the ordering learned by the net.

When run on a test set of numbers drawn from a standard uniform distribution, you can see that the net learned a perfectly correct sorting!

|               | N=5           | N=10          | N=15          | N=80          | N=100         | N=120         |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
|L1 loss        | .0            | .0            | .0            | .0            | .0            | .0            |
|L2 loss        | .0            | .0            | .0            | .0            | .0            | .0            |
|Prop. wrong    | .0            | .0            | .0            | .0            | .0            | .0            |
|Prop. any wrong| .0            | .0            | .0            | .0            | .0            | .0            |
|Kendall's tau  | 1.            | 1.            | 1.            | 1.            | 1.            | 1.            |


### Train
If you want to retrain the model run my_sorting_train.py
It is currently tuned with a specific set of hyper parameters, as you can see.

The loss is the mean squared reconstruction error between the correct ordering and the ordering obtained by applying the soft permutations on the random input numbers drawn iid from a standard uniform distribution.
The loss function graph is plotted below:

![Training Loss](https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch/blob/master/training_loss.png)

