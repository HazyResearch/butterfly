import pickle
from pathlib import Path
import numpy as np

result_dir = 'results_new'
experiment_names = []
experiment_names += [[f'dft_factorization_TrainableBP_True_{size}' for size in [8, 16, 32, 64, 128, 256, 512, 1024]]]
experiment_names += [[f'dct_factorization_TrainableBPP_True_{size}' for size in [8, 16, 32, 64, 128, 256, 512, 1024]]]
experiment_names += [[f'dst_factorization_TrainableBPP_True_{size}' for size in [8, 16, 32, 64, 128, 256, 512, 1024]]]
experiment_names += [[f'convolution_factorization_TrainableBPBP_True_{size}' for size in [8, 16, 32, 64, 128, 256, 512, 1024]]]
experiment_names += [[f'hadamard_factorization_TrainableBP_True_{size}' for size in [8, 16, 32, 64, 128, 256, 512, 1024]]]
experiment_names += [[f'hartley_factorization_TrainableBP_True_{size}' for size in [8, 16, 32, 64, 128, 256, 512, 1024]]]
experiment_names += [[f'legendre_factorization_TrainableBP_True_{size}' for size in [8, 16, 32, 64, 128, 256, 512, 1024]]]
experiment_names += [[f'randn_factorization_TrainableBP_True_{size}' for size in [8, 16, 32, 64, 128, 256, 512, 1024]]]

all_rmse = []
for experiment_names_ in experiment_names:
    print(experiment_names_[0])
    best_loss = []
    best_polished_loss = []
    for experiment_name in experiment_names_:
        checkpoint_path = Path(result_dir) / experiment_name / 'trial.pkl'
        with checkpoint_path.open('rb') as f:
            trials = pickle.load(f)
        losses = [-trial.last_result['negative_loss'] for trial in trials]
        polished_losses = [-trial.last_result.get('polished_negative_loss', float('-inf')) for trial in trials]
        # best_loss.append(min(losses))
        best_loss.append(np.sort(losses)[0])  # to deal with NaN
        best_polished_loss.append(np.sort(polished_losses)[0])  # to deal with NaN
        # print(np.array(losses))
        # print(np.sort(losses))
        # best_trial = max(trials, key=lambda trial: trial.last_result['negative_loss'])
        # train_model = best_trial._get_trainable_cls()(best_trial.config)
        # train_model = TrainableHadamardFactorFixedOrder(best_trial.config)
        # train_model = TrainableHadamardFactorSoftmax(best_trial.config)
        # train_model = TrainableHadamardFactorSparsemax(best_trial.config)
        # train_model.restore(str(Path(best_trial.logdir) / best_trial._checkpoint.value))
        # model = train_model.model
    # best_rmse = np.sqrt(best_loss)
    # print(best_rmse)
    print(np.sqrt(best_polished_loss))
    all_rmse.append(np.sqrt(best_polished_loss))

print(np.array(all_rmse))
transform_names = ['DFT', 'DCT', 'DST', 'Conv', 'Hadamard', 'Hartley', 'Legendre', 'Rand']

import pickle

with open('rmse.pkl', 'wb') as f:
    pickle.dump({'names': transform_names, 'rmse': all_rmse}, f)
