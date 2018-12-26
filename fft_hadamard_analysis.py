import pickle
from pathlib import Path
import numpy as np

result_dir = 'results'
experiment_names = [[f'Hadamard_factorization_fixed_order_{size}' for size in [8, 16, 32, 64, 128, 256]]]
experiment_names += [[f'Hadamard_factorization_softmax_{size}' for size in [8, 16, 32, 64, 128]]]
experiment_names += [[f'Hadamard_factorization_sparsemax_{size}' for size in [8, 16, 32, 64, 128]]]
experiment_names += [[f'Fft_factorization_fixed_order_{size}' for size in [8, 16, 32, 64, 128]]]
experiment_names += [[f'Fft_factorization_softmax_{size}' for size in [8, 16, 32, 64, 128]]]
experiment_names += [[f'Fft_factorization_sparsemax_{size}' for size in [8, 16, 32, 64]]]

for experiment_names_ in experiment_names:
    best_loss = []
    for experiment_name in experiment_names_:
        checkpoint_path = Path(result_dir) / experiment_name / 'trial.pkl'
        with checkpoint_path.open('rb') as f:
            trials = pickle.load(f)
        losses = [-trial.last_result['negative_loss'] for trial in trials]
        # best_loss.append(min(losses))
        best_loss.append(np.sort(losses)[0])  # to deal with NaN
        # print(np.array(losses))
        # print(np.sort(losses))
        # best_trial = max(trials, key=lambda trial: trial.last_result['negative_loss'])
        # train_model = best_trial._get_trainable_cls()(best_trial.config)
        # train_model = TrainableHadamardFactorFixedOrder(best_trial.config)
        # train_model = TrainableHadamardFactorSoftmax(best_trial.config)
        # train_model = TrainableHadamardFactorSparsemax(best_trial.config)
        # train_model.restore(str(Path(best_trial.logdir) / best_trial._checkpoint.value))
        # model = train_model.model
    best_rmse = np.sqrt(best_loss)
    print(best_rmse)
