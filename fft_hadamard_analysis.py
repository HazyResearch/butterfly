import pickle
from pathlib import Path
import numpy as np

result_dir = 'results'
experiment_names = []
experiment_names += [[f'Hadamard_factorization_True_softmax_{size}' for size in [8, 16, 32, 64, 128, 256]]]
experiment_names += [[f'Hadamard_factorization_False_softmax_{size}' for size in [8, 16, 32, 64, 128, 256]]]
experiment_names += [[f'Hadamard_factorization_False_sparsemax_{size}' for size in [8, 16, 32, 64, 128, 256]]]
experiment_names += [[f'Fft_factorization_True_softmax_{size}' for size in [8, 16, 32, 64, 128, 256]]]
experiment_names += [[f'Fft_factorization_False_softmax_{size}' for size in [8, 16, 32, 64, 128]]]
experiment_names += [[f'Fft_factorization_False_sparsemax_{size}' for size in [8, 16, 32, 64, 128]]]
experiment_names += [[f'Fft_factorization_sparsemax_no_perm_{size}' for size in [8, 16, 32]]]
experiment_names += [[f'Fft_factorization_softmax_no_perm_{size}' for size in [8, 16, 32]]]
experiment_names += [[f'Randn_factorization_softmax_no_perm_{size}' for size in [8, 16, 32]]]
experiment_names += [[f'Fft_factorization_sparsemax_perm_front_{size}' for size in [8, 16, 32]]]
experiment_names += [[f'Dct_factorization_True_softmax_{size}' for size in [8, 16, 32, 64, 128]]]
experiment_names += [[f'Dct_factorization_False_softmax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'Dct_factorization_False_sparsemax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'Dct_factorization_block_perm_one_extra_{size}' for size in [8, 16, 32, 64, 128]]]
experiment_names += [[f'LegendreEval_factorization_real_True_softmax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'LegendreEval_factorization_real_False_softmax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'LegendreEval_factorization_real_False_sparsemax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'LegendreEval_factorization_complex_True_softmax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'LegendreEval_factorization_complex_False_softmax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'LegendreEval_factorization_complex_False_sparsemax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'Circulant_factorization_real_True_softmax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'Circulant_factorization_real_False_softmax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'Circulant_factorization_real_False_sparsemax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'Circulant_factorization_complex_True_softmax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'Circulant_factorization_complex_False_softmax_{size}' for size in [8, 16, 32, 64]]]
experiment_names += [[f'Circulant_factorization_complex_False_sparsemax_{size}' for size in [8, 16, 32, 64]]]

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
    best_rmse = np.sqrt(best_loss)
    print(best_rmse)
    print(np.sqrt(best_polished_loss))
