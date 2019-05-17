import models


name_to_model = {
    'LeNet': lambda args: models.LeNet(**args),
    'ResNet18': lambda args: models.ResNet18(**args),
    'PResNet18': lambda args: models.PResNet18(**args),
    'Permutation': lambda args: models.TensorPermutation(32, 32, **args),
    'ResNet20Original': lambda args: models.resnet20original(),
    'MobileNet': lambda args: models.MobileNet(**args),
}


def get_model(model_config):
    name = model_config['name']
    return name_to_model[name](model_config.get('args', None))
