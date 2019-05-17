import models


name_to_model = {
    'LeNet': lambda args: models.LeNet(),
    'ResNet18': lambda args: models.ResNet18(**args),
    'ResNet20Original': lambda args: models.resnet20original(),
    'MobileNet': lambda args: models.MobileNet(**args),
    'ShuffleNet': lambda args: models.ShuffleNetG2(),
}


def get_model(model_config):
    name = model_config['name']
    return name_to_model[name](model_config.get('args', None))
