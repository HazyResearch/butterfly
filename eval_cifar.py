import torch
from cnn.dataset_utils import get_dataset
from cnn.models.resnet import ResNet18

def validate(state_dict):
    _, _, test_loader = get_dataset({'name': 'CIFAR10'})
    loader = test_loader
    model = ResNet18().cuda()
    model.load_state_dict(state_dict)
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred == target.data.view_as(pred)).long().cpu().sum()
    valid_accuracy = correct.item() / len(loader.dataset)
    return valid_accuracy * 100
