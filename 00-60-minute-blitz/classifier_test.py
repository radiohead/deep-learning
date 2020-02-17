import torch
import torchvision

from classifier import CIFAR10Data, CIFAR10Classifier

if __name__ == '__main__':
    path = './cifar_net.pth'

    net = CIFAR10Classifier()
    net.load_state_dict(torch.load(path))

    data = CIFAR10Data()
    it = iter(data.testloader)
    images, labels = it.next()

    print('Test: ', ' '.join('%5s' %
                             data.classes[labels[j]] for j in range(4)))

    predicted = net.predict(images)
    print('Predict: ', ' '.join('%5s' %
                                data.classes[predicted[j]] for j in range(4)))

    correct = 0
    total = 0

    with torch.no_grad():
        for data in data.testloader:
            images, labels = data
            predicted = net.predict(images)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Prediction accuracy on 10.000 test images: %d %%' %
          (100 * correct / total))
