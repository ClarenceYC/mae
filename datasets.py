import torch
import torchvision
from data.cifar import CIFAR10, CIFAR100
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

import transforms

class MYMNIST(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        pin_memory = True if use_gpu else False

        trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)

        trainloader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)

        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10

class MYCIFAR10(object):
    def __init__(self, batch_size, use_gpu, num_workers, noise_type, noise_rate, args, sampler=None, relabel=None,
                 transform_train=transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ]),
                 transform_test=transforms.Compose([
                     transforms.ToTensor(),
                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                 ])
                 ):

        pin_memory = True if use_gpu else False

        trainset = CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform_train,
                           noise_type=noise_type,
                           noise_rate=noise_rate
                           )
        # relabeling or not
        if relabel != None:
            for ind in relabel:
                trainset.train_noisy_labels[ind] = trainset.train_labels[ind][0]
                trainset.noise_or_not[ind] = True

        # sampling or not
        if sampler == None:
            trainloader = DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )
        else:
            trainloader = DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(sampler),
            )

        testset = CIFAR10(root='./data/cifar10', train=False, download=True, transform=transform_test,
                           noise_type=noise_type,
                           noise_rate=noise_rate
                          )

        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 10

class MYCIFAR100(object):
    def __init__(self, batch_size, use_gpu, num_workers, noise_type, noise_rate, sampler=None, relabel=None):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        pin_memory = True if use_gpu else False

        trainset = CIFAR100(root='./data/cifar100', train=True, download=True, transform=transform_train,
                           noise_type=noise_type,
                           noise_rate=noise_rate
                            )
        # relabeling or not
        if relabel != None:
            for ind in relabel:
                trainset.train_noisy_labels[ind] = trainset.train_labels[ind][0]
                trainset.noise_or_not[ind] = True

        # sampling or not
        if sampler == None:
            trainloader = DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )
        else:
            trainloader = DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(sampler),
            )

        testset = CIFAR100(root='./data/cifar100', train=False, download=True, transform=transform_test,
                           noise_type=noise_type,
                           noise_rate=noise_rate
                           )

        testloader = DataLoader(
            testset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )

        self.trainloader = trainloader
        self.testloader = testloader
        self.num_classes = 100

__factory = {
    'mnist': MYMNIST,
    'cifar100': MYCIFAR100,
    'cifar10': MYCIFAR10,
}

def create(name, batch_size, use_gpu, num_workers, noise_type, noise_rate, sampler=None, relabel=None):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers, noise_type, noise_rate, sampler, relabel)