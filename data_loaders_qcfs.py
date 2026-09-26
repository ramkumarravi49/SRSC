# data_loaders_qcfs.py
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
#from Preprocess.augment import CIFAR10Policy, Cutout  # <-- keep exactly this import
from data_agument import CIFAR10Policy, Cutout

def build_cifar_qcfs(cutout=True, use_cifar10=True, download=False):
    """
    QCFS-style CIFAR loader for TET:
    RandomCrop -> RandomHorizontalFlip -> CIFAR10Policy -> ToTensor -> Normalize -> Cutout
    """

    if use_cifar10:
        # --- Train transforms ---
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
            Cutout(n_holes=1, length=16) if cutout else transforms.Lambda(lambda x: x)
        ])

        # --- Test transforms ---
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = CIFAR10(root='./raw/', train=True,
                                download=download, transform=transform_train)
        val_dataset = CIFAR10(root='./raw/', train=False,
                              download=download, transform=transform_test)

    else:  # CIFAR-100 variant
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            CIFAR10Policy(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[x/255. for x in [129.3, 124.1, 112.4]],
                                 std=[x/255. for x in [68.2, 65.4, 70.4]]),
            Cutout(n_holes=1, length=16) if cutout else transforms.Lambda(lambda x: x)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x/255. for x in [129.3, 124.1, 112.4]],
                                 std=[x/255. for x in [68.2, 65.4, 70.4]])
        ])
        train_dataset = CIFAR100(root='./raw/', train=True,
                                 download=download, transform=transform_train)
        val_dataset = CIFAR100(root='./raw/', train=False,
                               download=download, transform=transform_test)

    return train_dataset, val_dataset
