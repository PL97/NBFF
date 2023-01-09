import torch
from torchvision import datasets, transforms



def get_data(name="MNIST", batch_size=128, random_seed=0, root='../data/'):
    stats = {}
    if name == "MNIST":
        # Training dataset
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=root, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=batch_size, shuffle=True, num_workers=8)
        # Test dataset
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root=root, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), batch_size=batch_size, shuffle=True, num_workers=8)
        stats['input_channels'] = 1
    elif name == "SVHN":
        # Training dataset
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=root, split="train", download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Resize(28),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])), batch_size=64, shuffle=True, num_workers=4)
        # Test dataset
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(root=root, split="test", download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(28),
                transforms.Normalize((0.1307,), (0.3081,))
            ])), batch_size=64, shuffle=True, num_workers=4)
        stats['input_channels'] = 3
    else:
        exit("dataset not found")
    
    return train_loader, test_loader, test_loader, stats