__all__ = [
    'get_dataset',
]








dataset_list = [
    'mnist', 
    'svhn', 
    'cifar10', 
    'cifar100', 
    'stl10', 
    'lsun', 
    'imagenet'
]

# dataset mean and std 
# https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
# https://discuss.pytorch.org/t/computing-the-mean-and-std-of-dataset/34949
means = {
    'mnist'         : [0.13066049],
    'svhn'          : [0.43768210, 0.44376970, 0.47280442],
    'cifar10'       : [0.49139968, 0.48215841, 0.44653091],
    'cifar100'      : [0.50707516, 0.48654887, 0.44091784],
    'stl10'         : [0.44671062, 0.43980984, 0.40664645],
    'lsun'          : [0.485, 0.456, 0.406], # copied from imagenet
    'cub200'        : [0.48310599, 0.49175689, 0.42481980],
    'imagenet'      : [0.485, 0.456, 0.406]
}

stds = {
    'mnist'         : [0.30810780],
    'svhn'          : [0.19803012, 0.20101562, 0.19703614],
    'cifar10'       : [0.24703223, 0.24348513, 0.26158784],
    'cifar100'      : [0.26733429, 0.25643846, 0.27615047],
    'stl10'         : [0.26034098, 0.25657727, 0.27126738],
    'lsun'          : [0.229, 0.224, 0.225], # copied from imagenet
    'cub200'        : [0.22814971, 0.22405523, 0.25914747],
    'imagenet'      : [0.229, 0.224, 0.225]
}


def check_dataset(name):
    if name in dataset_list:
        return True
    else:
        return False

def get_dataset_stats(name, device='cpu'):
    if check_dataset(name):
        mean = means[name]
        std = stds[name]
    else:
        mean = [0.5]
        std = [0.5]

    return torch.tensor(mean, dtype=torch.float32).to(device), torch.tensor(std, dtype=torch.float32).to(device)

def get_dataset(name, train, input_size=224, normalize=True, augment=True, num_samples=-1):
    root = os.path.join(data_root, name)

    transform = get_transform(name, train, input_size, normalize, augment)

    if name == 'mnist':
        dataset = torchvision.datasets.MNIST(root, train=train, download=True, transform=transform)
    elif name == 'svhn':
        dataset = torchvision.datasets.SVHN(root, split='train' if train else 'test', download=True, transform=transform)
    elif name == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root, train=train, download=True, transform=transform)
    elif name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root, train=train, download=True, transform=transform)
    elif name == 'stl10':
        dataset = torchvision.datasets.STL10(root, split='train' if train else 'test', download=True, transform=transform)		
    elif name in ['imagenet']:
        root = os.path.join(root, 'train' if train else 'val')
        dataset = ImageFolder(root, transform=transform)

    if num_samples != -1:
        num_samples = min(num_samples, len(dataset))
        indices = range(len(dataset))
        indices = random.sample(indices, num_samples)
        dataset = Subset(dataset, indices)

    return dataset