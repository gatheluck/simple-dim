__all__ = [
    'get_dataset',
]



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
    elif name in ['lsun', 'cub200', 'dog120', 'food101', 'flower102', 'tiny_imagenet', 'imagenet', 'SIN']:
        root = os.path.join(root, 'train' if train else 'val')
        dataset = ImageFolder(root, transform=transform)

    if num_samples != -1:
        num_samples = min(num_samples, len(dataset))
        indices = range(len(dataset))
        indices = random.sample(indices, num_samples)
        dataset = Subset(dataset, indices)

    return dataset