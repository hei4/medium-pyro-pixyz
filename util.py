from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


def make_MNIST_loader(root, batch_size=32):
    transform = transforms.ToTensor()
    kwargs = {'batch_size': batch_size, 'num_workers': 2, 'pin_memory': True}

    train_loader = DataLoader(
        MNIST(root, train=True, transform=transform, download=True),
        shuffle=True, drop_last=True, **kwargs)

    test_loader = DataLoader(
        MNIST(root, train=False, transform=transform, download=True),
        shuffle=False, drop_last=False, **kwargs)

    return train_loader, test_loader


def bilinear_latent(z):
    z_interpol = z[:4].transpose(1, 0).reshape(1, -1, 2, 2)  # [1, z_ch, 2, 2]
    z_interpol = F.interpolate(z_interpol, scale_factor=4, mode='bilinear', align_corners=False)    # [1, z_ch, 8, 8]
    z_interpol = z_interpol.squeeze(dim=0).flatten(start_dim=1).transpose(1, 0)  # [64, z_ch]

    return z_interpol   # [64, z_ch]
