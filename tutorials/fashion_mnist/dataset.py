from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_training_data(batch_size: int = 64) -> DataLoader:
    """
    Creates the training dataset.
    """
    return load_data(train=True, batch_size=batch_size)


def load_testing_data(batch_size: int = 64) -> DataLoader:
    """
    Creates the testing dataset.
    """
    return load_data(train=False, batch_size=batch_size)


def load_data(train: bool, batch_size: int) -> DataLoader:
    """
    Helper function to load the FashionMNIST dataset.
    """
    data = datasets.FashionMNIST(
        root="data",
        train=train,
        download=True,
        transform=ToTensor()
    )
    return DataLoader(data, batch_size=batch_size)
