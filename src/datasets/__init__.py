from .cifar100 import CIFAR100Dataset
from .dtd import DTDDataset
from .eurosat import EuroSATDataset
from .food101 import Food101Dataset
from .gtsrb import GTSRBDataset
from .stanford_cars import StanfordCarsDataset
from .svhn import SVHNDataset

datasets = {
    "cifar100": CIFAR100Dataset,
    "dtd": DTDDataset,
    "eurosat": EuroSATDataset,
    "food101": Food101Dataset,
    "gtsrb": GTSRBDataset,
    "stanford_cars": StanfordCarsDataset,
    "svhn": SVHNDataset,
}
