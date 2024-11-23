from .MagicBathyNet import MagicBathyNetDataLoader
from .SpecificImages import SpecificImagesDataLoader


def dataset_loader(**configs):
    dataset_name = configs.pop("dataset_name")
    dataset_wrapper = {
        "MagicBathyNet": MagicBathyNetDataLoader,
        "SpecificImages": SpecificImagesDataLoader,
    }
    dataset = dataset_wrapper[dataset_name](**configs)

    return dataset
