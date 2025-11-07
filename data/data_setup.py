from uci_data import UCIData
from augmentation import augment_features

dataset = UCIData()
augmented_data = augment_features(dataset.get_features(), num_aug=1)

