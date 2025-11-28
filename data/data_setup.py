from uci_data import UCIDataset
from augmentation import augment_features

dataset = UCIDataset()
num_aug = 1  # Number of feature vectors to augment
print("Original feature vectors:\n", dataset.get_features().head(num_aug))
dataset.augment()
print("Augmented feature vectors:\n", dataset.get_features().head(num_aug))

