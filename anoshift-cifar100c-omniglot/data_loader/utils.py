
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, samples, labels):
        self.labels = labels
        self.samples = samples
        self.dim_features = samples.shape[1]
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        sample = self.samples[idx]
        data = [sample, label]
        return data