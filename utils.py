
from torch.utils.data import Dataset

class tokenizedDataset(Dataset):
    def __init__(self,metadata):
        self.metadata = metadata

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        sample = self.metadata[idx]
        return {"input_ids" : sample['input_ids'][0], "label" : sample['labels'][0]}
