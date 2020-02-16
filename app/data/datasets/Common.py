import torch.utils.data as data
from data import transform
from torchvision import transforms
from data.datasets import Amazon



class Data_(data.Dataset):
    def __init__(self, size = 224):
        self.Amazon = Amazon.Amazon('./images/Amazon/', transform.Compose([
                                                        transform.Pad(size=size),
#                                                         transform.Resize(size=size),
                                                        transform.ToTensor(),
                                                        transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))

    def __len__(self):
        return len(self.Amazon)

    def __getitem__(self, idx):
        return self.Amazon[idx]
           
