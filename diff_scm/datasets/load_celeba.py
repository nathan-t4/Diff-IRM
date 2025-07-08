from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from torchvision.transforms import Resize, ToTensor, CenterCrop, Compose, ConvertImageDtype
import torch

# Fix to load truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

MIN_MAX = {
    'image': [0.0, 255.0]
}

def load_data(data_dir, split):
    transforms = Compose([CenterCrop(150), Resize((64, 64)), ToTensor(), ConvertImageDtype(dtype=torch.float32),])
    data = CelebA(root=data_dir, split=split, transform=transforms, download=True)
    return data

def unnormalize(value, name):
    # [0,1] -> [min,max]
    value = (value * (MIN_MAX[name][1] - MIN_MAX[name][0])) +  MIN_MAX[name][0]
    return value.to(torch.uint8)

class Celeba(Dataset):
    def __init__(self, attribute_size, train=True, normalize_=True,
                 transform=None, transform_cls=None, root_dir='./data/'):
        super().__init__()
        self.has_valid_set = True
        self.normalize = normalize_
        self.transform = transform
        self.transform_cls = transform_cls
        self.data = load_data(root_dir, "train" if train else "test")
        self.attributes = list(attribute_size.keys())
        attribute_ids = [self.data.attr_names.index(attr) for attr in self.attributes]
        self.metrics = {attr: torch.as_tensor(self.data.attr[:, attr_id], dtype=torch.int64) for attr, attr_id in zip(self.attributes, attribute_ids)}

        self.attrs = torch.cat([self.metrics[attr].unsqueeze(1)
                                for attr in self.attributes], dim=1)
        self.possible_values = {attr: torch.unique(values, dim=0) for attr, values in self.metrics.items()}
        self.bins = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # if self.transform:
        #     return self.transform(self.data[idx][0], self.attrs[idx])

        # if self.transform_cls:
        #     return self.transform_cls(self.data[idx][0]), self.attrs[idx]

        # return self.data[idx][0], self.attrs[idx]
        item = {col: values[idx] for col, values in self.metrics.items()}
        item['image'] = self.data[idx][0]
        item['attrs'] = self.attrs[idx]
        if self.transform:
            return self.transform(item["image"], item["attrs"])
        if self.transform_cls:
            return self.transform_cls(item["image"], item["attrs"])
        return item


if __name__ == "__main__":

    # attribute_size = {
    #     "Young": 1,
    #     "Male": 1,
    #     "No_Beard": 1,
    #     "Bald" : 1
    # }

    attribute_size = {
        "Smiling": 1,
        "Eyeglasses": 1,
    }

    dataset = Celeba(attribute_size, root_dir='/store/nt9637/Diff-IRM/datasets/')
    print(len(dataset))
    a = dataset[9]
    print(a["image"].shape, a["attrs"].shape, a["attrs"])
    print(a["Smiling"], a["Eyeglasses"])
