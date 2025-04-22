import torch
from torch.utils.data import Dataset
import torchvision

class colored_MNIST_dataset(Dataset):
    def __init__(self, root_dir, train: bool = True, irm: bool = True):
        dataset = torchvision.datasets.MNIST(root=root_dir, train=train, download=True)
        ## From training set
        self.images = torch.as_tensor(dataset.data, dtype=torch.float) / 127.5 - 1.0
        # self.images = torch.einsum("bwh -> bhw", self.images)
        self.labels = torch.as_tensor(dataset.targets, dtype=torch.long)
        self.irm = irm

        if train:
            env1 = self._create_environment(self.images[::2], self.labels[::2], 0.2)
            env2 = self._create_environment(self.images[1::2], self.labels[1::2], 0.1)
            
            if self.irm:
                self.images = {1: env1['images'], 2: env2['images']}
                self.labels = {1: self.labels[::2], 2: self.labels[1::2]}
                # self.labels = {1: env1['labels'], 2: env2['labels']}
                # self.images = {1: (self.images[::2], env1['labels']), 2: (self.images[1::2], env2['labels'])}
                assert env1['images'].shape[0] == env2['images'].shape[0]
                self.length = env1['images'].shape[0]
            else:
                self.images = torch.concatenate((env1['images'], env2['images']), dim=0)
                # self.labels = torch.concatenate((env1['labels'], env2['labels']), dim=0)
                self.labels = torch.concatenate((self.labels[::2], self.labels[1::2]), dim=0)
                self.length = self.images.shape[0]
        else:
            env = self._create_environment(dataset.images, dataset.labels, 0.9)

            self.images = env['images']
            self.labels = env['labels']
            self.length = self.images.shape[0]
        
        assert len(self.images) == len(self.labels)

    def _create_environment(self, images, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        zeros = torch.zeros(images.shape)
        images = torch.stack([images, images, zeros], dim=-1) # rgb
        # flip to red or green based on colors
        images[torch.tensor(range(len(images))), :, :, (1-colors).long()] *= 0
        
        return {
            'images': images.float().clip(0, 255),
            'labels': labels[:, None]
        }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        item = {}
        if self.irm:
            item['image'] = { env : self.images[env][idx].transpose(0,2) for env in self.images.keys() }
            item['y'] = { env : self.labels[env][idx] for env in self.labels.keys() }
        else:
            item['image'] = self.images[idx].transpose(0,2)
            item['y'] = self.labels[idx]
        return item