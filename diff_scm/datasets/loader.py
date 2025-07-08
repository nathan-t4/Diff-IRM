import numpy as np
import random
import torch

from diff_scm.datasets.load_brats import BrainDataset
from diff_scm.datasets.load_mnist import MNIST_dataset
from diff_scm.datasets.load_colored_mnist import colored_MNIST_dataset
from diff_scm.datasets.load_morpho_mnist import Morpho_MNIST_Dataset
from diff_scm.datasets.load_celeba import Celeba

datasets = {
    'mnist': MNIST_dataset,
    'colored_mnist': colored_MNIST_dataset,
    'morpho_mnist': Morpho_MNIST_Dataset,
    'celeba': Celeba,
}


def seed_worker(worker_id):
    np.random.seed(worker_id)
    random.seed(0)

g = torch.Generator()
g.manual_seed(0)

def _plot(dataset, colored=False):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(13, 8))
    columns = 6
    rows = 3
    # ax enables access to manipulate each of subplots
    ax = []

    irm = isinstance(dataset[0]["image"], dict)

    for i in range(columns * rows):
        img = dataset[i]["image"][1] if irm else dataset[i]["image"]
        img = np.einsum("cwh->whc", img)
        
        attributes = {att : dataset.metrics[att][i] for att in dataset.attributes}
        attributes = [dataset.unnormalize_fn(v, n) for n, v in attributes.items()] if dataset.normalize else [v for n,v in attributes.items()]
        if i == 0:
            print(attributes)
        labels = []
        for att in attributes:
            if att.shape == ():
                labels.append(f"{att:.2}" if (att.dtype == torch.float32) else f"{att}")
            else:
                raise NotImplementedError()
        label = ", ".join(labels)


        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i + 1))
        ax[-1].set_title(str(label))  # set title
        if img.shape[2] == 3: # rgb
            plt.imshow(img)
        else:
            plt.imshow(img, 'grey')

    plt.savefig(f"test.png")  # finally, render the plot


def get_data_loader(dataset, config, split_set, generator = True, irm = False):
    if dataset == "mnist":
        loader = get_data_loader_mnist(config.data.path, config.sampling.batch_size, 
                                    split_set=split_set, which_label=config.classifier.label)
    elif dataset == "colored_mnist":
        loader = get_data_loader_colored_mnist(config.data.path, config.sampling.batch_size,
                                               split_set=split_set, which_label=config.classifier.label, irm=irm)
    elif dataset == "brats":
        loader = get_data_loader_brats(config.data.path, config.sampling.batch_size, split_set=split_set,
                                            sequence_translation = config.data.sequence_translation)
    elif dataset in datasets:
        loader = get_data_loader_default(datasets[dataset], config.data.attribute_size, config.data.path, config.sampling.batch_size, config.data.normalize,
                                         split_set=split_set, which_label=config.classifier.label)
    else:
        raise Exception("Dataset does exit")
    
    return get_generator_from_loader(loader) if generator else loader

def get_data_loader_default(torch_dataset, attribute_size, path, batch_size, normalize: bool, split_set: str = 'train', which_label: str = 'class'):
    assert split_set in ["train", "val", "test"]
    default_kwargs = {"shuffle": True, "num_workers": 1, "drop_last": True, "batch_size": batch_size}
    
    # Make test set deterministic
    if split_set == "test":
        default_kwargs["shuffle"] = False

    dataset = torch_dataset(attribute_size=attribute_size, root_dir=path, train=split_set != "test", normalize_=normalize)
    
    # _plot(dataset)

    if split_set != "test":
        val_ratio = 0.1
        split = torch.utils.data.random_split(dataset,
                                              [int(len(dataset) * (1 - val_ratio)), int(len(dataset) * val_ratio)],
                                              generator=torch.Generator().manual_seed(42))
        dataset = split[0] if split_set == "train" else split[1]
        
    return torch.utils.data.DataLoader(dataset, **default_kwargs)

def get_data_loader_mnist(path, batch_size, split_set: str = 'train', which_label: str = "class"):
    assert split_set in ["train", "val", "test"]
    default_kwargs = {"shuffle": True, "num_workers": 1, "drop_last": True, "batch_size": batch_size}
    dataset = MNIST_dataset(root_dir=path, train=split_set != "test")

    _plot(dataset)

    if split_set != "test":
        val_ratio = 0.1
        split = torch.utils.data.random_split(dataset,
                                              [int(len(dataset) * (1 - val_ratio)), int(len(dataset) * val_ratio)],
                                              generator=torch.Generator().manual_seed(42))
        dataset = split[0] if split_set == "train" else split[1]
    
    return torch.utils.data.DataLoader(dataset, **default_kwargs)


def get_data_loader_colored_mnist(path, batch_size, split_set: str = 'train', which_label: str = 'class', irm: bool = False):
    assert split_set in ["train", "val", "test"]
    default_kwargs = {"shuffle": True, "num_workers": 1, "drop_last": True, "batch_size": batch_size}
    # Make test set deterministic
    if split_set == "test":
        default_kwargs["shuffle"] = False

    dataset = colored_MNIST_dataset(root_dir=path, train=split_set != "test", irm=irm) # create colored mnist

    _plot(dataset)

    if split_set != "test":
        val_ratio = 0.1
        split = torch.utils.data.random_split(dataset,
                                              [int(len(dataset) * (1 - val_ratio)), int(len(dataset) * val_ratio)],
                                              generator=torch.Generator().manual_seed(42))
        dataset = split[0] if split_set == "train" else split[1]
        
    return torch.utils.data.DataLoader(dataset, **default_kwargs)

def get_data_loader_morpho_mnist(attribute_size, path, batch_size, split_set: str = 'train', which_label: str = 'class'):
    assert split_set in ["train", "val", "test"]
    default_kwargs = {"shuffle": True, "num_workers": 1, "drop_last": True, "batch_size": batch_size}
    
    # Make test set deterministic
    if split_set == "test":
        default_kwargs["shuffle"] = False

    dataset = Morpho_MNIST_Dataset(attribute_size=attribute_size, root_dir=path, train=split_set != "test")
    
    _plot(dataset)

    if split_set != "test":
        val_ratio = 0.1
        split = torch.utils.data.random_split(dataset,
                                              [int(len(dataset) * (1 - val_ratio)), int(len(dataset) * val_ratio)],
                                              generator=torch.Generator().manual_seed(42))
        dataset = split[0] if split_set == "train" else split[1]
        
    return torch.utils.data.DataLoader(dataset, **default_kwargs)

def get_data_loader_brats(path, batch_size, split_set: str = 'train',
                             sequence_translation : bool = False, 
                             healthy_data_percentage : float = 1.0):

    assert split_set in ["train", "val", "test"]
    default_kwargs = {"drop_last": True, "batch_size": batch_size, "pin_memory" : True, "num_workers": 8,
                    "prefetch_factor" : 8, "worker_init_fn" : seed_worker, "generator": g,}

    if split_set == "test":
        default_kwargs["shuffle"] = False
        default_kwargs["num_workers"] = 1
        dataset = BrainDataset(path, n_tumour_patients = None,
                               n_healthy_patients = 0, split = split_set, 
                               sequence_translation = sequence_translation,
                               )
        return torch.utils.data.DataLoader(dataset, **default_kwargs)
    else:
        default_kwargs["shuffle"] = True
        default_kwargs["num_workers"] = 8
        dataset_healthy = BrainDataset(path, split = split_set,
                n_tumour_patients=0, n_healthy_patients=None,
                skip_healthy_s_in_tumour=True,skip_tumour_s_in_healthy=True,
                )
        dataset_unhealthy = BrainDataset(path, split = split_set,
                n_tumour_patients=None, n_healthy_patients=0,
                skip_healthy_s_in_tumour=True,skip_tumour_s_in_healthy=True,
                )
        if healthy_data_percentage is not None:
            healthy_size = int(len(dataset_healthy)*healthy_data_percentage)
            unhealthy_size = len(dataset_unhealthy)
            total_size = healthy_size + unhealthy_size
            samples_weight = torch.cat([torch.ones(healthy_size)   * total_size / healthy_size,
                                        torch.ones(unhealthy_size) * total_size / unhealthy_size]
                                        ).double()
            sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
            default_kwargs["sampler"] = sampler
            default_kwargs.pop('shuffle', None) # shuffle and sampler are mutually exclusive

            dataset = torch.utils.data.dataset.ConcatDataset([torch.utils.data.Subset(dataset_healthy, range(0, int(len(dataset_healthy)*healthy_data_percentage))),
                                 dataset_unhealthy])
        else:
            dataset = dataset_healthy
        
    print(f"dataset lenght: {len(dataset)}")
    return torch.utils.data.DataLoader(dataset, **default_kwargs)


def get_generator_from_loader(loader):
    while True:
        yield from loader