from diff_scm.configs import mnist_configs, colored_mnist_configs, brats_configs, morpho_mnist_config, celeba_config

configs = {
    "mnist": mnist_configs,
    "colored_mnist": colored_mnist_configs,
    "morpho_mnist": morpho_mnist_config,
    "celeba": celeba_config,
    "brats": brats_configs,
}

def file_from_dataset(dataset_name):
    assert dataset_name in configs, "Dataset not defined."
    return configs[dataset_name].get_default_configs()