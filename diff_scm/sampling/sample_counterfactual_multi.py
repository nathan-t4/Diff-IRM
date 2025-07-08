"""
Like score_sampling.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""
import os
import numpy as np
import torch as th
import torch.distributed as dist
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
import argparse
import random
import matplotlib.pyplot as plt


from diff_scm.datasets import loader
from diff_scm.configs import get_config
from diff_scm.utils import logger, dist_util, script_util
from diff_scm.sampling.sampling_multi_utils import get_models_functions, estimate_counterfactual # TODO

from diff_scm.datasets.load_morpho_mnist import unnormalize

def main(args):
    config = get_config.file_from_dataset(args.dataset)

    dist_util.setup_dist()
    logger.configure(Path(os.path.join(config.experiment_path, config.experiment_name, ("counterfactual_sampling_" + "_".join(config.classifier.label)))))
    config.sampling.model_path = config.sampling.model_path_fn("test_1")
    config.sampling.classifier_path = config.sampling.classifier_path_fn("")

    logger.log("creating loader...")
    test_loader = loader.get_data_loader(args.dataset, config, split_set='test', generator = False, irm = False)
    logger.log("creating model and diffusion...")

    classifier, diffusion, model = script_util.get_multi_models_from_config(config)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log(f"Number of parameteres: {pytorch_total_params}")

    cond_fn, model_fn, model_classifier_free_fn, denoised_fn = get_models_functions(config, model, classifier)

    logger.log("sampling...")

    all_results = []
    for i, data_dict in enumerate(test_loader):
        
        counterfactual_image, sampling_progression = estimate_counterfactual(config, 
                                                diffusion, cond_fn, model_fn, 
                                                model_classifier_free_fn, denoised_fn, 
                                                data_dict)
        
        counterfactual_image = np.einsum("bcwh -> bwhc", counterfactual_image.cpu().numpy().clip(0,1))
            
        results_per_sample = {"original": data_dict,
                              "counterfactual_sample" : counterfactual_image,
                                                                }

        if config.sampling.progress:
            results_per_sample.update({"diffusion_process": sampling_progression})
                                                        
        all_results.append(results_per_sample)

        if config.sampling.num_samples is not None and ((i+1) * config.sampling.batch_size) >= config.sampling.num_samples:
            break                

    all_results = {k: [dic[k] for dic in all_results] for k in all_results[0]}

    if dist.get_rank() == 0:
        out_path = os.path.join(logger.get_dir(), f"samples.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, all_results)
        sample_list = all_results
        print(all_results.keys())

        # Plot results
        columns = 6
        rows = 3
        fig = plt.figure(figsize=(13, 8))
        title = [f"{att}: {intervention}" for att, intervention in config.sampling.target_class.items()]
        fig.suptitle(" ".join(title))
        ax = []

        for i in range(columns * rows):
            img = sample_list["counterfactual_sample"][0][i]
            # Plot original attributes as subplot title
            attributes = {att : sample_list["original"][0][att][i] for att in config.data.parents}
            attributes = [unnormalize(attributes[att], att) for att in config.data.parents] if config.data.normalize else [attributes[att] for att in config.data.parents]

            labels = []
            for att in attributes:
                if att.shape == ():
                    labels.append(f"{att:.2}" if (att.dtype == th.float32) else f"{att}")
                else:
                    raise NotImplementedError()
            label = ", ".join(labels)
            # create subplot and append to ax
            ax.append(fig.add_subplot(rows, columns, i + 1))
            ax[-1].set_title(str(label))  # set title
            plt.imshow(img, cmap='grey')
        
        plt.axis("off")
        plt.savefig(f"{config.dataset_name}_counterfactual_{config.sampling.norm_cond_scale}_{config.sampling.beta}_{config.sampling.expected_value_samples}.png")

    dist.barrier()
    logger.log(f"sampling complete")

    # TODO: add counterfactual metrics


def reseed_random(seed):
    random.seed(seed)     # python random generator
    np.random.seed(seed)  # numpy random generator
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="mnist or brats", type=str)
    args = parser.parse_args()
    print(args.dataset)
    main(args)