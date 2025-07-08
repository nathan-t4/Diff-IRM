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


from diff_scm.datasets import loader
from diff_scm.configs import get_config
from diff_scm.utils import logger, dist_util, script_util
from diff_scm.sampling.sampling_multi_utils import get_models_functions, estimate_counterfactual

def main(args):
    config = get_config.file_from_dataset(args.dataset)

    dist_util.setup_dist()
    logger.configure(Path(os.path.join(config.experiment_path, config.experiment_name, ("classifier_eval" + "_".join(config.classifier.label)))))
    config.sampling.model_path = config.sampling.model_path_fn("test_4")
    config.sampling.classifier_path = config.sampling.classifier_path_fn("")

    logger.log("creating loader...")
    test_loader = loader.get_data_loader(args.dataset, config, split_set='test', generator = False, irm = False) 

    logger.log("creating model and diffusion...")

    classifier, diffusion, model = script_util.get_multi_models_from_config(config)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.log(f"Number of parameteres: {pytorch_total_params}")

    parent_fn, model_fn, model_classifier_free_fn, denoised_fn = get_models_functions(config, model, classifier)

    logger.log("sampling...")

    # all_results = []
    accuracy = {k : [] for k in config.data.parents}
    # the diffusion model generates. The anti-parent classifier predicts parents
    for i, data_dict in enumerate(test_loader):
        batch = data_dict["image"].to(dist_util.dev())
        attributes = data_dict
        attributes.pop("image", None)
        attributes.pop("attrs", None)

        y = {}
        for k,v in attributes.items():
            att = config.classifier.unconditional_default[k] * th.ones_like(v)
            att = att.unsqueeze(1) if len(att.shape) == 1 else att
            y[k] = att

            attributes[k] = v.to(dist_util.dev())
        
        # Get the model outputs and calculate the accuracy
        t = th.zeros(batch.shape[0], dtype=th.long, device=dist_util.dev())
        with th.no_grad():
            output = parent_fn(batch, y=y, timesteps=t)

        predicted_labels = th.argmax(output["digit"], 1)
        accuracy["digit"].append((predicted_labels == th.argmax(attributes["digit"], 1)).float().mean())

        print(predicted_labels, th.argmax(attributes["digit"], 1))

    if dist.get_rank() == 0:
        accuracy['digit'] = sum(accuracy["digit"]) / len(accuracy["digit"])
        logger.log(f"Accuracy is {accuracy['digit']}")
        # out_path = os.path.join(logger.get_dir(), f"samples.npz")
        # logger.log(f"saving to {out_path}")
        # np.savez(out_path, all_results)

    dist.barrier()
    logger.log("sampling complete")


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
