from typing import Dict
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append(str(Path.cwd()))
import numpy as np
from copy import deepcopy
from diff_scm.utils import logger, dist_util

from diff_scm.datasets.load_morpho_mnist import normalize_value

def estimate_counterfactual(config, diffusion, parent_fn, model_fn, model_classifier_free_fn, denoised_fn, data_dict):
    model_kwargs, init_image = get_input_data(config, data_dict)

    if config.sampling.unknown_parents:
        # Get parents
        parents = parent_fn(init_image, y=deepcopy(model_kwargs["parents"]))
        # Replace unconditional parents with parents in "y"
        for att in model_kwargs["overlaps"]:
            model_kwargs["parents"][att] = parents[att]

    # print("y", model_kwargs["y"])
    # print("parents", model_kwargs["parents"])

    # DDIM loop in reverse time order for inferring exogenous noise (image latent space)
    exogenous_noise, abduction_progression = diffusion.ddim_sample_loop(
            model_fn,
            (config.sampling.batch_size,
                config.score_model.num_input_channels,
                config.score_model.image_size,
                config.score_model.image_size),
            clip_denoised=config.sampling.clip_denoised,
            model_kwargs=model_kwargs,
            denoised_fn = denoised_fn if config.sampling.dynamic_sampling else None,
            noise=init_image,
            cond_fn=None,
            device=dist_util.dev(),
            progress=config.sampling.progress,
            eta=config.sampling.eta,
            reconstruction=True,
            sampling_progression_ratio = config.sampling.sampling_progression_ratio
        )
    init_image = exogenous_noise
    # DDIM diffusion inference  with conditioning (intervention), starting from a latent image instead of random noise
    counterfactual_image, diffusion_progression = diffusion.ddim_sample_loop(
            model_classifier_free_fn,
            (config.sampling.batch_size,
                config.score_model.num_input_channels,
                config.score_model.image_size,
                config.score_model.image_size),
            clip_denoised=config.sampling.clip_denoised,
            model_kwargs=model_kwargs,
            denoised_fn = denoised_fn if config.sampling.dynamic_sampling else None,
            noise=init_image,
            cond_fn=None,
            device=dist_util.dev(),
            progress=config.sampling.progress,
            eta=config.sampling.eta,
            reconstruction=False,
            sampling_progression_ratio = config.sampling.sampling_progression_ratio
        )
    sampling_progression = abduction_progression + diffusion_progression
    return counterfactual_image, sampling_progression

def get_models_functions(config, model, anti_parent_predictor):
    def parent_fn(x, y=None, **kwargs):
        assert y is not None
        # Get predicted parents from anti_parent_predictor
        t = torch.zeros(x.shape[0], dtype=torch.long, device=dist_util.dev())
        return anti_parent_predictor(x, y=y, timesteps=t)

    def model_fn(x, t, y=None, conditioning_x=None, parents=None, **kwargs):
        """ y are the interventions """
        if len(y) == 0:
            y = torch.cat([parents[p] for p in config.data.parents], dim=1)
        else:
            interventions = torch.cat([e for e in y.values()], dim=1)
            parents = torch.cat([parents[p] for p in config.data.parents], dim=1)
            y = torch.cat([interventions, parents], dim=1)
        return model(x, t, y = y, conditioning_x=conditioning_x)
    
    # Create an classifier-free guidance sampling function from Glide code
    def model_classifier_free_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        cond_eps, uncond_eps = torch.split(model_out, len(model_out) // 2, dim=0)
        half_eps = uncond_eps + config.sampling.norm_cond_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return eps
    
    # classifier-free guidance without increasing batch - trading off space for time
    def model_classifier_free_opt_fn(x_t, ts, y=None, parents=None, intervened_parents=None, **kwargs):
        """ this returns epsilon, the same as model_fn"""
        ## conditional diffusion output -- eps(x_t, t, do(Z), w)
        cond_y = torch.cat([parents[p] for p in config.data.parents], dim=1)
        if len(y) != 0:
            interventions = torch.cat([e for e in y.values()], dim=1)
            cond_y = torch.cat([interventions, cond_y], dim=1)
        # print("y", cond_y[0])
        cond_eps = model(x_t, ts, cond_y, **kwargs)
        ## unconditional diffusion output
        uncond_kwargs = kwargs.copy()
        # Set parents to unconditional -- w = \emptyset -- to evaluate expected value
        uncond_parents = []
        for att in config.data.parents:
            if att in intervened_parents:
                uncond_parents.append(parents[att])
            else:
                uncond_parents.append(config.classifier.unconditional_default[att] * torch.ones_like(parents[att]))
        uncond_y = torch.cat(uncond_parents, dim=1)
        if len(y) != 0:
            uncond_y = torch.cat([interventions, uncond_y], dim=1)
        # print("uncond y", uncond_y[0])
        # Monte-carlo sampling over parents
        uncond_eps = []
        for _ in range(config.sampling.expected_value_samples):
            uncond_eps.append(model(x_t, ts, uncond_y, **uncond_kwargs))
        uncond_eps = sum(uncond_eps) / len(uncond_eps)

        # print("EPS", cond_eps - uncond_eps)

        eps = (1 + config.sampling.norm_cond_scale) * cond_eps - config.sampling.beta * uncond_eps

        return eps
    
    def inpainting_denoised_fn(x_start,**kwargs):
        # Force the model to have the exact right x_start predictions
        # for the part of the image which is known.
        return (
            x_start * kwargs['inpaint_mask']
            + kwargs['image'] * (1 - kwargs['inpaint_mask'])
    )

    # dynamic normalisation
    def clamp_to_spatial_quantile(x : torch.Tensor, **kwargs):
        p = 0.99
        b, c, *spatial = x.shape
        quantile = torch.quantile(torch.abs(x).view(b,c,-1), p, dim = -1, keepdim =True)
        quantile = torch.max(quantile,torch.ones_like(quantile))
        quantile_broadcasted, _ = torch.broadcast_tensors(quantile.unsqueeze(-1),x)
        return torch.min(torch.max(x,-quantile_broadcasted), quantile_broadcasted) / quantile_broadcasted

    return parent_fn, model_fn, model_classifier_free_opt_fn, clamp_to_spatial_quantile

def get_input_data(config, data_dict):
    def get_target(attr):
        target = None
        if attr in config.sampling.target_class:
            target = config.sampling.target_class[attr]
            if not config.sampling.target_class_normalized and target is not None:
                target = normalize_value(config.sampling.target_class[attr], attr) 
        
        if target is None:
            intervention_target = unconditional_default[attr] * torch.ones_like(model_kwargs[attr])
        else:
            target = torch.as_tensor(target)
            if target.shape == ():
                intervention_target = target.repeat(model_kwargs[attr].shape[0]).to(dist_util.dev())
            else:
                target = target.unsqueeze(0)
                intervention_target = target.repeat(model_kwargs[attr].shape[0], 1).to(dist_util.dev())
        
        return intervention_target if len(intervention_target.shape) == 2 else intervention_target.unsqueeze(1)
    
    model_kwargs = {k: v.to(dist_util.dev()) for k, v in data_dict.items()}
    # Set intervention targets
    unconditional_default = config.score_model.training.unconditional_default

    y = {}
    parents = {}

    interventions = list(model_kwargs.keys())
    interventions.remove("image")
    interventions.remove("attrs")
    # Remove parents from intervention set
    for attr in config.data.parents:
        interventions.remove(attr)
    # Set intervened variables (excluding parents) to target. Otherwise set to unconditional label.
    for attr in interventions:
        y[attr] = get_target(attr)
    # Set parents - if intervened, add value. Else add unconditional value
    intervened_parents = []
    for attr in config.data.parents:
        if attr in config.sampling.target_class:
            if config.sampling.target_class[attr] is not None:
                intervened_parents.append(attr)
        parents[attr] = get_target(attr)
    # TODO: make sure this cat is consistent with morpho_mnist cat
    model_kwargs["y"] = y
    model_kwargs["parents"] = parents
    model_kwargs["intervened_parents"] = intervened_parents

    print("Intervened parents", intervened_parents)

    init_image = data_dict['image'].to(dist_util.dev())

    print(list(model_kwargs.keys()))

    return model_kwargs,init_image