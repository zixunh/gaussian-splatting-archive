#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import time
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.image_utils import psnr
import numpy as np
import cv2

def render_set(model_path, mask_tensor, name, iteration, views, gaussians, pipeline, background, train_test_exp):
    max_allocated_memory_before = torch.cuda.max_memory_allocated()
    print(f"Max Allocated Memory Before Rendering: {max_allocated_memory_before} bytes")
    torch.cuda.empty_cache()

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    render_times = []
    image_save_times = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_start = time.time()
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp)["render"]
        torch.cuda.synchronize()
        render_end = time.time()
        render_times.append((render_end - render_start)*1000)

        image_save_start = time.time()
        gt = view.original_image[0:3, :, :]
        rendering[mask_tensor == 0] = 0.0
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        image_save_end = time.time()
        image_save_times.append((image_save_end - image_save_start)*1000)
        try:
            ps = psnr(rendering, gt).mean()
            print(f"  PSNR: {ps}")
        except:
            pass
    
    means = torch.tensor(render_times).mean()
    maxs = torch.tensor(render_times).max()
    FPS = 1.0 / (means / 1000.0)
    print(f"  AVG_Render_Time : {means} ms")
    print(f"  MAX_Render_Time : {maxs} ms")
    print(f"  FPS: {FPS}")   
    max_allocated_memory_after = torch.cuda.max_memory_allocated()
    print(f"Max Allocated Memory After Rendering: {max_allocated_memory_after} bytes")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, fov_mod, sample_step, mask_path):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        dataset.fov_mod = fov_mod
        dataset.sample_step = sample_step
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, skip_train_cameras=skip_train, skip_test_cameras=skip_test)
        valid_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        valid_mask = np.repeat(valid_mask[None, ...], 3, axis=0)
        valid_mask = torch.tensor(valid_mask)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, valid_mask, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp)

        if not skip_test:
             dataset.train_test_exp = False
             render_set(dataset.model_path, valid_mask, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    #parser.add_argument("--model_path", default="", type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mask_path", type=str, default = None)
    parser.add_argument("--sample_step", type=float, default = None)
    parser.add_argument("--fov_mod", type=float, default = None)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.fov_mod, args.sample_step, args.mask_path)