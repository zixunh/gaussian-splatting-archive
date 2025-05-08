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
import time


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, fetch_nearest_exp=False):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    render_times_python = []
    render_times_overall = []
    render_times_prep = []
    render_times_dup = []
    render_times_sort = []
    render_times_render = []

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_start = time.time()
        renderings = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, fetch_prev_next_exp=fetch_nearest_exp)
        rendering = renderings["render"]
        runtime = renderings["time"]
        torch.cuda.synchronize()
        render_end = time.time()
        render_times_python.append((render_end - render_start)*1000)

        render_times_overall.append(runtime[0])
        render_times_prep.append(runtime[1])
        render_times_dup.append(runtime[2])
        render_times_sort.append(runtime[3])
        render_times_render.append(runtime[4])
        
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

    means = torch.tensor(render_times_python).mean()
    maxs = torch.tensor(render_times_python).max()
    FPS = 1.0 / (means / 1000.0)
    print(f"  AVG_Render_Time : {means} ms")
    print(f"  MAX_Render_Time : {maxs} ms")
    print(f"  FPS: {FPS}")   

    means = torch.tensor(render_times_overall).mean()
    maxs = torch.tensor(render_times_overall).max()
    FPS = 1.0 / (means / 1000.0)
    print(f"  AVG_OVERALL_Time : {means} ms")
    print(f" AVG_OVERALL_Time FPS: {FPS}")   

    means = torch.tensor(render_times_prep).mean()
    maxs = torch.tensor(render_times_prep).max()
    print(f"  AVG_PREP_Time : {means} ms")

    means = torch.tensor(render_times_dup).mean()
    maxs = torch.tensor(render_times_dup).max()
    print(f"  AVG_DUP_Time : {means} ms")

    means = torch.tensor(render_times_sort).mean()
    maxs = torch.tensor(render_times_sort).max()
    print(f"  AVG_SORT_Time : {means} ms")

    means = torch.tensor(render_times_render).mean()
    maxs = torch.tensor(render_times_render).max()
    print(f"  AVG_RenFunc_Time : {means} ms")

    max_allocated_memory_after = torch.cuda.max_memory_allocated()
    print(f"Max Allocated Memory After Rendering: {max_allocated_memory_after} bytes")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, fetch_nearest_exp: bool = False):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp)

        if not skip_test:
             render_set(dataset.model_path, "test_test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, False, fetch_nearest_exp=fetch_nearest_exp)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--fetch_nearest_exp", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)


    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.fetch_nearest_exp)