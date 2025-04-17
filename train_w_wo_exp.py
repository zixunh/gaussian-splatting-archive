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

import os
from argparse import ArgumentParser
import time

tanks_and_temples_scenes = ["truck", "train"]
deep_blending_scenes = ["drjohnson", "playroom"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--output_path", default="./eval")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()

if not args.skip_training:
    common_args_wo_exp = " --iterations 30_000 --checkpoint_iterations 7_000 15_000 30_000 --save_iterations 7_000 15_000 30_000 --test_iterations 7_000 15_000 30_000 --eval"
    common_args_w_exp = common_args_wo_exp + " --exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp"
    common_argss = [common_args_w_exp, common_args_wo_exp]
    subs = ["_w_exp", "_wo_exp"]

    for sub, common_args in zip(subs, common_argss):
        start_time = time.time()
        for scene in tanks_and_temples_scenes:
            source = args.tanksandtemples + "/" + scene
            os.system("python train.py -s " + source + " -m " + args.output_path + sub + "/" + scene + common_args)
        tandt_timing = (time.time() - start_time)/60.0

        start_time = time.time()
        for scene in deep_blending_scenes:
            source = args.deepblending + "/" + scene
            os.system("python train.py -s " + source + " -m " + args.output_path + sub + "/" + scene + common_args)
        db_timing = (time.time() - start_time)/60.0

        with open("timing"+sub+".txt", 'w') as file:
            file.write(f"tandt: {tandt_timing} minutes \n db: {db_timing} minutes\n")

