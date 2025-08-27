import os
import numpy as np
import cv2
import glob
import json
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from scene.colmap_loader import read_intrinsics_binary


def main(args):
    root_dir = args.path
    json_path = Path(root_dir) / "data_views.json"

    with open(json_path, "r") as f:
        data = json.load(f)
    first = data["views"][0]["value"]["ptr_wrapper"]["data"]
    W, H = first["width"], first["height"]
    fov_y_deg = 90
    fov_y = np.deg2rad(fov_y_deg)
    aspect = W / H
    t_y = np.tan(fov_y / 2)
    t_x = t_y * aspect

    # Pixel centers in camera plane z=1
    xs = np.linspace(-t_x, t_x, W, dtype=np.float32)
    ys = np.linspace(t_y, -t_y, H, dtype=np.float32)  # top -> bottom
    xg, yg = np.meshgrid(xs, ys)  # [H,W]

    rays_cam = np.stack([xg, yg, np.ones_like(xg)], axis=-1)
    rays_cam /= np.linalg.norm(rays_cam, axis=-1, keepdims=True)  # unit
    np.save(Path(args.path) / "raymap_fisheye.npy", rays_cam)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, default="/media/scannetpp/demo/0a5c013435/dslr/")
    args = parser.parse_args()
    main(args)
