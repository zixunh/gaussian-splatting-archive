# This is the code to prepare the FOV images from the scannet fisheye images.

import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from utils.graphics_utils import focal2fov
import torch

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(np.float64, elems[4:])))
    return camera_id, model, width, height, params

def fov2tan(fovx, fovy, interval):
    theta_arr = np.arange(interval / 2, fovx, interval)
    theta_arr = np.sort(np.concatenate((-theta_arr, theta_arr)))
    phi_arr = np.arange(interval / 2, fovy, interval)
    phi_arr = np.sort(np.concatenate((-phi_arr, phi_arr)))

    sin_t = np.sin(theta_arr)
    cos_t = np.cos(theta_arr)
    sin_p = np.sin(phi_arr)[:, None]
    cos_p = np.cos(phi_arr)[:, None]
    tan_t = sin_t / cos_t
    tan_p = sin_p / cos_p
    return tan_t, tan_p

    r = ((sin_t**2)*(cos_p**2)+(cos_t**2)*(sin_p**2)+(cos_t**2)*(cos_p**2))**0.5
    x = (sin_t * cos_p) / r
    y = (cos_t * sin_p) / r
    z = (cos_t * cos_p) / r
    ray = torch.cat((x[...,None], y[...,None], z[...,None]), dim=-1).flatten(0,-2)

    return ray, theta_arr, phi_arr

def focal2halffov2(focal, pixels):
    return pixels / 2 / focal

def colmap_main(args):
    root_dir = args.path
    camera_dir = Path(root_dir) / "colmap" / "cameras_fish.txt"
    input_image_dir = Path(root_dir) / args.src
    out_image_dir = Path(root_dir) / args.dst
    
    _, _, width, height, params = read_intrinsics_text(camera_dir)

    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]

    FoVx = focal2halffov2(fx, width) 
    FoVy = focal2halffov2(fy, height) 
    print("FOVx in deg: ", 2 * FoVx * 180 / np.pi)
    print("FOVy in deg: ", 2 * FoVy * 180 / np.pi)
    tan_theta, tan_phi = fov2tan(FoVx, FoVy, 5e-3)
    
    distortion_params = params[4:]
    kk = distortion_params
    
    frames = sorted(os.listdir(input_image_dir))
    radius = np.sqrt(tan_theta ** 2 + tan_phi ** 2)
    theta = np.arctan(radius)
    r = theta * (1.0 + kk[0] * theta**2 + kk[1] * theta**4 + kk[2] * theta**6 + kk[3] * theta**8)
    u = tan_theta * r * fx / radius + cx
    v = tan_phi * r * fy / radius + cy
    u_mask = np.logical_and(u >= 0, u < width)
    v_mask =  np.logical_and(v >= 0, v < height) 
    valid_mask = u_mask & v_mask
    valid_mask = (valid_mask).astype(np.uint8)
    out_image_path = Path(out_image_dir) / "fov_0.75_step_2e-3_mask.png"
    cv2.imwrite(str(out_image_path), valid_mask * 255)

    u, v = u.astype(np.float32), v.astype(np.float32)

    for frame in tqdm(frames, desc="frame"):
        image_path = Path(input_image_dir) / frame
        image = cv2.imread(str(image_path))

        FOV_image = cv2.remap(
            image,
            u,
            v,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        out_image_path = Path(out_image_dir) / frame
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        FOV_image = FOV_image * (valid_mask[:,:,None])
        FOV_image = FOV_image.astype(np.uint8)
        cv2.imwrite(str(out_image_path), FOV_image)

        ## Compute backward mapping for checking and converting back to EQ fisheye
        # mapx = np.zeros((width, height), dtype=np.float32)
        # mapy = np.zeros((width, height), dtype=np.float32)
        # for i in tqdm(range(0, width), desc="calculate_maps_inverse"):
        #     for j in range(0, height):
        #         x = float(i)
        #         y = float(j)
        #         x1 = (x - cx) / fx
        #         y1 = (y - cy) / fy
        #         psi = np.sqrt(x1 **2 + y1 ** 2)
        #         if np.abs(psi) < 1e-7:
        #             psi = 1e-7
        #         tan_psi = np.tan(psi)
        #         x2 = np.arctan(tan_psi * x1 / psi) / 8e-4
        #         y2 = np.arctan(tan_psi * y1 / psi) / 8e-4
        #         mapx[i, j] = x2
        #         mapy[i, j] = y2
        
        # mapx += (FOV_image.shape[1] - 1) / 2
        # mapy += (FOV_image.shape[0] - 1)/ 2
        # #map = np.stack((mapx, mapy), axis=-1)
        
        # back_image = cv2.remap(
        #     FOV_image,
        #     mapx.T,
        #     mapy.T,
        #     interpolation=cv2.INTER_LINEAR,
        #     borderMode=cv2.BORDER_REFLECT_101,
        # )
        # cv2.imwrite('backward.png', back_image)

        # ref = cv2.imread('/home/choyingw/Documents/0221_clone/gaussian-splatting/datasets/scannetpp_data1/0a5c013435/dslr/image_undistorted_fisheye_original/DSC01752.JPG', -1)
        # score = psnr(back_image, ref)
        # print(score)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/choyingw/Documents/0221_clone/gaussian-splatting/datasets/scannetpp_data1/0a5c013435/dslr/")
    parser.add_argument('--src', type=str, default="resized_images")
    parser.add_argument('--dst', type=str, default="image_undistorted_fisheye_fov7")
    args = parser.parse_args()
    colmap_main(args)