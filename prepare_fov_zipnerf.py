# This is the code to prepare the FOV images from the scannet fisheye images.

import os
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
# from utils.graphics_utils import focal2fov
# import torch
import shutil
import struct
import collections


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            params = np.array(tuple(map(np.float64, params)))
    return camera_id, model_name, width, height, params

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

def focal2halffov2(focal, pixels):
    return pixels / 2 / focal

def colmap_main(args):
    root_dir = args.path
    camera_dir = Path(root_dir) / "sparse" / "0" / "cameras.bin"
    input_image_dir = Path(root_dir) / args.src
    out_image_dir = Path(root_dir) / args.dst
    
    _, _, width, height, params = read_intrinsics_binary(camera_dir)

    fx = params[0]
    fy = params[1]
    cx = params[2]
    cy = params[3]

    FoVx = min(focal2halffov2(fx, width) * args.fov_mod, np.pi / 2)
    FoVy = min(focal2halffov2(fy, height) * args.fov_mod, np.pi / 2)
    print("FOVx in deg: ", 2 * FoVx * 180 / np.pi)
    print("FOVy in deg: ", 2 * FoVy * 180 / np.pi)
    tan_theta, tan_phi = fov2tan(FoVx, FoVy, args.step)
    
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

    if valid_mask is not None:
        mask_output_path = Path(out_image_dir) / args.mask_dst
        mask_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mask_output_path), valid_mask.astype(np.uint8) * 255)
        print("Save mask to:", mask_output_path, "with shape: ", valid_mask.shape)
    else:
        print("Warning: valid_mask is None")

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
    parser.add_argument('--path', type=str, default="/media/scannetpp/0a5c013435/dslr/")
    parser.add_argument('--src', type=str, default="images")
    parser.add_argument('--dst', type=str, default="undistorted_fovmaps")
    parser.add_argument('--mask_dst', type=str, default=None)
    parser.add_argument('--step', type=float, default=None)
    parser.add_argument('--fov_mod', type=float, default=None)
    args = parser.parse_args()
    colmap_main(args)