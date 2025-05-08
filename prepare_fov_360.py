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

def fov2tan_EQUREC(fovx, fovy, interval, xi=1.0):
    # create a grid of theta and phi values (omni)
    omni_theta_arr = np.arange(interval / 2, fovx, interval)
    omni_theta_arr = np.sort(np.concatenate((-omni_theta_arr, omni_theta_arr)))
    omni_phi_arr = np.arange(interval / 2, fovy, interval)
    omni_phi_arr = np.sort(np.concatenate((-omni_phi_arr, omni_phi_arr)))
    omni_theta_map, omni_phi_map = np.meshgrid(omni_theta_arr, omni_phi_arr, indexing = 'xy')

    # get the tan values of incidence angles (omni)
    omnitan_theta_map = np.tan(omni_theta_map)
    omnitan_phi_map = np.tan(omni_phi_map)
    omnitan_incident_angle_map = np.sqrt(omnitan_theta_map * omnitan_theta_map + omnitan_phi_map * omnitan_phi_map)
    omni_incident_angle_map = np.arctan(omnitan_incident_angle_map) # from 0 to pi/2

    # the incident angle map is the twice of the omni incident angle map (only when xi=1.0)
    if xi == 1.0:
        incident_angle_map = 2.0 * omni_incident_angle_map # from 0 to pi
    else:
        assert xi >= 0.0, "xi should be positive"
        assert xi < 1.0, "xi should be less than 1.0, otherwise the ray starting from the mirror point will have two intersections with the unit sphere"
        incident_angle_map = omni_incident_angle_map + np.arcsin(xi * np.sin(omni_incident_angle_map))

    tan_incident_angle_map = np.tan(incident_angle_map) # negative from pi/2 to pi, z < 0
    # get the tan values of incidence angles (fov)
    tan_theta_map = tan_incident_angle_map * omnitan_theta_map / omnitan_incident_angle_map
    tan_phi_map = tan_incident_angle_map * omnitan_phi_map / omnitan_incident_angle_map

    # calculate the phi map (erp)
    z_signal = np.where(tan_incident_angle_map > 0.0, 1, -1)
    erptan_phi_map = -tan_phi_map / np.sqrt(tan_theta_map * tan_theta_map + 1) * z_signal
    erp_phi_map = np.arctan(erptan_phi_map)

    x_signal = np.where(tan_theta_map > 0.0, 1, -1) * z_signal
    erp_theta_map = np.arctan(tan_theta_map) + np.where(z_signal > 0.0, 0, np.pi) * x_signal

    return erp_theta_map, erp_phi_map

def colmap_main(args):
    root_dir = args.path
    input_image_dir = Path(root_dir) / args.src
    out_image_dir = Path(root_dir) / args.dst
    
    FoVx = np.pi /3
    FoVy = np.pi /3
    xi = 0.5
    print("omni FOVx in deg: ", 2 * FoVx * 180 / np.pi)
    print("omni FOVy in deg: ", 2 * FoVy * 180 / np.pi)
    print("mirror param: ", xi)

    theta_arr, phi_arr = fov2tan_EQUREC(FoVx, FoVy, args.step, xi=xi)

    frames = sorted(os.listdir(input_image_dir))
    example_image_path = Path(input_image_dir) / frames[0]
    example_image = cv2.imread(str(example_image_path), -1)
    height, width, _ = example_image.shape

    u = width / (2 * np.pi) * theta_arr + width / 2
    v = -height / (np.pi) * phi_arr + height / 2
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
            borderMode=cv2.BORDER_CONSTANT,
        )
        out_image_path = Path(out_image_dir) / frame
        out_image_path.parent.mkdir(parents=True, exist_ok=True)
        FOV_image = FOV_image.astype(np.uint8)
        FOV_image = FOV_image * (valid_mask[:,:,None])
        cv2.imwrite(str(out_image_path), FOV_image)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, default="/media/scannetpp/0a5c013435/dslr/")
    parser.add_argument('--src', type=str, default="images")
    parser.add_argument('--dst', type=str, default="undistorted_fovmaps")
    parser.add_argument('--mask_dst', type=str, default="fov_0.75_step_2e-3_mask.png")
    parser.add_argument('--step', type=float, default=2e-3)
    parser.add_argument('--fov_mod', type=float, default=1.3)
    parser.add_argument('--resize_ratio', type=float, default=1.0)
    args = parser.parse_args()
    colmap_main(args)