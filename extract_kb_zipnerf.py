import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from scene.colmap_loader import read_intrinsics_binary
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
                params = np.array(tuple(map(float, elems[4:])))
    return camera_id, model, width, height, params

def focal2halffov2(focal, pixels):
    return pixels / 2 / focal

def colmap_main(args):
    root_dir = args.path
    camera_dir = Path(root_dir) / "sparse" / "0" / "cameras.bin"
    input_image_dir = args.src
    out_image_dir = args.dst
    
    _, _, width, height, params = read_intrinsics_binary(camera_dir)
    print(params)
    
    # adjust fx, fy, cx, cy by the actual image size
    if args.r == -1:
        ratio = 1.0
    else:
        ratio = 1 / args.r
    
    fx = params[0] * ratio
    fy = params[1] * ratio
    print("ratio", ratio)

    # FoVx = min(focal2halffov2(fx, width) * args.fov_mod, np.pi / 2)
    # FoVy = min(focal2halffov2(fy, height) * args.fov_mod, np.pi / 2)
    # FoVx = min(focal2halffov2(fx, width * ratio) * args.fov_mod, np.pi / 2)
    # FoVy = min(focal2halffov2(fy, height * ratio) * args.fov_mod, np.pi / 2)
    FoVx = min(focal2halffov2(fx, width * ratio) * args.fov_mod, np.pi / 2)
    FoVy = min(focal2halffov2(fy, height * ratio) * args.fov_mod, np.pi / 2)
    print("FOVx in deg: ", 2 * FoVx * 180 / np.pi)
    print("FOVy in deg: ", 2 * FoVy * 180 / np.pi)

    width = int(width * ratio)
    height = int(height * ratio)
    
    # Use prepared fisheye grid map by DAC https://github.com/yuliangguo/depth_any_camera
    try:
        grid_map_file = Path(args.path) / "grid_fisheye.npy"
        grid_fisheye = np.load(grid_map_file)
    except:
        if root_dir[-1] == "/":
            root_dir = root_dir[:-1]
        seq_name = os.path.basename(root_dir)
        grid_fisheye = np.load(f"./gridmap/zipnerf/{seq_name}/grid_fisheye.npy")

    #grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (width, height), interpolation=cv2.INTER_NEAREST)
    grid_isnan = cv2.resize(grid_fisheye[:, :, 3], (width, height), interpolation=cv2.INTER_LINEAR)
    grid_fisheye = cv2.resize(grid_fisheye[:, :, :3], (width, height), interpolation=cv2.INTER_LINEAR)
    grid_fisheye = np.concatenate([grid_fisheye, grid_isnan[:, :, None]], axis=2)
    
    # Reverse warping
    reverse_mapx = np.zeros((width, height), dtype=np.float32)
    reverse_mapy = np.zeros((width, height), dtype=np.float32)
    # More exact reverse warping using grid_fisheye
    for i in tqdm(range(0, width), desc="calculate_reverse_maps"):
        for j in range(0, height):
            X_c = grid_fisheye[j, i, 0]
            Y_c = grid_fisheye[j, i, 1]
            Z_c = grid_fisheye[j, i, 2]
            tan_theta = X_c / Z_c
            tan_phi = Y_c / Z_c
            
            theta = np.arctan(tan_theta)
            phi = np.arctan(tan_phi)
            
            x2 = (theta + FoVx) / args.step
            y2 = (phi + FoVy) / args.step
            reverse_mapx[i, j] = x2
            reverse_mapy[i, j] = y2
    frames = os.listdir(input_image_dir)

    for frame in tqdm(frames, desc="frame"):
        image_path = Path(input_image_dir) / frame
        undistorted_image = cv2.imread(str(image_path))
        
        reversed_image = cv2.remap(
            undistorted_image,
            reverse_mapx.T,
            reverse_mapy.T,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        reversed_image_path = Path(out_image_dir) / frame
        reversed_image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(reversed_image_path), reversed_image)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-r', type=int, default=-1)

    parser.add_argument('--path', type=str, default="/media/scannetpp/demo/0a5c013435/dslr/")
    parser.add_argument('--src', type=str, default="undistorted_fovmaps")
    parser.add_argument('--dst', type=str, default="remapped_fisheye")
    parser.add_argument('--step', type=float, default=2e-3)
    parser.add_argument('--fov_mod', type=float, default=1.3)
    args = parser.parse_args()
    colmap_main(args)