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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, resolution, colmap_id, R, T, FoVx, FoVy, depth_params, image, invdepthmap,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 train_test_exp = False, is_test_dataset = False, is_test_view = False
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        resized_image_rgb = PILtoTorch(image, resolution)
        gt_image = resized_image_rgb[:3, ...]
        self.alpha_mask = None
        if resized_image_rgb.shape[0] == 4:
            self.alpha_mask = resized_image_rgb[3:4, ...].to(self.data_device)
        else: 
            self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))

        if train_test_exp and is_test_view:
            if is_test_dataset:
                self.alpha_mask[..., :self.alpha_mask.shape[-1] // 2] = 0
            else:
                self.alpha_mask[..., self.alpha_mask.shape[-1] // 2:] = 0

        self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        self.invdepthmap = None
        self.depth_reliable = False
        if invdepthmap is not None:
            self.depth_mask = torch.ones_like(self.alpha_mask)
            self.invdepthmap = cv2.resize(invdepthmap, resolution)
            self.invdepthmap[self.invdepthmap < 0] = 0
            self.depth_reliable = True

            if depth_params is not None:
                if depth_params["scale"] < 0.2 * depth_params["med_scale"] or depth_params["scale"] > 5 * depth_params["med_scale"]:
                    self.depth_reliable = False
                    self.depth_mask *= 0
                
                if depth_params["scale"] > 0:
                    self.invdepthmap = self.invdepthmap * depth_params["scale"] + depth_params["offset"]

            if self.invdepthmap.ndim != 2:
                self.invdepthmap = self.invdepthmap[..., 0]
            self.invdepthmap = torch.from_numpy(self.invdepthmap[None]).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # for ray-splatting start
    #     sampled_rays, arr_theta, arr_phi = self.fov_sample2ray(FoVx/2, FoVy/2, 5e-3)
    #     self.sampled_rays = sampled_rays
    #     cos_theta = torch.cos(arr_theta)
    #     cos_phi = torch.cos(arr_phi)
        
    #     cos_theta = torch.where(torch.abs(cos_theta) < 1e-7, torch.full_like(cos_theta, 1e-7), cos_theta).cuda()
    #     cos_phi = torch.where(torch.abs(cos_phi) < 1e-7, torch.full_like(cos_phi, 1e-7), cos_phi).cuda()
    #     self.tan_theta = torch.tan(arr_theta).cuda()
    #     self.tan_phi = torch.tan(arr_phi).cuda()
    #     self.omni_tan_theta = self.omni_map_z(self.tan_theta, cos_theta)
    #     self.omni_tan_phi = self.omni_map_z(self.tan_phi, cos_phi)
        
    #     # init_from_dataset() only
    #     if self.original_image is not None:
    #         self.sampled_image, _ = self.project_to_fovmap(
    #             self.sampled_rays, 
    #             self.original_image,  
    #             self.fx, self.fy, self.cx, self.cy
    #             )
    #         self.sampled_image = self.sampled_image.reshape(-1, self.tan_phi.shape[0], self.tan_theta.shape[0])

    # @staticmethod
    # def project_to_fovmap(sampled_rays, image, fx, fy, cx, cy, depth=None):
    #     u = (sampled_rays[:, 0] / sampled_rays[:, 2]) * fx + cx
    #     v = (sampled_rays[:, 1] / sampled_rays[:, 2]) * fy + cy
    #     u, v = u.long(), v.long()
    #     sampled_image = image[:, v, u]
    #     sampled_depth = None
    #     if depth is not None:
    #         sampled_depth = depth[v, u]
        
    #     return sampled_image, sampled_depth

    # @staticmethod
    # def fov_sample2ray(fovx, fovy, interval):
    #     theta_arr = torch.arange(interval / 2, fovx, interval).float()
    #     theta_arr, _ = torch.sort(torch.cat((-theta_arr, theta_arr)))
    #     phi_arr = torch.arange(interval / 2, fovy, interval).float()
    #     phi_arr, _ = torch.sort(torch.cat((-phi_arr, phi_arr)))

    #     sin_t = torch.sin(theta_arr)
    #     cos_t = torch.cos(theta_arr)
    #     sin_p = torch.sin(phi_arr).unsqueeze(1)
    #     cos_p = torch.cos(phi_arr).unsqueeze(1)

    #     r = ((sin_t**2)*(cos_p**2)+(cos_t**2)*(sin_p**2)+(cos_t**2)*(cos_p**2))**0.5
    #     x = (sin_t * cos_p) / r
    #     y = (cos_t * sin_p) / r
    #     z = (cos_t * cos_p) / r
    #     ray = torch.cat((x[...,None], y[...,None], z[...,None]), dim=-1).to('cuda').flatten(0,-2)

    #     return ray, theta_arr, phi_arr

    # @staticmethod
    # def omni_map_z(m, z, xi=0.0): #1.1
    #     return m / (1+xi*(z/(torch.abs(z)))*(1+m**2)**0.5)
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

