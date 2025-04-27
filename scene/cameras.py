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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal
from utils.general_utils import PILtoTorch
import cv2

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, step,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",  xi=1.0
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        # Use Full FOV in original angle
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.invdepthmap = None
        self.depth_reliable = False
        
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # for ray-splatting start
        # Change the step adjust resolution
        # Convert to omni angle
        FoVx_omni = FoVx / 2
        FoVy_omni = FoVy / 2
        omni_theta_arr = np.arange(step / 2, FoVx_omni / 2, step)
        omni_theta_arr = np.sort(np.concatenate((-omni_theta_arr, omni_theta_arr)))
        omni_phi_arr = np.arange(step / 2, FoVy_omni / 2, step)
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
        self.tan_theta = torch.Tensor(tan_theta_map).to(self.data_device).float()
        self.tan_phi = torch.Tensor(tan_phi_map).to(self.data_device).float()
        self.omni_tan_theta = torch.Tensor(omni_theta_arr).to(self.data_device).float()
        self.omni_tan_phi = torch.Tensor(omni_phi_arr).to(self.data_device).float()
        self.sampled_image = self.original_image

        # print("CAV", self.omni_tan_theta)
        # print("fefe", self.omni_tan_phi)


        # arr_theta, arr_phi = self.fov_sample2ray(FoVx/2, FoVy/2, step)
        # cos_theta = torch.cos(arr_theta)
        # cos_phi = torch.cos(arr_phi)
        
        # cos_theta = torch.where(torch.abs(cos_theta) < 1e-7, torch.full_like(cos_theta, 1e-7), cos_theta).to(self.data_device)
        # cos_phi = torch.where(torch.abs(cos_phi) < 1e-7, torch.full_like(cos_phi, 1e-7), cos_phi).to(self.data_device)
        # self.tan_theta = torch.tan(arr_theta).to(self.data_device)
        # self.tan_phi = torch.tan(arr_phi).to(self.data_device)
        # self.omni_tan_theta = self.omni_map_z(self.tan_theta, cos_theta).float()
        # self.omni_tan_phi = self.omni_map_z(self.tan_phi, cos_phi).float()
        # self.sampled_image = self.original_image

    @staticmethod
    def project_to_fovmap(sampled_rays, image, fx, fy, cx, cy, depth=None):
        u = (sampled_rays[:, 0] / sampled_rays[:, 2]) * fx + cx
        v = (sampled_rays[:, 1] / sampled_rays[:, 2]) * fy + cy
        u, v = u.long(), v.long()
        sampled_image = image[:, v, u]
        sampled_depth = None
        if depth is not None:
            sampled_depth = depth[v, u]
        
        return sampled_image, sampled_depth
    
    @staticmethod
    def project_to_fovmap_scannetpp(sampled_rays, image, fx, fy, cx, cy, depth=None):
        u = (sampled_rays[:, 0] / sampled_rays[:, 2]) * fx + cx
        v = (sampled_rays[:, 1] / sampled_rays[:, 2]) * fy + cy
        u, v = u.long(), v.long()
        sampled_image = image[:, v, u]
        sampled_depth = None
        if depth is not None:
            sampled_depth = depth[v, u]
        
        return sampled_image, sampled_depth

    @staticmethod
    def fov_sample2ray(fovx, fovy, interval):
        theta_arr = torch.arange(interval / 2, fovx, interval)
        theta_arr, _ = torch.sort(torch.cat((-theta_arr, theta_arr)))
        phi_arr = torch.arange(interval / 2, fovy, interval)
        phi_arr, _ = torch.sort(torch.cat((-phi_arr, phi_arr)))

        return theta_arr.float(), phi_arr.float()

    @staticmethod
    def omni_map_z(m, z, xi=0.0): #1.1
        return m / (1+xi*(z/(torch.abs(z)))*(1+m**2)**0.5)
        
class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, sample_step):
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
        self.sample_step = sample_step
        arr_theta, arr_phi = self.fov_sample2ray(self.FoVx/2, self.FoVy/2, sample_step)
        
        cos_theta = torch.cos(arr_theta)
        cos_phi = torch.cos(arr_phi)
        
        cos_theta = torch.where(torch.abs(cos_theta) < 1e-7, torch.full_like(cos_theta, 1e-7), cos_theta).to(self.data_device)
        cos_phi = torch.where(torch.abs(cos_phi) < 1e-7, torch.full_like(cos_phi, 1e-7), cos_phi).to(self.data_device)
        self.tan_theta = torch.tan(arr_theta).to(self.data_device)
        self.tan_phi = torch.tan(arr_phi).to(self.data_device)
        self.omni_tan_theta = self.omni_map_z(self.tan_theta, cos_theta)
        self.omni_tan_phi = self.omni_map_z(self.tan_phi, cos_phi)

    @staticmethod
    def fov_sample2ray(fovx, fovy, interval):
        theta_arr = torch.arange(interval / 2, fovx, interval)
        theta_arr, _ = torch.sort(torch.cat((-theta_arr, theta_arr)))
        phi_arr = torch.arange(interval / 2, fovy, interval)
        phi_arr, _ = torch.sort(torch.cat((-phi_arr, phi_arr)))
        
        return theta_arr.float(), phi_arr.float()

    @staticmethod
    def omni_map_z(m, z, xi=0.0): #1.1
        return m / (1+xi*(z/(torch.abs(z)))*(1+m**2)**0.5)
    
    def get_viewpoint_mask(self, ref_camera_dir):
        from prepare_fov import read_intrinsics_text, fov2tan
        _, _, width, height, params = read_intrinsics_text(ref_camera_dir)
        fx = params[0]
        fy = params[1]
        cx = params[2]
        cy = params[3]
        distortion_params = params[4:]
        kk = distortion_params
        tan_theta, tan_phi = fov2tan(self.FoVx/2, self.FoVy/2, self.sample_step)
        radius = np.sqrt(tan_theta ** 2 + tan_phi ** 2)
        theta = np.arctan(radius)
        r = theta * (1.0 + kk[0] * theta**2 + kk[1] * theta**4 + kk[2] * theta**6 + kk[3] * theta**8)
        u = tan_theta * r * fx / radius + cx
        v = tan_phi * r * fy / radius + cy
        u_mask = np.logical_and(u >= 0, u < width)
        v_mask =  np.logical_and(v >= 0, v < height) 
        valid_mask = u_mask & v_mask
        self.valid_mask = (valid_mask).astype(np.uint8)
        return self.valid_mask

