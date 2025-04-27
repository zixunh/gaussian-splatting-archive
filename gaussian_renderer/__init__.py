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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, use_trained_exp=False, fetch_prev_next_exp=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) 
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    omni_tanfovx = math.tan(viewpoint_camera.FoVx * 0.5 * 0.5) 
    omni_tanfovy = math.tan(viewpoint_camera.FoVy * 0.5 * 0.5) 
    # print("DW", viewpoint_camera.FoVx, tanfovx)
    # exit()

    scaling_modifier = scaling_modifier
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.omni_tan_phi.shape[0]), # for ray-splatting
        image_width=int(viewpoint_camera.omni_tan_theta.shape[0]), # for ray-splatting
        tanfovx=omni_tanfovx,
        tanfovy=omni_tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        # projmatrix=viewpoint_camera.full_proj_transform,
        omni_tan_theta=viewpoint_camera.omni_tan_theta.cuda(), # for ray-splatting
        omni_tan_phi=viewpoint_camera.omni_tan_phi.cuda(), # for ray-splatting
        tan_theta=viewpoint_camera.tan_theta.cuda(), # for ray-splatting
        tan_phi=viewpoint_camera.tan_phi.cuda(), # for ray-splatting
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        antialiasing=pipe.antialiasing
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        print("We use the inverse-sigma S^(-1)Rt for rasterizer forwards and backwards; precomputing cov3D is disabled.")
        raise NotImplementedError
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth_image = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations)
        # cov3D_precomp = cov3D_precomp)
        
    if use_trained_exp: # Apply exposure to rendered image (training only)
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
    elif fetch_prev_next_exp: # Fetch nearest exposure to rendered image (testing)
        try:
            # +- 5 for london; otherwise +- 1
            fetch_prev = int(viewpoint_camera.image_name[3:]) - 1
            fetch_next = int(viewpoint_camera.image_name[3:]) + 1
            name_prev = f'indoor_DSC{fetch_prev:05d}.JPG'
            fetch_next = f'indoor_DSC{fetch_next:05d}.JPG'
            exposure_prev = exposure_next = None
            try:
                exposure_prev = pc.get_exposure_from_name(name_prev)
                exposure_next = pc.get_exposure_from_name(fetch_next)
                exposure = (exposure_prev + exposure_next) / 2
            except:
                if exposure_prev is not None:
                    exposure = exposure_prev
                else:
                    exposure = pc.get_exposure_from_name(fetch_next)
            rendered_image = torch.matmul(rendered_image.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1) + exposure[:3, 3,   None, None]
        except:
            print("Error fetching exposure for image:", viewpoint_camera.image_name)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0).nonzero(),
        "radii": radii,
        "depth" : depth_image
        }
    
    return out
