import torch
import torch.nn as nn
from torchvision import transforms as T

from utils import config_parser
from model_k.net import MLP, UNet, AFNet, KFN

from shencoder import SHEncoder
from gridencoder import GridEncoder



"""
v1:
    feat_bias = self.mlp_rect(torch.cat([self.enc_grid(feat_o), self.enc_sh(feat_dirs)], dim=-1))

v2:
    feat_bias = self.mlp_rect(torch.cat([self.enc_sh(feat_o+feat_dirs)], dim=-1)) 

v3:
    feat_bias = self.mlp_rect(torch.cat([self.enc_sh(feat_dirs)], dim=-1)) 
"""


class Renderer_k_v1(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """
    def __init__(self, args):
        super(Renderer_k_v1, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        self.kfn = KFN(args).to(args.device)
        self.unet = UNet(args).to(args.device)

        self.enc_grid = GridEncoder(input_dim=3, 
                    num_levels=16, 
                    level_dim=2, 
                    base_resolution=16, 
                    log2_hashmap_size=19,
                    per_level_scale=1.447,
                    gridtype='hash').to(args.device) # outputdim 32

        self.enc_sh = SHEncoder(degree=4).to(args.device) # outputdim 16

        self.mlp_rect = nn.Sequential(
            nn.Linear(32+16, 64),
            # nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, args.dim),
        ).to(args.device)

        self.dim = args.dim
        self.use_crop = args.use_crop

        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)
        else:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)

        self.pad_w = T.Pad(args.pad, 1., 'constant')
        self.pad_b = T.Pad(args.pad, -1., 'constant')

        self.xyznear = args.xyznear  # bool
        self.mask = args.pix_mask
        self.train_size = args.train_size
        self.points_per_pixel = args.points_per_pixel

    def forward(self, zbufs, idbufs, ray, gt, mask_gt, isTrain, xyz_o):
        """
        Args:
            zbuf: z-buffer from rasterization (index buffer when xyznear is True)
            ray: ray direction map
            gt: gt image (used in training to maintain consistent cropping and resizing with input)
            mask_gt: gt mask (used in dtu dataset)
            isTrain: train mode or not
            xyz_o: world coordinates of point clouds (used when xyzenar is True)

        Output:
            img: rendered image
            gt: gt image after cropping and resizing
            mask_gt: gt mask after cropping and resizing 
        """
        H, W, P = zbufs.shape  # [H, W, points_per_pixel]
        o = ray[..., :3]  # [H, W, 3]
        o = o[0, 0, :] # [3]
        dirs = ray[..., 3:6]  # [H, W, 3]
        cos = ray[..., -1:].expand(H, W, P)  # [H, W, P]

        if self.use_crop and isTrain:
            ray_pad = self.pad_w(ray.permute(2, 0, 1).unsqueeze(0))
            gt_pad = self.pad_w(gt.permute(2, 0, 1).unsqueeze(0))
            zbufs_pad = self.pad_b(zbufs.permute(2, 0, 1).unsqueeze(0))
            idbufs_pad = self.pad_b(idbufs.permute(2, 0, 1).unsqueeze(0))

            if mask_gt is not None:
                mask_gt = mask_gt.permute(2, 0, 1).unsqueeze(0)
                cat_img = torch.cat([ray_pad, gt_pad, mask_gt, idbufs_pad, zbufs_pad], dim=1)
            else:
                cat_img = torch.cat([ray_pad, gt_pad, idbufs_pad, zbufs_pad], dim=1)

            # [1, _, train_size, train_size]
            cat_img = self.randomcrop(cat_img)
            _, _, H, W = cat_img.shape
            # o_crop = cat_img[0, :3].permute(1, 2, 0)
            dirs_crop = cat_img[0, 3:6].permute(1, 2, 0)
            cos_crop = cat_img[0, 6:7].permute(1, 2, 0)
            gt_crop = cat_img[0, 7:10].permute(1, 2, 0)
            gt = gt_crop
            
            if mask_gt is not None:
                mask_gt = cat_img[0, 10:11].permute(1, 2, 0)
                idbufs_crop = cat_img[0, 11:11+P].permute(1, 2, 0)
                zbufs_crop = cat_img[0, 11+P:].permute(1, 2, 0)
            else:
                idbufs_crop = cat_img[0, 10:10+P].permute(1, 2, 0)
                zbufs_crop = cat_img[0, 10+P:].permute(1, 2, 0)

            zbufs = zbufs_crop.clone()
            idbufs = idbufs_crop.clone()
            dirs = dirs_crop.clone() # h w 3
            cos = cos_crop.clone().expand(H, W, P) # h w 1 -> h w P

            # [TODO] why this happend
            # if comment two lines, GPU memory increased dramatically more than 20GB
            # if not 6-7GB
            del cat_img, ray_pad, gt_pad, zbufs_pad, idbufs_crop
            torch.cuda.empty_cache()

        feature_map = torch.zeros([H, W, P, self.dim], device=zbufs.device)
        feature_map_bias = torch.zeros([H, W, self.dim], device=zbufs.device)

        if isTrain:
            pixel_mask = zbufs > 0.2 # 800 800 P
        else:
            pixel_mask = zbufs > 0 # 800 800 P

        
        occ_id_buf = idbufs[pixel_mask]
        occ_z_buf = zbufs[pixel_mask]
        occ_cos = cos[pixel_mask]
        dirs_x = dirs[:, :, 0].unsqueeze(-1).expand(cos.shape)[pixel_mask]
        dirs_y = dirs[:, :, 1].unsqueeze(-1).expand(cos.shape)[pixel_mask]
        dirs_z = dirs[:, :, 2].unsqueeze(-1).expand(cos.shape)[pixel_mask]

        # Computate the occupied pixel index by unique 3d point
        u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
        inv_sorted = inverse_indices.argsort(stable=True)
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]

        u_elements_z = occ_z_buf[index].unsqueeze(-1)
        u_cos = occ_cos[index].unsqueeze(-1)
        dx = dirs_x[index]
        dy = dirs_y[index]
        dz = dirs_z[index]
        dir_u = torch.cat([dx.unsqueeze(1), dy.unsqueeze(1), dz.unsqueeze(1)], dim=1)
        xyz_u = o + u_elements_z / u_cos * dir_u
        unique_pixel_mask = torch.isin(idbufs, u_elements)

        #### 1. Generate mlp feature
        feature = self.mlp(xyz_u, dir_u) # [unique.shape[0], dim]
        feature_map[unique_pixel_mask] = feature[inverse_indices, :]

        #### 2. Generate mlp_rect feature
        feat_o = o.repeat(H, W, 1)
        feat_dirs = dirs/cos[..., 0].unsqueeze(-1) # [H, W, 3]
        surface_mask = pixel_mask[..., 0]
        feat_o = feat_o[surface_mask]
        feat_dirs = feat_dirs[surface_mask]
        
        feat_bias = self.mlp_rect(torch.cat([self.enc_grid(feat_o), self.enc_sh(feat_dirs)], dim=-1))
        feature_map_bias[surface_mask] = feat_bias

        #### 3. Feature addition
        feature_map += feature_map_bias.unsqueeze(-2)
        feature_map = feature_map.reshape(H, W, P*self.dim).permute(2, 0, 1).unsqueeze(0)
        # 13ms
        # del xyz_u, dir_u
        # torch.cuda.empty_cache()

        #### 4. Fuse K feature maps
        # [1, self.points_per_pixel * self.dim, H, W]
        # TODO: pred_mask to regurlize
        denoise_fm, pred_mask = self.kfn(feature_map)
        
        ret = self.unet(denoise_fm)  # [1, 3, H, W]

        # For dtu dataset, self.mask that distinguishes foreground and background
        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,3,0,1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # [1, 3, H, W]

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
        return {'img': img, 'gt': gt, 'mask_gt': mask_gt, 'pred_mask':pred_mask}

class Renderer_k_v2(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """
    def __init__(self, args):
        super(Renderer_k_v2, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.denc_shim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        self.kfn = KFN(args).to(args.device)
        self.unet = UNet(args).to(args.device)

        self.enc_sh = SHEncoder(degree=4).to(args.device) # outputdim 16

        self.mlp_rect = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, args.dim),
        ).to(args.device)

        self.dim = args.dim
        self.use_crop = args.use_crop

        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)
        else:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)

        self.pad_w = T.Pad(args.pad, 1., 'constant')
        self.pad_b = T.Pad(args.pad, -1., 'constant')

        self.xyznear = args.xyznear  # bool
        self.mask = args.pix_mask
        self.train_size = args.train_size
        self.points_per_pixel = args.points_per_pixel

    def forward(self, zbufs, idbufs, ray, gt, mask_gt, isTrain, xyz_o):
        """
        Args:
            zbuf: z-buffer from rasterization (index buffer when xyznear is True)
            ray: ray direction map
            gt: gt image (used in training to maintain consistent cropping and resizing with input)
            mask_gt: gt mask (used in dtu dataset)
            isTrain: train mode or not
            xyz_o: world coordinates of point clouds (used when xyzenar is True)

        Output:
            img: rendered image
            gt: gt image after cropping and resizing
            mask_gt: gt mask after cropping and resizing 
        """
        H, W, P = zbufs.shape  # [H, W, points_per_pixel]
        o = ray[..., :3]  # [H, W, 3]
        o = o[0, 0, :] # [3]
        dirs = ray[..., 3:6]  # [H, W, 3]
        cos = ray[..., -1:].expand(H, W, P)  # [H, W, P]

        if self.use_crop and isTrain:
            ray_pad = self.pad_w(ray.permute(2, 0, 1).unsqueeze(0))
            gt_pad = self.pad_w(gt.permute(2, 0, 1).unsqueeze(0))
            zbufs_pad = self.pad_b(zbufs.permute(2, 0, 1).unsqueeze(0))
            idbufs_pad = self.pad_b(idbufs.permute(2, 0, 1).unsqueeze(0))

            if mask_gt is not None:
                mask_gt = mask_gt.permute(2, 0, 1).unsqueeze(0)
                cat_img = torch.cat([ray_pad, gt_pad, mask_gt, idbufs_pad, zbufs_pad], dim=1)
            else:
                cat_img = torch.cat([ray_pad, gt_pad, idbufs_pad, zbufs_pad], dim=1)

            # [1, _, train_size, train_size]
            cat_img = self.randomcrop(cat_img)
            _, _, H, W = cat_img.shape
            # o_crop = cat_img[0, :3].permute(1, 2, 0)
            dirs_crop = cat_img[0, 3:6].permute(1, 2, 0)
            cos_crop = cat_img[0, 6:7].permute(1, 2, 0)
            gt_crop = cat_img[0, 7:10].permute(1, 2, 0)
            gt = gt_crop
            
            if mask_gt is not None:
                mask_gt = cat_img[0, 10:11].permute(1, 2, 0)
                idbufs_crop = cat_img[0, 11:11+P].permute(1, 2, 0)
                zbufs_crop = cat_img[0, 11+P:].permute(1, 2, 0)
            else:
                idbufs_crop = cat_img[0, 10:10+P].permute(1, 2, 0)
                zbufs_crop = cat_img[0, 10+P:].permute(1, 2, 0)

            zbufs = zbufs_crop.clone()
            idbufs = idbufs_crop.clone()
            dirs = dirs_crop.clone() # h w 3
            cos = cos_crop.clone().expand(H, W, P) # h w 1 -> h w P

            # [TODO] why this happend
            # if comment two lines, GPU memory increased dramatically more than 20GB
            # if not 6-7GB
            del cat_img, ray_pad, gt_pad, zbufs_pad, idbufs_crop
            torch.cuda.empty_cache()

        feature_map = torch.zeros([H, W, P, self.dim], device=zbufs.device)
        feature_map_bias = torch.zeros([H, W, self.dim], device=zbufs.device)

        if isTrain:
            pixel_mask = zbufs > 0.2 # 800 800 P
        else:
            pixel_mask = zbufs > 0 # 800 800 P

        
        occ_id_buf = idbufs[pixel_mask]
        occ_z_buf = zbufs[pixel_mask]
        occ_cos = cos[pixel_mask]
        dirs_x = dirs[:, :, 0].unsqueeze(-1).expand(cos.shape)[pixel_mask]
        dirs_y = dirs[:, :, 1].unsqueeze(-1).expand(cos.shape)[pixel_mask]
        dirs_z = dirs[:, :, 2].unsqueeze(-1).expand(cos.shape)[pixel_mask]

        # Computate the occupied pixel index by unique 3d point
        u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
        inv_sorted = inverse_indices.argsort(stable=True)
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]

        u_elements_z = occ_z_buf[index].unsqueeze(-1)
        u_cos = occ_cos[index].unsqueeze(-1)
        dx = dirs_x[index]
        dy = dirs_y[index]
        dz = dirs_z[index]
        dir_u = torch.cat([dx.unsqueeze(1), dy.unsqueeze(1), dz.unsqueeze(1)], dim=1)
        xyz_u = o + u_elements_z / u_cos * dir_u
        unique_pixel_mask = torch.isin(idbufs, u_elements)

        #### 1. Generate mlp feature
        feature = self.mlp(xyz_u, dir_u) # [unique.shape[0], dim]
        feature_map[unique_pixel_mask] = feature[inverse_indices, :]

        #### 2. Generate mlp_rect feature
        feat_o = o.repeat(H, W, 1)
        feat_dirs = dirs/cos[..., 0].unsqueeze(-1) # [H, W, 3]
        surface_mask = pixel_mask[..., 0]
        feat_o = feat_o[surface_mask]
        feat_dirs = feat_dirs[surface_mask]

        feat_bias = self.mlp_rect(torch.cat([self.enc_sh(feat_o+feat_dirs)], dim=-1)) 
        feature_map_bias[surface_mask] = feat_bias

        #### 3. Feature addition
        feature_map += feature_map_bias.unsqueeze(-2)
        feature_map = feature_map.reshape(H, W, P*self.dim).permute(2, 0, 1).unsqueeze(0)
        # 13ms
        # del xyz_u, dir_u
        # torch.cuda.empty_cache()

        #### 4. Fuse K feature maps
        # [1, self.points_per_pixel * self.dim, H, W]
        # TODO: pred_mask to regurlize
        denoise_fm, pred_mask = self.kfn(feature_map)
        
        ret = self.unet(denoise_fm)  # [1, 3, H, W]

        # For dtu dataset, self.mask that distinguishes foreground and background
        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,3,0,1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # [1, 3, H, W]

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
        return {'img': img, 'gt': gt, 'mask_gt': mask_gt, 'pred_mask':pred_mask}

class Renderer_k_v3(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """
    def __init__(self, args):
        super(Renderer_k_v3, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        self.kfn = KFN(args).to(args.device)
        self.unet = UNet(args).to(args.device)

        self.enc_sh = SHEncoder(degree=4).to(args.device) # outputdim 16

        self.mlp_rect = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, args.dim),
        ).to(args.device)

        self.dim = args.dim
        self.use_crop = args.use_crop

        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)
        else:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)

        self.pad_w = T.Pad(args.pad, 1., 'constant')
        self.pad_b = T.Pad(args.pad, -1., 'constant')

        self.xyznear = args.xyznear  # bool
        self.mask = args.pix_mask
        self.train_size = args.train_size
        self.points_per_pixel = args.points_per_pixel

    def forward(self, zbufs, idbufs, ray, gt, mask_gt, isTrain, xyz_o):
        """
        Args:
            zbuf: z-buffer from rasterization (index buffer when xyznear is True)
            ray: ray direction map
            gt: gt image (used in training to maintain consistent cropping and resizing with input)
            mask_gt: gt mask (used in dtu dataset)
            isTrain: train mode or not
            xyz_o: world coordinates of point clouds (used when xyzenar is True)

        Output:
            img: rendered image
            gt: gt image after cropping and resizing
            mask_gt: gt mask after cropping and resizing 
        """
        H, W, P = zbufs.shape  # [H, W, points_per_pixel]
        o = ray[..., :3]  # [H, W, 3]
        o = o[0, 0, :] # [3]
        dirs = ray[..., 3:6]  # [H, W, 3]
        cos = ray[..., -1:].expand(H, W, P)  # [H, W, P]

        if self.use_crop and isTrain:
            ray_pad = self.pad_w(ray.permute(2, 0, 1).unsqueeze(0))
            gt_pad = self.pad_w(gt.permute(2, 0, 1).unsqueeze(0))
            zbufs_pad = self.pad_b(zbufs.permute(2, 0, 1).unsqueeze(0))
            idbufs_pad = self.pad_b(idbufs.permute(2, 0, 1).unsqueeze(0))

            if mask_gt is not None:
                mask_gt = mask_gt.permute(2, 0, 1).unsqueeze(0)
                cat_img = torch.cat([ray_pad, gt_pad, mask_gt, idbufs_pad, zbufs_pad], dim=1)
            else:
                cat_img = torch.cat([ray_pad, gt_pad, idbufs_pad, zbufs_pad], dim=1)

            # [1, _, train_size, train_size]
            cat_img = self.randomcrop(cat_img)
            _, _, H, W = cat_img.shape
            # o_crop = cat_img[0, :3].permute(1, 2, 0)
            dirs_crop = cat_img[0, 3:6].permute(1, 2, 0)
            cos_crop = cat_img[0, 6:7].permute(1, 2, 0)
            gt_crop = cat_img[0, 7:10].permute(1, 2, 0)
            gt = gt_crop
            
            if mask_gt is not None:
                mask_gt = cat_img[0, 10:11].permute(1, 2, 0)
                idbufs_crop = cat_img[0, 11:11+P].permute(1, 2, 0)
                zbufs_crop = cat_img[0, 11+P:].permute(1, 2, 0)
            else:
                idbufs_crop = cat_img[0, 10:10+P].permute(1, 2, 0)
                zbufs_crop = cat_img[0, 10+P:].permute(1, 2, 0)

            zbufs = zbufs_crop.clone()
            idbufs = idbufs_crop.clone()
            dirs = dirs_crop.clone() # h w 3
            cos = cos_crop.clone().expand(H, W, P) # h w 1 -> h w P

            # [TODO] why this happend
            # if comment two lines, GPU memory increased dramatically more than 20GB
            # if not 6-7GB
            del cat_img, ray_pad, gt_pad, zbufs_pad, idbufs_crop
            torch.cuda.empty_cache()

        feature_map = torch.zeros([H, W, P, self.dim], device=zbufs.device)
        feature_map_bias = torch.zeros([H, W, self.dim], device=zbufs.device)

        if isTrain:
            pixel_mask = zbufs > 0.2 # 800 800 P
        else:
            pixel_mask = zbufs > 0 # 800 800 P

        
        occ_id_buf = idbufs[pixel_mask]
        occ_z_buf = zbufs[pixel_mask]
        occ_cos = cos[pixel_mask]
        dirs_x = dirs[:, :, 0].unsqueeze(-1).expand(cos.shape)[pixel_mask]
        dirs_y = dirs[:, :, 1].unsqueeze(-1).expand(cos.shape)[pixel_mask]
        dirs_z = dirs[:, :, 2].unsqueeze(-1).expand(cos.shape)[pixel_mask]

        # Computate the occupied pixel index by unique 3d point
        u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
        inv_sorted = inverse_indices.argsort(stable=True)
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]

        u_elements_z = occ_z_buf[index].unsqueeze(-1)
        u_cos = occ_cos[index].unsqueeze(-1)
        dx = dirs_x[index]
        dy = dirs_y[index]
        dz = dirs_z[index]
        dir_u = torch.cat([dx.unsqueeze(1), dy.unsqueeze(1), dz.unsqueeze(1)], dim=1)
        xyz_u = o + u_elements_z / u_cos * dir_u
        unique_pixel_mask = torch.isin(idbufs, u_elements)

        #### 1. Generate mlp feature
        feature = self.mlp(xyz_u, dir_u) # [unique.shape[0], dim]
        feature_map[unique_pixel_mask] = feature[inverse_indices, :]

        #### 2. Generate mlp_rect feature
        feat_o = o.repeat(H, W, 1)
        feat_dirs = dirs/cos[..., 0].unsqueeze(-1) # [H, W, 3]
        surface_mask = pixel_mask[..., 0]
        feat_o = feat_o[surface_mask]
        feat_dirs = feat_dirs[surface_mask]

        feat_bias = self.mlp_rect(torch.cat([self.enc_sh(feat_dirs)], dim=-1)) 
        feature_map_bias[surface_mask] = feat_bias

        #### 3. Feature addition
        feature_map += feature_map_bias.unsqueeze(-2)
        feature_map = feature_map.reshape(H, W, P*self.dim).permute(2, 0, 1).unsqueeze(0)
        # 13ms
        # del xyz_u, dir_u
        # torch.cuda.empty_cache()

        #### 4. Fuse K feature maps
        # [1, self.points_per_pixel * self.dim, H, W]
        # TODO: pred_mask to regurlize
        denoise_fm, pred_mask = self.kfn(feature_map)
        
        ret = self.unet(denoise_fm)  # [1, 3, H, W]

        # For dtu dataset, self.mask that distinguishes foreground and background
        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,3,0,1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # [1, 3, H, W]

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
        return {'img': img, 'gt': gt, 'mask_gt': mask_gt, 'pred_mask':pred_mask}

class Renderer_k_v4(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """
    def __init__(self, args):
        super(Renderer_k_v4, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        self.kfn = KFN(args).to(args.device)
        self.unet = UNet(args).to(args.device)

        self.mlp_rect = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, args.dim),
        ).to(args.device)

        self.dim = args.dim
        self.use_crop = args.use_crop

        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)
        else:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)

        self.pad_w = T.Pad(args.pad, 1., 'constant')
        self.pad_b = T.Pad(args.pad, -1., 'constant')

        self.xyznear = args.xyznear  # bool
        self.mask = args.pix_mask
        self.train_size = args.train_size
        self.points_per_pixel = args.points_per_pixel

    def forward(self, zbufs, idbufs, ray, gt, mask_gt, isTrain, xyz_o):
        """
        Args:
            zbuf: z-buffer from rasterization (index buffer when xyznear is True)
            ray: ray direction map
            gt: gt image (used in training to maintain consistent cropping and resizing with input)
            mask_gt: gt mask (used in dtu dataset)
            isTrain: train mode or not
            xyz_o: world coordinates of point clouds (used when xyzenar is True)

        Output:
            img: rendered image
            gt: gt image after cropping and resizing
            mask_gt: gt mask after cropping and resizing 
        """
        H, W, P = zbufs.shape  # [H, W, points_per_pixel]
        o = ray[..., :3]  # [H, W, 3]
        o = o[0, 0, :] # [3]
        dirs = ray[..., 3:6]  # [H, W, 3]
        cos = ray[..., -1:].expand(H, W, P)  # [H, W, P]

        if self.use_crop and isTrain:
            ray_pad = self.pad_w(ray.permute(2, 0, 1).unsqueeze(0))
            gt_pad = self.pad_w(gt.permute(2, 0, 1).unsqueeze(0))
            zbufs_pad = self.pad_b(zbufs.permute(2, 0, 1).unsqueeze(0))
            idbufs_pad = self.pad_b(idbufs.permute(2, 0, 1).unsqueeze(0))

            if mask_gt is not None:
                mask_gt = mask_gt.permute(2, 0, 1).unsqueeze(0)
                cat_img = torch.cat([ray_pad, gt_pad, mask_gt, idbufs_pad, zbufs_pad], dim=1)
            else:
                cat_img = torch.cat([ray_pad, gt_pad, idbufs_pad, zbufs_pad], dim=1)

            # [1, _, train_size, train_size]
            cat_img = self.randomcrop(cat_img)
            _, _, H, W = cat_img.shape
            # o_crop = cat_img[0, :3].permute(1, 2, 0)
            dirs_crop = cat_img[0, 3:6].permute(1, 2, 0)
            cos_crop = cat_img[0, 6:7].permute(1, 2, 0)
            gt_crop = cat_img[0, 7:10].permute(1, 2, 0)
            gt = gt_crop
            
            if mask_gt is not None:
                mask_gt = cat_img[0, 10:11].permute(1, 2, 0)
                idbufs_crop = cat_img[0, 11:11+P].permute(1, 2, 0)
                zbufs_crop = cat_img[0, 11+P:].permute(1, 2, 0)
            else:
                idbufs_crop = cat_img[0, 10:10+P].permute(1, 2, 0)
                zbufs_crop = cat_img[0, 10+P:].permute(1, 2, 0)

            zbufs = zbufs_crop.clone()
            idbufs = idbufs_crop.clone()
            dirs = dirs_crop.clone() # h w 3
            cos = cos_crop.clone().expand(H, W, P) # h w 1 -> h w P

            # [TODO] why this happend
            # if comment two lines, GPU memory increased dramatically more than 20GB
            # if not 6-7GB
            del cat_img, ray_pad, gt_pad, zbufs_pad, idbufs_crop
            torch.cuda.empty_cache()

        feature_map = torch.zeros([H, W, P, self.dim], device=zbufs.device)
        feature_map_bias = torch.zeros([H, W, self.dim], device=zbufs.device)

        if isTrain:
            pixel_mask = zbufs > 0.2 # 800 800 P
        else:
            pixel_mask = zbufs > 0 # 800 800 P

        
        occ_id_buf = idbufs[pixel_mask]
        occ_z_buf = zbufs[pixel_mask]
        occ_cos = cos[pixel_mask]
        dirs_x = dirs[:, :, 0].unsqueeze(-1).expand(cos.shape)[pixel_mask]
        dirs_y = dirs[:, :, 1].unsqueeze(-1).expand(cos.shape)[pixel_mask]
        dirs_z = dirs[:, :, 2].unsqueeze(-1).expand(cos.shape)[pixel_mask]

        # Computate the occupied pixel index by unique 3d point
        u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
        inv_sorted = inverse_indices.argsort(stable=True)
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]

        u_elements_z = occ_z_buf[index].unsqueeze(-1)
        u_cos = occ_cos[index].unsqueeze(-1)
        dx = dirs_x[index]
        dy = dirs_y[index]
        dz = dirs_z[index]
        dir_u = torch.cat([dx.unsqueeze(1), dy.unsqueeze(1), dz.unsqueeze(1)], dim=1)
        xyz_u = o + u_elements_z / u_cos * dir_u
        unique_pixel_mask = torch.isin(idbufs, u_elements)

        #### 1. Generate mlp feature
        feature = self.mlp(xyz_u, dir_u) # [unique.shape[0], dim]
        feature_map[unique_pixel_mask] = feature[inverse_indices, :]

        #### 2. Generate mlp_rect feature
        feat_o = o.repeat(H, W, 1)
        feat_dirs = dirs/cos[..., 0].unsqueeze(-1) # [H, W, 3]
        surface_mask = pixel_mask[..., 0]
        feat_o = feat_o[surface_mask]
        feat_dirs = feat_dirs[surface_mask]

        feat_bias = self.mlp_rect(feat_dirs)
        feature_map_bias[surface_mask] = feat_bias

        #### 3. Feature addition
        feature_map += feature_map_bias.unsqueeze(-2)
        feature_map = feature_map.reshape(H, W, P*self.dim).permute(2, 0, 1).unsqueeze(0)
        # 13ms
        # del xyz_u, dir_u
        # torch.cuda.empty_cache()

        #### 4. Fuse K feature maps
        # [1, self.points_per_pixel * self.dim, H, W]
        # TODO: pred_mask to regurlize
        denoise_fm, pred_mask = self.kfn(feature_map)
        
        ret = self.unet(denoise_fm)  # [1, 3, H, W]

        # For dtu dataset, self.mask that distinguishes foreground and background
        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,3,0,1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # [1, 3, H, W]

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
        return {'img': img, 'gt': gt, 'mask_gt': mask_gt, 'pred_mask':pred_mask}




class Renderer_k_timeit(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """
    def __init__(self, args):
        super(Renderer_k_timeit, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        self.kfn = KFN(args).to(args.device)
        self.unet = UNet(args).to(args.device)

        self.enc_grid = GridEncoder(input_dim=3, 
                    num_levels=16, 
                    level_dim=2, 
                    base_resolution=16, 
                    log2_hashmap_size=19,
                    per_level_scale=1.447,
                    gridtype='hash').to(args.device) # outputdim 32

        self.enc_sh = SHEncoder(degree=4).to(args.device) # outputdim 16

        self.mlp_rect = nn.Sequential(
            nn.Linear(32+16, 64),
            # nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, args.dim),
        ).to(args.device)

        self.dim = args.dim
        self.use_crop = args.use_crop

        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)
        else:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)

        self.pad_w = T.Pad(args.pad, 1., 'constant')
        self.pad_b = T.Pad(args.pad, -1., 'constant')

        self.xyznear = args.xyznear  # bool
        self.mask = args.pix_mask
        self.train_size = args.train_size
        self.points_per_pixel = args.points_per_pixel

    def forward(self, zbufs, idbufs, ray, gt, mask_gt, isTrain, xyz_o):
        """
        Args:
            zbuf: z-buffer from rasterization (index buffer when xyznear is True)
            ray: ray direction map
            gt: gt image (used in training to maintain consistent cropping and resizing with input)
            mask_gt: gt mask (used in dtu dataset)
            isTrain: train mode or not
            xyz_o: world coordinates of point clouds (used when xyznear is True)

        Output:
            img: rendered image
            gt: gt image after cropping and resizing
            mask_gt: gt mask after cropping and resizing 
        """
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        #====================================================================================================#
        begin.record()

        H, W, P = zbufs.shape  # [H, W, points_per_pixel]
        o = ray[..., :3]  # [H, W, 3]
        o = o[0, 0, :] # [3]
        dirs = ray[..., 3:6]  # [H, W, 3]
        cos = ray[..., -1:].expand(H, W, P)  # [H, W, P]

        if self.use_crop and isTrain:
            ray_pad = self.pad_w(ray.permute(2, 0, 1).unsqueeze(0))
            gt_pad = self.pad_w(gt.permute(2, 0, 1).unsqueeze(0))
            zbufs_pad = self.pad_b(zbufs.permute(2, 0, 1).unsqueeze(0))
            idbufs_pad = self.pad_b(idbufs.permute(2, 0, 1).unsqueeze(0))

            if mask_gt is not None:
                mask_gt = mask_gt.permute(2, 0, 1).unsqueeze(0)
                cat_img = torch.cat([ray_pad, gt_pad, mask_gt, idbufs_pad, zbufs_pad], dim=1)
            else:
                cat_img = torch.cat([ray_pad, gt_pad, idbufs_pad, zbufs_pad], dim=1)

            # [1, _, train_size, train_size]
            cat_img = self.randomcrop(cat_img)
            _, _, H, W = cat_img.shape
            # o_crop = cat_img[0, :3].permute(1, 2, 0)
            dirs_crop = cat_img[0, 3:6].permute(1, 2, 0)
            cos_crop = cat_img[0, 6:7].permute(1, 2, 0)
            gt_crop = cat_img[0, 7:10].permute(1, 2, 0)
            gt = gt_crop
            
            if mask_gt is not None:
                mask_gt = cat_img[0, 10:11].permute(1, 2, 0)
                idbufs_crop = cat_img[0, 11:11+P].permute(1, 2, 0)
                zbufs_crop = cat_img[0, 11+P:].permute(1, 2, 0)
            else:
                idbufs_crop = cat_img[0, 10:10+P].permute(1, 2, 0)
                zbufs_crop = cat_img[0, 10+P:].permute(1, 2, 0)

            zbufs = zbufs_crop.clone()
            idbufs = idbufs_crop.clone()
            dirs = dirs_crop.clone() # h w 3
            cos = cos_crop.clone().expand(H, W, P) # h w 1 -> h w P

            # [TODO] why this happend
            # if comment two lines, GPU memory increased dramatically more than 20GB
            # if not 6-7GB
            del cat_img, ray_pad, gt_pad, zbufs_pad, idbufs_crop
            torch.cuda.empty_cache()

        feature_map = torch.zeros([H, W, P, self.dim], device=zbufs.device)
        feature_map_bias = torch.zeros([H, W, self.dim], device=zbufs.device)

        if isTrain:
            pixel_mask = zbufs > 0.2 # 800 800 P
        else:
            pixel_mask = zbufs > 0 # 800 800 P

        
        occ_id_buf = idbufs[pixel_mask]
        occ_z_buf = zbufs[pixel_mask]
        occ_cos = cos[pixel_mask]
        dirs_x = dirs[:, :, 0].unsqueeze(-1).expand(cos.shape)[pixel_mask]
        dirs_y = dirs[:, :, 1].unsqueeze(-1).expand(cos.shape)[pixel_mask]
        dirs_z = dirs[:, :, 2].unsqueeze(-1).expand(cos.shape)[pixel_mask]

        # Computate the occupied pixel index by unique 3d point
        u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
        inv_sorted = inverse_indices.argsort(stable=True)
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]

        u_elements_z = occ_z_buf[index].unsqueeze(-1)
        u_cos = occ_cos[index].unsqueeze(-1)
        dx = dirs_x[index]
        dy = dirs_y[index]
        dz = dirs_z[index]
        dir_u = torch.cat([dx.unsqueeze(1), dy.unsqueeze(1), dz.unsqueeze(1)], dim=1)
        xyz_u = o + u_elements_z / u_cos * dir_u
        unique_pixel_mask = torch.isin(idbufs, u_elements)

        end.record()
        torch.cuda.synchronize()
        elapsed_preprocessing = begin.elapsed_time(end) # ms

        #====================================================================================================#
        begin.record()
        #### 1. Generate mlp feature
        feature = self.mlp(xyz_u, dir_u) # [unique.shape[0], dim]
        feature_map[unique_pixel_mask] = feature[inverse_indices, :]

        end.record()
        torch.cuda.synchronize()
        elapsed_mlp = begin.elapsed_time(end) # ms

        #====================================================================================================#
        begin.record()
        #### 2. Generate mlp_rect feature
        feat_o = o.repeat(H, W, 1)
        feat_dirs = dirs/cos[..., 0].unsqueeze(-1) # [H, W, 3]
        surface_mask = pixel_mask[..., 0]
        feat_o = feat_o[surface_mask]
        feat_dirs = feat_dirs[surface_mask]
        
        feat_bias = self.mlp_rect(torch.cat([self.enc_grid(feat_o), self.enc_sh(feat_dirs)], dim=-1))
        # TODO: better on nerf-synthetic, very worse on dtu maybe.
        # feat_bias = self.mlp_rect(torch.cat([self.enc_sh(feat_o+feat_dirs)], dim=-1)) 
        feature_map_bias[surface_mask] = feat_bias

        end.record()
        torch.cuda.synchronize()
        elapsed_mlp_rect = begin.elapsed_time(end) # ms

        #====================================================================================================#
        begin.record()
        #### 3. Feature addition
        feature_map += feature_map_bias.unsqueeze(-2)
        feature_map = feature_map.reshape(H, W, P*self.dim).permute(2, 0, 1).unsqueeze(0)
        # del xyz_u, dir_u
        # torch.cuda.empty_cache()

        end.record()
        torch.cuda.synchronize()
        elapsed_feat_plus = begin.elapsed_time(end) # ms

        #====================================================================================================#
        begin.record()
        #### 4. Fuse K feature maps
        # [1, self.points_per_pixel * self.dim, H, W]
        # TODO: pred_mask to regurlize
        denoise_fm, pred_mask = self.kfn(feature_map)

        end.record()
        torch.cuda.synchronize()
        elapsed_kfn = begin.elapsed_time(end) # ms

        #====================================================================================================#
        begin.record()
        #### 5. UNet
        ret = self.unet(denoise_fm)  # [1, 3, H, W]
        end.record()
        torch.cuda.synchronize()
        elapsed_unet= begin.elapsed_time(end) # ms


        elapsed_sum = elapsed_preprocessing + elapsed_mlp + elapsed_mlp_rect + elapsed_kfn + elapsed_unet
        print(f'elapsed_sum: {elapsed_sum:.2f}')
        print(f'elapsed_pre: {elapsed_preprocessing:.2f} | %{(elapsed_preprocessing/elapsed_sum * 100):.2f}')
        print(f'elapsed_mlp: {elapsed_mlp:.2f} | %{(elapsed_mlp/elapsed_sum * 100):.2f}')
        print(f'elapsed_mlp_rect: {elapsed_mlp_rect:.2f} | %{(elapsed_mlp_rect/elapsed_sum * 100):.2f}')
        print(f'elapsed_mlp_feat_add: {elapsed_feat_plus:.2f} | %{(elapsed_feat_plus/elapsed_sum * 100):.2f}')
        print(f'elapsed_kfn: {elapsed_kfn:.2f} | %{(elapsed_kfn/elapsed_sum * 100):.2f}')
        print(f'elapsed_unet: {elapsed_unet:.2f} | %{(elapsed_unet/elapsed_sum * 100):.2f}')
        print('====================================================================================')


        # For dtu dataset, self.mask that distinguishes foreground and background
        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,3,0,1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # [1, 3, H, W]

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]
        return {'img': img, 'gt': gt, 'mask_gt': mask_gt}