from .net import MLP, UNet, UNet_color, AFNet, UNet_bpcr 
from .mpn import MPN, MPN_tiny
import torch
import torch.nn as nn
from torchvision import transforms as T
# profile
from torchsummary import summary
from utils import config_parser
import time

from shencoder import SHEncoder
from gridencoder import GridEncoder


def unique(x, dim=0):
    unique, inverse, counts = torch.unique(x, dim=dim, 
        sorted=True, return_inverse=True, return_counts=True)
    decimals = torch.arange(inverse.numel(), device=inverse.device) / inverse.numel()
    inv_sorted = (inverse+decimals).argsort()
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    index = index.sort().values
    return unique, inverse, counts, index

class Renderer(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        if args.mpn_tiny:  # better performance, less computation
            self.mpn = MPN_tiny(
                in_dim=args.dim * args.points_per_pixel).to(args.device)
        else:
            self.mpn = MPN(U=2, udim='pp', in_dim=args.dim *
                           args.points_per_pixel).to(args.device)

        self.unet = UNet(args).to(args.device)

        self.dim = args.dim
        self.use_crop = args.use_crop

        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.))
        else:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)

        self.pad_w = T.Pad(args.pad, 1., 'constant')
        self.pad_b = T.Pad(args.pad, -1., 'constant')

        self.xyznear = args.xyznear  # bool
        self.mask = args.pix_mask
        self.train_size = args.train_size
        self.points_per_pixel = args.points_per_pixel

    def forward(self, zbufs, ray, gt, mask_gt, isTrain, xyz_o):
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
            fea_map: the first three dimensions of the feature map of radiance mapping
        """
        # torch.cuda.synchronize()
        # start_ts = time.time()

        H, W, _ = zbufs.shape  # [H, W, points_per_pixel]
        o = ray[..., :3]  # [H, W, 1]
        dirs = ray[..., 3:6]  # [H, W, 3]
        cos = ray[..., -1:]  # [H, W, 1]

        _o = o.unsqueeze(-2).expand(H, W, 1, 3)
        _dirs = dirs.unsqueeze(-2).expand(H, W, 1, 3)
        _cos = cos.unsqueeze(-2).expand(H, W, 1, 3)

        if self.use_crop and isTrain:
            ray_pad = self.pad_w(ray.permute(2, 0, 1).unsqueeze(0))
            gt_pad = self.pad_w(gt.permute(2, 0, 1).unsqueeze(0))
            zbufs_pad = self.pad_b(zbufs.permute(2, 0, 1).unsqueeze(0))

            if mask_gt is not None:
                mask_gt = mask_gt.permute(2, 0, 1).unsqueeze(0)
                cat_img = torch.cat([ray_pad, gt_pad, mask_gt, zbufs_pad], dim=1)
            else:
                cat_img = torch.cat([ray_pad, gt_pad, zbufs_pad], dim=1)

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
                zbufs_crop = cat_img[0, 11:].permute(1, 2, 0)
            else:
                zbufs_crop = cat_img[0, 10:].permute(1, 2, 0)

            zbufs = zbufs_crop.clone()
            _o = ray[:H, :W, :3].unsqueeze(-2)
            _dirs = dirs_crop.clone().unsqueeze(-2).expand(H, W, 1, 3)
            _cos = cos_crop.clone().unsqueeze(-2).expand(H, W, 1, 3)

            # [TODO] why this happend
            # if comment two lines, GPU memory increased dramatically more than 20GB
            # if not 6-7GB
            del cat_img, ray_pad, gt_pad, zbufs_pad
            torch.cuda.empty_cache()

        # torch.cuda.synchronize()
        # end_ts = time.time()
        # print(f"preprocessing time: {end_ts-start_ts}")
        # start_ts = time.time()
 
        radiance_map = []
        for i in range(zbufs.shape[-1]):
            zbuf = zbufs[..., i].unsqueeze(-1)  # [H, W, 1]

            if isTrain:
                pix_mask = zbuf > 0.2  # [H, W, 1]
            else:
                pix_mask = zbuf > 0

            o = _o[pix_mask]  # occ_point 3
            dirs = _dirs[pix_mask]  # occ_point 3
            cos = _cos[pix_mask]  # occ_point 1
            zbuf = zbuf.unsqueeze(-1)[pix_mask]  # occ_point 1

            if self.xyznear:
                xyz_near = o + dirs * zbuf / cos  # occ_point 3
            else:
                xyz_near = xyz_o[zbuf.squeeze(-1).long()]

            feature = self.mlp(xyz_near, dirs)  # occ_point 3

            feature_map = torch.zeros([H, W, 1, self.dim], device=zbuf.device)
            feature_map[pix_mask] = feature  # [400, 400, 1, self.dim]
            radiance_map.append(feature_map.permute(2, 3, 0, 1))

        # [1, self.dim * self.points_per_pixel, H, W]
        rdmp = torch.cat(radiance_map, dim=1)

        # torch.cuda.synchronize()
        # end_ts = time.time()
        # print(f"mlp+feature time: {end_ts-start_ts}")

        ################################### print(f"rdmp {t_cnt}") # 0.2s


        del radiance_map, _o, _dirs, _cos
        torch.cuda.empty_cache()

        # [1, self.dim * self.points_per_pixel, H, W]
        pred_mask = self.mpn(rdmp)

        # confidence = pred_mask.clone().detach()
        # confidence = confidence.mean(dim=1)

        ###################################     try to add regulize term #################################
        # for i in range(self.points_per_pixel):
            # pred_mask[:, i*self.dim:(i+1)*self.dim, :, :] #[1, dim, H, W]

        # [1, self.dim * self.points_per_pixel, H, W]
        fuse_rdmp = rdmp.mul(pred_mask)
        # fuse_rdmp = rdmp.mul(pred_mask).sum(dim=1).unsqueeze(1) # [1, 1, H, W]
        # fuse_rdmp = fuse_rdmp.expand(1, self.dim, H, W)

        ################################## print(f"mpn {t_cnt}") # 0.01s


        # feature_map_view = fuse_rdmp.clone().squeeze(0)[:3, :, :]
        # feature_map_view = torch.sigmoid(feature_map_view.permute(1, 2, 0))
        ret = self.unet(fuse_rdmp)  # [1, 3, H, W]
        ################################## print(f"unet {t_cnt}") # 0.01s

        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,
                                                            3, 0, 1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # 1 3 h w

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

        # return {'img':img, 'gt':gt, 'mask_gt':mask_gt, 'fea_map':feature_map_view}
        return {'img': img, 'gt': gt, 'mask_gt': mask_gt}

class Renderer_fast(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer_fast, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        if args.mpn_tiny:  # better performance, less computation
            self.mpn = MPN_tiny(
                in_dim=args.dim * args.points_per_pixel).to(args.device)
        else:
            self.mpn = MPN(U=2, udim='pp', in_dim=args.dim *
                           args.points_per_pixel).to(args.device)

        self.unet = UNet(args).to(args.device)

        self.dim = args.dim
        self.use_crop = args.use_crop

        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(
                args.scale_min, args.scale_max), ratio=(1., 1.))
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
            fea_map: the first three dimensions of the feature map of radiance mapping
        """

        H, W, P = zbufs.shape  # [H, W, points_per_pixel]
        o = ray[..., :3]  # [H, W, 1]
        o = o[0, 0, :] # [3]
        dirs = ray[..., 3:6]  # [H, W, 3]
        cos = ray[..., -1:].expand(H, W, P)  # [H, W, 1]

        # crop BUG
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

        if isTrain:
            pixel_mask = zbufs > 0.2 # 800 800 P
        else:
            pixel_mask = zbufs > 0 # 800 800 P
        occ_id_buf = idbufs[pixel_mask]
        u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
        unique_pixel_mask = torch.zeros_like(pixel_mask).to(zbufs.device)

        xyz_u = torch.zeros(u_elements.shape[0], 3).to(zbufs.device)
        dir_u = torch.zeros(u_elements.shape[0], 3).to(zbufs.device)

        print('--------------------------------')
        for i, ue in enumerate(u_elements):
            u_mask = torch.where(idbufs == ue)

            z = zbufs[u_mask][0]
            dir = dirs[u_mask[:2]].mean(dim=0)
            if self.xyznear:
                xyz_near = o + z * dir / cos[u_mask].mean()
            else:
                # xyz_near = xyz_o[zbuf.squeeze(-1).long()]
                # [TODO]
                pass
            xyz_u[i] = xyz_near
            dir_u[i]= dir

            unique_pixel_mask[u_mask] = True

        feature = self.mlp(xyz_u, dir_u) # [unique.shape[0], dim]
        feature_map = torch.zeros([H, W, P, self.dim]).to(zbufs.device)
        feature_map[unique_pixel_mask] = feature[inverse_indices, :]
        feature_map = feature_map.reshape(H, W, P*self.dim).permute(2, 0, 1).unsqueeze(0)

        del xyz_u, dir_u 
        torch.cuda.empty_cache()

        # [1, self.dim * self.points_per_pixel, H, W]
        pred_mask = self.mpn(feature_map)
        fuse_rdmp = feature_map.mul(pred_mask)

        ret = self.unet(fuse_rdmp)  # [1, 3, H, W]

        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,
                                                            3, 0, 1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # 1 3 h w

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

        return {'img': img, 'gt': gt, 'mask_gt': mask_gt}

class Renderer_fast2(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer_fast2, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        if args.mpn_tiny:  # better performance, less computation
            self.mpn = MPN_tiny(
                in_dim=args.dim * args.points_per_pixel).to(args.device)
        else:
            self.mpn = MPN(U=2, udim='pp', in_dim=args.dim *
                           args.points_per_pixel).to(args.device)

        self.unet = UNet(args).to(args.device)

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
            fea_map: the first three dimensions of the feature map of radiance mapping
        """

        # torch.cuda.synchronize()
        # start_ts = time.time()

        H, W, P = zbufs.shape  # [H, W, points_per_pixel]
        o = ray[..., :3]  # [H, W, 1]
        o = o[0, 0, :] # [3]
        dirs = ray[..., 3:6]  # [H, W, 3]
        cos = ray[..., -1:].expand(H, W, P)  # [H, W, P]

        # crop BUG
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

        u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
        inv_sorted = inverse_indices.argsort(stable=True)
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]
        # [BUG] perform bad
        # u_elements, inverse_indices, counts, index = unique(occ_id_buf)

        u_elements_z = occ_z_buf[index].unsqueeze(-1)
        u_cos = occ_cos[index].unsqueeze(-1)
        dx = dirs_x[index]
        dy = dirs_y[index]
        dz = dirs_z[index]
        dir_u = torch.cat([dx.unsqueeze(1), dy.unsqueeze(1), dz.unsqueeze(1)], dim=1)
        xyz_u = o + u_elements_z / u_cos * dir_u
        unique_pixel_mask = torch.isin(idbufs, u_elements)

        # print(f"{xyz_u.shape[0]}")
        # print(f"{inverse_indices.shape[0]}")
        # torch.cuda.synchronize()
        # end_ts = time.time()
        # print(f"xyz.shape: {xyz_u.shape[0]}")
        # print(f"preprocessing time: {end_ts-start_ts}")
        
        # start_ts = time.time()
        feature = self.mlp(xyz_u, dir_u) # [unique.shape[0], dim]
        # torch.cuda.synchronize()
        # end_ts = time.time()
        # print(f"mlp time: {end_ts-start_ts}")

        # start_ts = time.time()
        # feature_map = torch.zeros([H, W, P, self.dim]).to(zbufs.device) # 0.1s
        feature_map = torch.zeros([H, W, P, self.dim], device=zbufs.device) #0.005s
        feature_map[unique_pixel_mask] = feature[inverse_indices, :]
        feature_map = feature_map.reshape(H, W, P*self.dim).permute(2, 0, 1).unsqueeze(0)
        # torch.cuda.synchronize()
        # end_ts = time.time()
        # print(f"feature time: {end_ts-start_ts}")


        # del xyz_u, dir_u
        # torch.cuda.empty_cache()

        # [1, self.dim * self.points_per_pixel, H, W]
        # start_ts = time.time()
        pred_mask = self.mpn(feature_map)
        fuse_rdmp = feature_map.mul(pred_mask)
        # torch.cuda.synchronize()
        # end_ts = time.time()
        # print(f"mpn time: {end_ts-start_ts}")

        # start_ts = time.time()
        ret = self.unet(fuse_rdmp)  # [1, 3, H, W]
        # torch.cuda.synchronize()
        # end_ts = time.time()
        # print(f"unet time: {end_ts-start_ts}")

        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,
                                                            3, 0, 1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # 1 3 h w

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

        return {'img': img, 'gt': gt, 'mask_gt': mask_gt}

class Renderer_fast3(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer_fast3, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        if args.mpn_tiny:  # better performance, less computation
            self.mpn = MPN_tiny(
                in_dim=args.dim * args.points_per_pixel).to(args.device)
        else:
            self.mpn = MPN(U=2, udim='pp', in_dim=args.dim *
                           args.points_per_pixel).to(args.device)

        self.unet = UNet(args).to(args.device)

        # self.recolor = tcnn.Encoding(
        #           n_input_dims=3,
        #           encoding_config={
        #              "otype": "HashGrid",
        #              "n_levels": 16,
        #              "n_features_per_level": 2,
        #              "log2_hashmap_size": 19,
        #              "base_resolution": 16,
        #              "per_level_scale": 1.447,
        #          },
        # )
        # self.direction_encoding = tcnn.Encoding(
        #      n_input_dims=3,
        #      encoding_config={
        #          "otype": "SphericalHarmonics",
        #          "degree": 3 
        #      },
        # )
        # self.recolor = tcnn.Encoding(
        #         n_input_dims=3,
        #         encoding_config={
        #             "otype": "Frequency",
        #             "degree": 10
        #     },
        # )
        # self.mlp_head = tcnn.Network(
        #         n_input_dims=(self.direction_encoding.n_output_dims+self.recolor.n_output_dims),
        #         n_output_dims=args.dim,
        #         network_config={
        #             "otype": "FullyFusedMLP",
        #             "activation": "ReLU",
        #             "output_activation": "None",
        #             "n_neurons": 64,
        #             "n_hidden_layers": 2,
        #         },
        # )

        self.enc_grid = GridEncoder(input_dim=3, 
                    num_levels=16, 
                    level_dim=2, 
                    base_resolution=16, 
                    log2_hashmap_size=19,
                    per_level_scale=1.447,
                    gridtype='hash').to(args.device) # outputdim 32

        self.enc_sh = SHEncoder(degree=4).to(args.device) # outputdim 16

        self.mlp_head = nn.Sequential(
            #nn.Linear(32+16, 64),
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
            fea_map: the first three dimensions of the feature map of radiance mapping
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

        u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
        inv_sorted = inverse_indices.argsort(stable=True)
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]
        #[BUG] perform bad
        #u_elements, inverse_indices, counts, index = unique(occ_id_buf)

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

        #### 2. Generate mlp_bias feature
        feat_o = o.repeat(H, W, 1)
        feat_dirs = dirs/cos[..., 0].unsqueeze(-1) # [H, W, 3]
        surface_mask = pixel_mask[..., 0]
        feat_o = feat_o[surface_mask]
        feat_dirs = feat_dirs[surface_mask]
        
        # RAW TCNN
        #feat_bias = self.mlp_head(torch.cat([self.recolor(feat_o), self.direction_encoding(feat_dirs)], dim=-1))
        
        # pytorch define
        #print(feat_o.shape)
        #print(feat_dirs.shape)
        #feat_bias = self.mlp_head(torch.cat([self.enc_grid(feat_o), self.enc_sh(feat_dirs)], dim=-1))
        feat_bias = self.mlp_head(torch.cat([self.enc_sh(feat_o+feat_dirs)], dim=-1))
        feature_map_bias[surface_mask] = feat_bias.float()

        #### 3. Concat feature
        feature_map += feature_map_bias.unsqueeze(-2)
        feature_map = feature_map.reshape(H, W, P*self.dim).permute(2, 0, 1).unsqueeze(0)

        # del xyz_u, dir_u
        # torch.cuda.empty_cache()

        # [1, self.dim * self.points_per_pixel, H, W]
        pred_mask = self.mpn(feature_map)
        fuse_rdmp = feature_map.mul(pred_mask)

        
        ret = self.unet(fuse_rdmp)  # [1, 3, H, W]

        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,
                                                            3, 0, 1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # 1 3 h w

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

        return {'img': img, 'gt': gt, 'mask_gt': mask_gt}


class Renderer_fast3_profile(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer_fast3_profile, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        if args.mpn_tiny:  # better performance, less computation
            self.mpn = MPN_tiny(
                in_dim=args.dim * args.points_per_pixel).to(args.device)
        else:
            self.mpn = MPN(U=2, udim='pp', in_dim=args.dim *
                           args.points_per_pixel).to(args.device)

        # self.unet = UNet(args).to(args.device)
        self.unet = UNet_bpcr(args).to(args.device)

        # self.unet = nn.Sequential(
        #     nn.Conv2d(in_channels=args.dim * args.points_per_pixel, out_channels=32, kernel_size=1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        # ).to(args.device)

        # self.unet = nn.Sequential(
        #     nn.Conv2d(in_channels=args.dim * args.points_per_pixel, out_channels=32, kernel_size=3, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        # ).to(args.device)

        # self.recolor = tcnn.Encoding(
        #           n_input_dims=3,
        #           encoding_config={
        #              "otype": "HashGrid",
        #              "n_levels": 16,
        #              "n_features_per_level": 2,
        #              "log2_hashmap_size": 19,
        #              "base_resolution": 16,
        #              "per_level_scale": 1.447,
        #          },
        # )
        # self.direction_encoding = tcnn.Encoding(
        #      n_input_dims=3,
        #      encoding_config={
        #          "otype": "SphericalHarmonics",
        #          "degree": 3 
        #      },
        # )
        # self.recolor = tcnn.Encoding(
        #         n_input_dims=3,
        #         encoding_config={
        #             "otype": "Frequency",
        #             "degree": 10
        #     },
        # )
        # self.mlp_head = tcnn.Network(
        #         n_input_dims=(self.direction_encoding.n_output_dims+self.recolor.n_output_dims),
        #         n_output_dims=args.dim,
        #         network_config={
        #             "otype": "FullyFusedMLP",
        #             "activation": "ReLU",
        #             "output_activation": "None",
        #             "n_neurons": 64,
        #             "n_hidden_layers": 2,
        #         },
        # )

        self.enc_grid = GridEncoder(input_dim=3, 
                    num_levels=16, 
                    level_dim=2, 
                    base_resolution=16, 
                    log2_hashmap_size=19,
                    per_level_scale=1.447,
                    gridtype='hash').to(args.device) # outputdim 32

        self.enc_sh = SHEncoder(degree=4).to(args.device) # outputdim 16

        self.mlp_head = nn.Sequential(
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
            fea_map: the first three dimensions of the feature map of radiance mapping
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

        u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
        inv_sorted = inverse_indices.argsort(stable=True)
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]
        #[BUG] perform bad
        #u_elements, inverse_indices, counts, index = unique(occ_id_buf)

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
        #### 2. Generate mlp_bias feature
        feat_o = o.repeat(H, W, 1)
        feat_dirs = dirs/cos[..., 0].unsqueeze(-1) # [H, W, 3]
        surface_mask = pixel_mask[..., 0]
        feat_o = feat_o[surface_mask]
        feat_dirs = feat_dirs[surface_mask]

        
        # RAW TCNN
        #feat_bias = self.mlp_head(torch.cat([self.recolor(feat_o), self.direction_encoding(feat_dirs)], dim=-1))
        
        # pytorch define
        feat_bias = self.mlp_head(torch.cat([self.enc_grid(feat_o), self.enc_sh(feat_dirs)], dim=-1))
        #feat_bias = self.mlp_head(torch.cat([self.enc_sh(feat_o+feat_dirs)], dim=-1))
        feature_map_bias[surface_mask] = feat_bias.float()

        end.record()
        torch.cuda.synchronize()
        elapsed_mlp_head = begin.elapsed_time(end) # ms
        
        
        #====================================================================================================#
        begin.record()
        #### 3. Concat feature
        feature_map += feature_map_bias.unsqueeze(-2)
        feature_map = feature_map.reshape(H, W, P*self.dim).permute(2, 0, 1).unsqueeze(0)
        
        end.record()
        torch.cuda.synchronize()
        elapsed_feat_plus = begin.elapsed_time(end) # ms

        # del xyz_u, dir_u
        # torch.cuda.empty_cache()

        #====================================================================================================#
        begin.record()
        # [1, self.dim * self.points_per_pixel, H, W]
        pred_mask = self.mpn(feature_map)
        fuse_rdmp = feature_map.mul(pred_mask)
        end.record()
        torch.cuda.synchronize()
        elapsed_mpn = begin.elapsed_time(end) # ms

        #====================================================================================================#
        begin.record()
        ret = self.unet(fuse_rdmp)  # [1, 3, H, W]
        end.record()
        torch.cuda.synchronize()
        elapsed_unet = begin.elapsed_time(end) # ms

        elapsed_sum = elapsed_preprocessing + elapsed_mlp + elapsed_mlp_head + elapsed_mpn + elapsed_unet
        print(f'elapsed_sum: {elapsed_sum:.2f}')
        print(f'elapsed_pre: {elapsed_preprocessing:.2f} | %{(elapsed_preprocessing/elapsed_sum * 100):.2f}')
        print(f'elapsed_mlp: {elapsed_mlp:.2f} | %{(elapsed_mlp/elapsed_sum * 100):.2f}')
        print(f'elapsed_mlp_head: {elapsed_mlp_head:.2f} | %{(elapsed_mlp_head/elapsed_sum * 100):.2f}')
        print(f'elapsed_mlp_faet_add: {elapsed_feat_plus:.2f} | %{(elapsed_feat_plus/elapsed_sum * 100):.2f}')
        print(f'elapsed_mpn: {elapsed_mpn:.2f} | %{(elapsed_mpn/elapsed_sum * 100):.2f}')
        print(f'elapsed_unet: {elapsed_unet:.2f} | %{(elapsed_unet/elapsed_sum * 100):.2f}')
        print('====================================================================================')

        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,
                                                            3, 0, 1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # 1 3 h w

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

        return {'img': img, 'gt': gt, 'mask_gt': mask_gt}

# Compared with fast3, more than 0.5db in PSNR
class Renderer_sh(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer_sh, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        if args.mpn_tiny:  # better performance, less computation
            self.mpn = MPN_tiny(
                in_dim=args.dim * args.points_per_pixel).to(args.device)
        else:
            self.mpn = MPN(U=2, udim='pp', in_dim=args.dim *
                           args.points_per_pixel).to(args.device)

        self.unet = UNet(args).to(args.device)

        # self.recolor = tcnn.Encoding(
        #           n_input_dims=3,
        #           encoding_config={
        #              "otype": "HashGrid",
        #              "n_levels": 16,
        #              "n_features_per_level": 2,
        #              "log2_hashmap_size": 19,
        #              "base_resolution": 16,
        #              "per_level_scale": 1.447,
        #          },
        # )
        # self.direction_encoding = tcnn.Encoding(
        #      n_input_dims=3,
        #      encoding_config={
        #          "otype": "SphericalHarmonics",
        #          "degree": 3 
        #      },
        # )
        # self.recolor = tcnn.Encoding(
        #         n_input_dims=3,
        #         encoding_config={
        #             "otype": "Frequency",
        #             "degree": 10
        #     },
        # )
        # self.mlp_head = tcnn.Network(
        #         n_input_dims=(self.direction_encoding.n_output_dims+self.recolor.n_output_dims),
        #         n_output_dims=args.dim,
        #         network_config={
        #             "otype": "FullyFusedMLP",
        #             "activation": "ReLU",
        #             "output_activation": "None",
        #             "n_neurons": 64,
        #             "n_hidden_layers": 2,
        #         },
        # )

        self.enc_grid = GridEncoder(input_dim=3, 
                    num_levels=16, 
                    level_dim=2, 
                    base_resolution=16, 
                    log2_hashmap_size=19,
                    per_level_scale=1.447,
                    gridtype='hash').to(args.device) # outputdim 32

        self.enc_sh = SHEncoder(degree=4).to(args.device) # outputdim 16

        self.mlp_head = nn.Sequential(
            # nn.Linear(32+16, 64),
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
            fea_map: the first three dimensions of the feature map of radiance mapping
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

        u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
        inv_sorted = inverse_indices.argsort(stable=True)
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]
        #[BUG] perform bad
        #u_elements, inverse_indices, counts, index = unique(occ_id_buf)

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

        #### 2. Generate mlp_bias feature
        feat_o = o.repeat(H, W, 1)
        feat_dirs = dirs/cos[..., 0].unsqueeze(-1) # [H, W, 3]
        surface_mask = pixel_mask[..., 0]
        feat_o = feat_o[surface_mask]
        feat_dirs = feat_dirs[surface_mask]
        
        # RAW TCNN
        #feat_bias = self.mlp_head(torch.cat([self.recolor(feat_o), self.direction_encoding(feat_dirs)], dim=-1))
        
        # pytorch define
        # feat_bias = self.mlp_head(torch.cat([self.enc_grid(feat_o), self.enc_sh(feat_dirs)], dim=-1))
        feat_bias = self.mlp_head(self.enc_sh(feat_o+feat_dirs))
        feature_map_bias[surface_mask] = feat_bias

        #### 3. Concat feature
        feature_map += feature_map_bias.unsqueeze(-2)
        feature_map = feature_map.reshape(H, W, P*self.dim).permute(2, 0, 1).unsqueeze(0)

        # del xyz_u, dir_u
        # torch.cuda.empty_cache()

        # [1, self.dim * self.points_per_pixel, H, W]
        pred_mask = self.mpn(feature_map)
        fuse_rdmp = feature_map.mul(pred_mask)

        
        ret = self.unet(fuse_rdmp)  # [1, 3, H, W]

        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,
                                                            3, 0, 1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # 1 3 h w

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

        return {'img': img, 'gt': gt, 'mask_gt': mask_gt}

class Renderer_sh_1x1(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer_sh_1x1, self).__init__()
        
        if args.af_mlp:
            self.mlp = AFNet(args.dim).to(args.device)
        else:
            self.mlp = MLP(args.dim, args.use_fourier).to(args.device)
            
        if args.mpn_tiny:  # better performance, less computation
            self.mpn = MPN_tiny(
                in_dim=args.dim * args.points_per_pixel).to(args.device)
        else:
            self.mpn = MPN(U=2, udim='pp', in_dim=args.dim *
                           args.points_per_pixel).to(args.device)

        # 10ms
        self.unet = UNet_bpcr(args).to(args.device)

        # 17ms
        # self.unet = UNet(args).to(args.device)

        # 2.88ms
        # self.unet = nn.Conv2d(in_channels=args.dim * args.points_per_pixel, out_channels=3, kernel_size=1).to(args.device)

        # 3.88ms
        # self.unet = nn.Sequential(
        #     nn.Conv2d(in_channels=args.dim * args.points_per_pixel, out_channels=32, kernel_size=1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        # ).to(args.device)

        # 4.78ms
        self.unet = nn.Sequential(
            nn.Conv2d(in_channels=args.dim * args.points_per_pixel, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1)
        ).to(args.device)
 
        self.enc_grid = GridEncoder(input_dim=3, 
                    num_levels=16, 
                    level_dim=2, 
                    base_resolution=16, 
                    log2_hashmap_size=19,
                    per_level_scale=1.447,
                    gridtype='hash').to(args.device) # outputdim 32

        self.enc_sh = SHEncoder(degree=4).to(args.device) # outputdim 16

        self.mlp_head = nn.Sequential(
            # nn.Linear(32+16, 64),
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
            fea_map: the first three dimensions of the feature map of radiance mapping
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

        u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
        inv_sorted = inverse_indices.argsort(stable=True)
        tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
        index = inv_sorted[tot_counts]
        #[BUG] perform bad
        #u_elements, inverse_indices, counts, index = unique(occ_id_buf)

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

        #### 2. Generate mlp_bias feature
        feat_o = o.repeat(H, W, 1)
        feat_dirs = dirs/cos[..., 0].unsqueeze(-1) # [H, W, 3]
        surface_mask = pixel_mask[..., 0]
        feat_o = feat_o[surface_mask]
        feat_dirs = feat_dirs[surface_mask]
        
        # RAW TCNN
        #feat_bias = self.mlp_head(torch.cat([self.recolor(feat_o), self.direction_encoding(feat_dirs)], dim=-1))
        
        # pytorch define
        # feat_bias = self.mlp_head(torch.cat([self.enc_grid(feat_o), self.enc_sh(feat_dirs)], dim=-1))
        feat_bias = self.mlp_head(self.enc_sh(feat_o+feat_dirs))
        feature_map_bias[surface_mask] = feat_bias

        #### 3. Concat feature
        feature_map += feature_map_bias.unsqueeze(-2)
        feature_map = feature_map.reshape(H, W, P*self.dim).permute(2, 0, 1).unsqueeze(0)

        # del xyz_u, dir_u
        # torch.cuda.empty_cache()

        # [1, self.dim * self.points_per_pixel, H, W]
        pred_mask = self.mpn(feature_map)
        fuse_rdmp = feature_map.mul(pred_mask)

        ret = self.unet(fuse_rdmp)  # [1, 3, H, W]

        if self.mask and (not isTrain):
            pix_mask = pix_mask.int().unsqueeze(-1).permute(2,
                                                            3, 0, 1)  # [1, 1, H, W]
            ret = ret * pix_mask + (1 - pix_mask)  # 1 3 h w

        img = ret.squeeze(0).permute(1, 2, 0)  # [H, W, 3]

        return {'img': img, 'gt': gt, 'mask_gt': mask_gt}


if __name__ == '__main__':
    device = torch.device("cuda")
    parser = config_parser()
    args = parser.parse_args()
