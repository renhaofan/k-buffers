import torch
from rasterizer.rasterizer import rasterize
from utils import config_parser
from dataset.dataset import nerfDataset, ScanDataset, DTUDataset
import os
import numpy as np
import time
from tqdm import tqdm
from utils import config_parser, load_fragments, load_idx


if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()

    # train_z_buf, _ = load_fragments(args)  # cpu 100 800 800 8
    # train_id_buf, _ = load_idx(args)  # cpu 100 800 800 8
    # 
    # c_id = 15
    # z_buf = train_z_buf[c_id].to(args.device) # 800 800 8
    # id_buf = train_id_buf[c_id].to(args.device) # 800 800 8
    
    z_buf = torch.load("z_buf.pt")
    id_buf = torch.load("id_buf.pt")
    
    torch.cuda.synchronize()
    start = time.time()

    _cos = torch.rand(800, 800, 8).to(args.device)
    _dirs = torch.rand(800, 800, 3).to(args.device)
    dirs_x = _dirs[:, :, 0].unsqueeze(-1).expand(_cos.shape)
    dirs_y = _dirs[:, :, 1].unsqueeze(-1).expand(_cos.shape)
    dirs_z = _dirs[:, :, 2].unsqueeze(-1).expand(_cos.shape)


    # pixel_mask = z_buf > 0.2 # 800 800 8
    pixel_mask = z_buf > 0 # 800 800 8
    occ_id_buf = id_buf[pixel_mask]
    occ_z_buf = z_buf[pixel_mask]
    
    occ_cos = _cos[pixel_mask]
    occ_x = dirs_x[pixel_mask]
    occ_y = dirs_y[pixel_mask]
    occ_z = dirs_z[pixel_mask]
    

    u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
    # u_elements[inverse_indices] == occ_id_buf

    # https://github.com/pytorch/pytorch/issues/36748
    # get the first occurring unique slices.
    inv_sorted = inverse_indices.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    u_elements_z = occ_z_buf[index]
    u_cos = occ_cos[index]
    dx = occ_x[index]
    dy = occ_y[index]
    dz = occ_z[index]
    dirs = torch.cat([dx.unsqueeze(1), dy.unsqueeze(1), dz.unsqueeze(1)], dim=1)
    z = u_elements_z.unsqueeze(-1)
    cos = u_cos.unsqueeze(-1)
    xyz_near = z / cos * dirs
 
    torch.cuda.synchronize()
    end = time.time()
    print(f"time: {end - start} s")
   
    unique_pixel_mask = torch.zeros_like(pixel_mask).to(args.device)

    xyz_list = torch.zeros(u_elements.shape[0], 3).to(args.device)
    dir_list = torch.zeros(u_elements.shape[0], 3).to(args.device)

    torch.cuda.synchronize()
    start = time.time()
    
    slice_size = 50_000  # balance speed and memory
 
    for i in range(0, len(u_elements), slice_size):
        start_index = i
        end_index = i + slice_size
        current_slice = u_elements[start_index:end_index]
        current_counts = counts[start_index:end_index]
        z = u_elements_z[start_index:end_index]

        u_mask = torch.isin(id_buf, current_slice)
        unique_pixel_mask = unique_pixel_mask | u_mask
        indices = torch.nonzero(u_mask)

        dirs = _dirs[indices[:, 0], indices[:, 1], :]
        dir_list[start_index:end_index] = dirs
        # a = torch.split(indices, current_counts.tolist())

        # [TODO]
        # indices.shape torch.Size([418706, 3])
        # z_buf[u_mask].shape torch.Size([418706])
        
        # find a permulation matrix  between z_buf[u_mask] ==> A and z ==> B
        # apply this permulation matrix to adjust indices
        # z_repeat = z.repeat_interleave(current_counts)
        # z_found = z_buf[u_mask]
        # cost_matrix = torch.abs(z_found.view(z_found.shape[0], 1) - z_repeat).cpu().numpy()
        
        

    torch.cuda.synchronize()
    end = time.time()
    print(f"time: {end - start} s")

    # for i, ue in enumerate(u_elements):
    #     u_mask = torch.isin(id_buf, ue)
    #     unique_pixel_mask = unique_pixel_mask | u_mask

    #     z = z_buf[u_mask][0]
    #     cos = _cos[u_mask].mean()

    #     indices = torch.nonzero(u_mask)[:, :2]
    #     dir = _dirs[indices[:, 0], indices[:, 1], :].mean(dim=0)
    #     xyz_near = z * dir / cos
    #     xyz_list[i] = xyz_near
    #     dir_list[i]= dir

    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"time: {end - start} s") # 120s

    # feature = self.mlp
    feature = torch.randn([u_elements.shape[0], 8])[args.device]
    feature_map = torch.zeros([800, 800, 8, 8]).to(args.device)
    # [TODO]    
    feature_map[unique_pixel_mask] = feature[inverse_indices, :]
   
    # feature_map = feature_map.reshape(H, W, f) 
    # feature_map = torch.zeros([1, 8*8, 800, 800]).to(args.device)


    print('-------')
    print('-------')
    # print('-------')

