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

    # pixel_mask = z_buf > 0.2 # 800 800 8
    pixel_mask = z_buf > 0 # 800 800 8
    occ_id_buf = id_buf[pixel_mask]
    
    
    torch.cuda.synchronize()
    start = time.time()
    u_elements, inverse_indices, counts = torch.unique(occ_id_buf, sorted=True, return_inverse=True, return_counts=True)
    torch.cuda.synchronize()
    end = time.time()
    print(f"time: {end - start} s") 

    inv_sorted = inverse_indices.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]

    # u_elements[inverse_indices] == occ_id_buf

    _cos = torch.rand(800, 800, 8).to(args.device)
    _dirs = torch.rand(800, 800, 3).to(args.device)

    unique_pixel_mask = torch.zeros_like(pixel_mask).to(args.device)

    xyz_list = torch.zeros(u_elements.shape[0], 3).to(args.device)
    dir_list = torch.zeros(u_elements.shape[0], 3).to(args.device)

    torch.cuda.synchronize()
    start = time.time()
    
    for i, ue in enumerate(u_elements):
        u_mask = torch.where(id_buf == ue)

        z = z_buf[u_mask][0]
        dir = _dirs[u_mask[:2]].mean(dim=0)
        # [TODO] add origin
        xyz_near = z * dir / _cos[u_mask].mean()
        xyz_list[i] = xyz_near
        dir_list[i]= dir

        unique_pixel_mask[u_mask] = True

    torch.cuda.synchronize()
    end = time.time()
    print(f"time: {end - start} s") 
    # 71s
    

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

