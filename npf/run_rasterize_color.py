import torch
from rasterizer.rasterizer import rasterize
from utils import config_parser
from dataset.dataset import nerfDataset, ScanDataset, DTUDataset
import os
import numpy as np
import time
from tqdm import tqdm
from utils import config_parser
import open3d as o3d
import matplotlib.pyplot as plt


if __name__ == '__main__':

    parser = config_parser()
    args = parser.parse_args()

    if args.dataset == 'scan':
        train_set = ScanDataset(args, 'train', 'rasterize')
        test_set = ScanDataset(args, 'test', 'rasterize')
    elif args.dataset == 'nerf':
        train_set = nerfDataset(args, 'train', 'rasterize')
        test_set = nerfDataset(args, 'test', 'rasterize')
    elif args.dataset == 'dtu':
        train_set = DTUDataset(args, 'train', 'rasterize')
        test_set = DTUDataset(args, 'test', 'rasterize')
    else:
        assert False

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)

    if not os.path.exists(args.frag_path):
        os.makedirs(args.frag_path)

    test_id_path = str(args.radius) + '-idx-' + str(args.H) + \
        '-' + str(args.points_per_pixel) + '-test.npy'
    test_z_path = str(args.radius) + '-z-' + str(args.H) + '-' + \
        str(args.points_per_pixel) + '-test.npy'
    test_color_path = str(args.radius) + '-color-' + str(args.H) + '-' + \
        str(args.points_per_pixel) + '-test.npy'

    test_id_path = os.path.join(args.frag_path, test_id_path)
    test_z_path = os.path.join(args.frag_path, test_z_path)
    test_color_path = os.path.join(args.frag_path, test_color_path)

    train_id_path = str(args.radius) + '-idx-' + str(args.H) + \
        '-' + str(args.points_per_pixel) + '-train.npy'
    train_z_path = str(args.radius) + '-z-' + str(args.H) + \
        '-' + str(args.points_per_pixel) + '-train.npy'
    train_color_path = str(args.radius) + '-color-' + str(args.H) + \
        '-' + str(args.points_per_pixel) + '-train.npy'

    train_id_path = os.path.join(args.frag_path, train_id_path)
    train_z_path = os.path.join(args.frag_path, train_z_path)
    train_color_path = os.path.join(args.frag_path, train_color_path)


    pcd = o3d.io.read_point_cloud(args.pcdir)
    if args.every_k_points > 1:
        pcd = o3d.geometry.PointCloud.uniform_down_sample(pcd, args.every_k_points)
    pc_colors = torch.tensor(np.asarray(pcd.colors), device=args.device, dtype=torch.float32) # (pc_num, 3)

    pc = train_set.get_pc()

    my_dpi = 96
    view_id = 9
 
    # test set
    begin = time.time()
    color_list = []
    for i, batch in tqdm(enumerate(test_loader), desc='test', unit=' frames'):
    # for i, batch in enumerate(test_loader):
        pose = batch['w2c'][0]
        xyz_ndc = pc.get_ndc(pose)
        idbuf, zbuf = rasterize(xyz_ndc, (args.H, args.W),
                             args.radius, args.points_per_pixel)

        # DEBUG
        # if i != view_id:
            # continue
        # else:
            # print("------------------------done-----------------------")
        img_list = []
        for j in range(args.points_per_pixel):
            color = torch.zeros([1, args.H, args.W, 3], device=zbuf.device)
            pc_id = idbuf[:, :, :, j]
            mask = pc_id >= 0
            lut = pc_id.view(-1)
            lut = lut[lut >= 0].long()
            color[mask] = pc_colors[lut]
            # DEBUG
            # img = color.squeeze().cpu().numpy()
            # h, w, _ = img.shape
            # plt.figure(figsize=(h/my_dpi, w/my_dpi), dpi=my_dpi)
            # plt.imshow(img)
            # plt.axis('off')
            # plt.margins(0,0)
            # plt.savefig(str(j)+".png", bbox_inches="tight", pad_inches=0.0)
            # img_list.append(color)

        color = torch.cat(img_list, dim=-1) # [1, 800, 800, 3 * args.points_per_pixel]
        color_list.append(color.float().cpu())
    
    end = time.time()
    print(f'time cost: {end-begin} s')
    color_list = torch.cat(color_list, dim=0).numpy()

    threshold = 1e-4  # 设置阈值，根据实际情况调整
    color_list[color_list < threshold] = 1.0
 
    np.save(test_color_path, color_list)
    print('z_list.shape', color_list.shape)



    begin = time.time()
    color_list = []
    for i, batch in tqdm(enumerate(train_loader), desc='train', unit=' frames'):
        pose = batch['w2c'][0]
        xyz_ndc = pc.get_ndc(pose)
        id, zbuf = rasterize(xyz_ndc, (args.H, args.W),
                             args.radius, args.points_per_pixel)

        # DEBUG
        # if i != view_id:
            # continue
        # else:
            # print("------------------------done-----------------------")
        img_list = []
        for j in range(args.points_per_pixel):
            color = torch.zeros([1, args.H, args.W, 3], device=zbuf.device)
            pc_id = id[:, :, :, j]
            mask = pc_id >= 0
            lut = pc_id.view(-1)
            lut = lut[lut >= 0].long()
            color[mask] = pc_colors[lut]
            # DEBUG
            # img = color.squeeze().cpu().numpy()
            # h, w, _ = img.shape
            # plt.figure(figsize=(h/my_dpi, w/my_dpi), dpi=my_dpi)
            # plt.imshow(img)
            # plt.axis('off')
            # plt.margins(0,0)
            # plt.savefig(str(j)+".png", bbox_inches="tight", pad_inches=0.0)
            # img_list.append(color)

        color = torch.cat(img_list, dim=-1) # [1, 800, 800, 3 * args.points_per_pixel]
        color_list.append(color.float().cpu())

    end = time.time()
    print(f'time cost: {end-begin} s')
    color_list = torch.cat(color_list, dim=0).numpy()

    threshold = 1e-4  # 设置阈值，根据实际情况调整
    color_list[color_list < threshold] = 1.0

    np.save(train_color_path, color_list)
    print('z_list.shape', color_list.shape)