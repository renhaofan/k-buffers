import torch
from rasterizer.rasterizer import rasterize
from utils import config_parser
from dataset.dataset import nerfDataset, ScanDataset, DTUDataset
import os
import numpy as np
import time
from tqdm import tqdm
from utils import config_parser


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
    test_id_path = os.path.join(args.frag_path, test_id_path)
    test_z_path = os.path.join(args.frag_path, test_z_path)

    train_id_path = str(args.radius) + '-idx-' + str(args.H) + \
        '-' + str(args.points_per_pixel) + '-train.npy'
    train_z_path = str(args.radius) + '-z-' + str(args.H) + \
        '-' + str(args.points_per_pixel) + '-train.npy'
    train_id_path = os.path.join(args.frag_path, train_id_path)
    train_z_path = os.path.join(args.frag_path, train_z_path)


    pc = train_set.get_pc()

    # test set
    begin = time.time()
    z_list = []
    id_list = []
    for i, batch in tqdm(enumerate(test_loader), desc='test', unit=' frames'):
    # for i, batch in enumerate(test_loader):
        pose = batch['w2c'][0]
        xyz_ndc = pc.get_ndc(pose)
        id, zbuf = rasterize(xyz_ndc, (args.H, args.W),
                             args.radius, args.points_per_pixel)
        z_list.append(zbuf.float().cpu())
        id_list.append(id.long().cpu())
        # if i % 20 == 0:
            # print('test', i)
    end = time.time()
    print(f'time cost: {end-begin} s')
    z_list = torch.cat(z_list, dim=0).numpy()
    id_list = torch.cat(id_list, dim=0).numpy()
    print('z_list.shape', z_list.shape)
    np.save(test_z_path, z_list)
    np.save(test_id_path, id_list)

    # train set
    begin = time.time()
    z_list = []
    id_list = []
    for i, batch in tqdm(enumerate(train_loader), desc='train ', unit=' frames'):
    # for i, batch in enumerate(train_loader):
        pose = batch['w2c'][0]
        xyz_ndc = pc.get_ndc(pose)
        id, zbuf = rasterize(xyz_ndc, (args.H, args.W),
                             args.radius, args.points_per_pixel)
        z_list.append(zbuf.float().cpu())
        id_list.append(id.long().cpu())
        # if i % 20 == 0:
            # print('train', i)
    end = time.time()
    print(f'time cost: {end-begin} s')
    z_list = torch.cat(z_list, dim=0).numpy()
    id_list = torch.cat(id_list, dim=0).numpy()
    print('z_list.shape', z_list.shape)
    np.save(train_z_path, z_list)
    np.save(train_id_path, id_list)

    