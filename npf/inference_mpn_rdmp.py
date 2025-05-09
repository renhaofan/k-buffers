import os
import time
import torch
from dataset.dataset import nerfDataset
from utils import config_parser, load_fragments, load_idx, lr_decay, write_video, mse2psnr
from model.renderer import Renderer, Renderer_fast2, Renderer_fast3, Renderer_fast3_profile
import matplotlib.pyplot as plt
from backup_utils import backup_terminal_outputs, backup_code, set_seed
from torch.utils.tensorboard import SummaryWriter

parser = config_parser()
args = parser.parse_args()

if __name__ == '__main__':

    test_set = nerfDataset(args, 'test', 'render')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
    
    renderer = Renderer(args)
    # if args.ckpt is not None:
        # renderer.load_state_dict(torch.load(args.ckpt))
    # edge = args.edge_mask

    _, test_buf = load_fragments(args)  # cpu 100 800 800 k
    _, test_buf_id = load_idx(args)  # cpu 100 800 800 k

    test_buf = test_buf.to(args.device)
    test_buf_id = test_buf_id.to(args.device)

    # print('TEST BEGIN!!!')
    renderer.eval()
    TIMES = 10
    S = 0

    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # for t in range(TIMES):
    #     with torch.autograd.no_grad():
    #         for i, batch in enumerate(test_loader):
    #             idx = int(batch['idx'][0])
    #             ray = batch['ray'][0]
    #             img_gt = batch['rgb'][0]
    #             zbuf = test_buf[idx]
    #             output = renderer(zbuf, ray, gt=None,
    #                               mask_gt=None, isTrain=False, xyz_o=None)
    #             if i == 5:
    #                 begin.record()
    #         end.record()
    #     torch.cuda.synchronize()
    #     elapsed = begin.elapsed_time(end) # ms
    #     fps = 195 / (elapsed / 1000)
    #     S += fps
    #     print(f'BPCR_MPN_RDMP FPS: {fps:.02f}')
    # print(f"{TIMES} times, average FPS: {S / TIMES}")

    # renderer = Renderer_fast2(args)
    # renderer.eval()
    # TIMES = 10
    # S = 0
    # for t in range(TIMES):
    #     with torch.autograd.no_grad():
    #         for i, batch in enumerate(test_loader):
    #             idx = int(batch['idx'][0])
    #             ray = batch['ray'][0]
    #             img_gt = batch['rgb'][0]
    #             zbuf = test_buf[idx]
    #             idbuf = test_buf_id[idx]

    #             output = renderer(zbuf, idbuf, ray, gt=None,
    #                                       mask_gt=None, isTrain=False, xyz_o=None)
    #             if i == 5: # warm-up
    #                 begin.record()
    #         end.record()
    #     torch.cuda.synchronize()
    #     elapsed = begin.elapsed_time(end) # ms
    #     fps = 195 / (elapsed / 1000)
    #     S += fps
    #     print(f'BPCR_MPN_RDMP_FAST_2 FPS: {fps:.02f}')
    # print(f"{TIMES} times, average FPS: {S / TIMES}")
    
    renderer = Renderer_fast3_profile(args)
    renderer.eval()
    TIMES = 10
    S = 0
    for t in range(TIMES):
        with torch.autograd.no_grad():
            for i, batch in enumerate(test_loader):
                idx = int(batch['idx'][0])
                ray = batch['ray'][0]
                img_gt = batch['rgb'][0]
                zbuf = test_buf[idx]
                idbuf = test_buf_id[idx]

                output = renderer(zbuf, idbuf, ray, gt=None,
                                          mask_gt=None, isTrain=False, xyz_o=None)
                if i == 5: # warm-up
                    begin.record()
            end.record()
        torch.cuda.synchronize()
        elapsed = begin.elapsed_time(end) # ms
        fps = 195 / (elapsed / 1000)
        S += fps
        print(f'BPCR_MPN_RDMP_FAST_3 FPS: {fps:.02f}')
    print(f"{TIMES} times, average FPS: {S / TIMES}")
