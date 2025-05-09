import os
import time
import torch
from dataset.dataset import nerfDataset
from utils import config_parser, load_fragments, load_idx, lr_decay, write_video, mse2psnr
import matplotlib.pyplot as plt
from backup_utils import backup_terminal_outputs, backup_code, set_seed
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from model.renderer import Renderer, Renderer_fast2, Renderer_fast3, Renderer_ex

parser = config_parser()
args = parser.parse_args()

# set_seed(1023)
# back_path = os.path.join('logs', time.strftime("%y%m%d-%H%M%S-" + f'{args.expname}'))
# os.makedirs(back_path)
# backup_terminal_outputs(back_path)
# backup_code(back_path, ignored_in_current_folder=['logs_pc_opt','logs_edit','ckpt','data','.git','pytorch_rasterizer.egg-info','build','logs','__pycache__'])
# print(back_path)
# logger = SummaryWriter(back_path)
# video_path = os.path.join(back_path, 'video')
# os.makedirs(video_path)


if __name__ == '__main__':

    test_set = nerfDataset(args, 'test', 'render')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
    

    # renderer = Renderer(args)
    # renderer = Renderer_fast2(args)
    # renderer = Renderer_fast3(args)
    # renderer.eval()
    # ray = torch.randn([800, 800, 7], device=args.device)
    # zbuf = torch.randn([800, 800, args.points_per_pixel], device=args.device)
    # with profile(activities=[
    #     ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    #         output = renderer(zbuf, ray, gt=None, mask_gt=None, isTrain=False, xyz_o=None)

    # print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
    
    
    _, test_buf = load_fragments(args)  # cpu 100 800 800 k
    _, test_buf_id = load_idx(args)  # cpu 100 800 800 k

    test_buf = test_buf.to(args.device)
    test_buf_id = test_buf_id.to(args.device)
    
    # renderer = Renderer(args)
    # renderer.eval()
    # for i, batch in enumerate(test_loader):
    #     idx = int(batch['idx'][0])
    #     ray = batch['ray'][0]
    #     img_gt = batch['rgb'][0]
    #     zbuf = test_buf[idx]
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #         with record_function("model_inference"):
    #             output = renderer(zbuf, ray, gt=None,
    #                       mask_gt=None, isTrain=False, xyz_o=None)
    #     print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
    #     break

    # renderer = Renderer_fast2(args)
    # renderer.eval()
    # for i, batch in enumerate(test_loader):
    #     idx = int(batch['idx'][0])
    #     ray = batch['ray'][0]
    #     img_gt = batch['rgb'][0]
    #     zbuf = test_buf[idx]
    #     idbuf = test_buf_id[idx]
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #         with record_function("model_inference"):
    #             output = renderer(zbuf, idbuf, ray, gt=None,
    #                       mask_gt=None, isTrain=False, xyz_o=None)
    #     print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
    #     break

    # renderer = Renderer_fast3(args)
    # renderer.eval()
    # for i, batch in enumerate(test_loader):
    #     idx = int(batch['idx'][0])
    #     ray = batch['ray'][0]
    #     img_gt = batch['rgb'][0]
    #     zbuf = test_buf[idx]
    #     idbuf = test_buf_id[idx]
    #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #         with record_function("model_inference"):
    #             output = renderer(zbuf, idbuf, ray, gt=None,
    #                       mask_gt=None, isTrain=False, xyz_o=None)
    #     print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
    #     break
    
    renderer = Renderer_ex(args)
    renderer.eval()
    for i, batch in enumerate(test_loader):
        idx = int(batch['idx'][0])
        ray = batch['ray'][0]
        img_gt = batch['rgb'][0]
        zbuf = test_buf[idx]
        idbuf = test_buf_id[idx]
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            with record_function("model_inference"):
                output = renderer(zbuf, idbuf, ray, gt=None,
                          mask_gt=None, isTrain=False, xyz_o=None)
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
        break