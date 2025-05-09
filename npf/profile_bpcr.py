import os
import time
import torch
from dataset.dataset import nerfDataset
from utils import config_parser, load_fragments, load_idx, lr_decay, write_video, mse2psnr
import matplotlib.pyplot as plt
from backup_utils import backup_terminal_outputs, backup_code, set_seed
from torch.utils.tensorboard import SummaryWriter
from raw_bpcr_model.renderer import BPCR
from torch.profiler import profile, record_function, ProfilerActivity

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
    
    # if args.ckpt is not None:
        # renderer.load_state_dict(torch.load(args.ckpt))
    # edge = args.edge_mask

    # _, test_buf = load_fragments(args)  # cpu 100 800 800 k
    # _, test_buf_id = load_idx(args)  # cpu 100 800 800 k
# 
    # test_buf = test_buf.to(args.device)
    # test_buf_id = test_buf_id.to(args.device)
# 
    # renderer = BPCR(args)
    # renderer.eval()
    # with torch.autograd.no_grad():
        # for i, batch in enumerate(test_loader):
            # idx = int(batch['idx'][0])
            # ray = batch['ray'][0]
            # img_gt = batch['rgb'][0]
            # zbuf = test_buf[idx]
            # idbuf = test_buf_id[idx]
            # output = renderer(zbuf, ray, gt=None,
                #   mask_gt=None, isTrain=False, xyz_o=None)

    renderer = BPCR(args)
    renderer.eval()
    ray = torch.randn([800, 800, 7], device=args.device)
    zbuf = torch.randn([800, 800, 1], device=args.device)
    with profile(activities=[
        ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            output = renderer(zbuf, ray, gt=None, mask_gt=None, isTrain=False, xyz_o=None)

    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
    
# STAGE:2024-01-21 08:28:59 1779093:1779093 ActivityProfilerController.cpp:294] Completed Stage: Warm Up
# STAGE:2024-01-21 08:29:06 1779093:1779093 ActivityProfilerController.cpp:300] Completed Stage: Collection
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                                    Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
#                                         model_inference         0.14%      10.397ms       100.00%        7.306s        7.306s       0.000us         0.00%      38.453ms      38.453ms             1  
#                                            aten::linear         0.00%      68.000us        44.41%        3.245s     648.934ms       0.000us         0.00%      12.045ms       2.409ms             5  
#                                             aten::addmm         0.07%       5.176ms        44.41%        3.244s     648.887ms      12.045ms        43.44%      12.045ms       2.409ms             5  
#                                  ampere_sgemm_128x64_tn         0.00%       0.000us         0.00%       0.000us       0.000us      11.652ms        42.02%      11.652ms       2.913ms             4  
#                                             aten::index         0.01%     470.000us         0.19%      14.188ms       3.547ms     109.000us         0.39%       9.743ms       2.436ms             4  
#                                           aten::nonzero         0.01%     752.000us         0.13%       9.524ms       1.905ms     238.000us         0.86%       9.656ms       1.931ms             5  
#                                        cudaLaunchKernel        45.88%        3.352s        45.88%        3.352s      12.553ms       5.315ms        19.17%       5.315ms      19.906us           267  
#                                            aten::conv2d         0.00%     191.000us        54.66%        3.993s     173.613ms       0.000us         0.00%       5.010ms     217.826us            23  
#                                       aten::convolution         0.01%     571.000us        54.65%        3.993s     173.604ms       0.000us         0.00%       5.010ms     217.826us            23  
#                                      aten::_convolution         0.01%     776.000us        54.65%        3.992s     173.579ms       0.000us         0.00%       5.010ms     217.826us            23  
# -------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
# Self CPU time total: 7.306s
# Self CUDA time total: 27.730ms