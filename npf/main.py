import logging
import os
import time
import torch
from utils import config_parser, load_fragments, load_idx, lr_decay, write_video, mse2psnr
from dataset.dataset import nerfDataset, ScanDataset, DTUDataset
from model.renderer import Renderer
import matplotlib.pyplot as plt
import torch.optim as optim
from backup_utils import backup_terminal_outputs, backup_code, set_seed
from torch.utils.tensorboard import SummaryWriter
from piqa import SSIM, LPIPS, PSNR
import lpips
# flip
from utils import flip_error_map
# disable vgg warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

DEBUG = False
# DEBUG = True

parser = config_parser()
args = parser.parse_args()

set_seed(42)
log_dir = './debug_logs/' if DEBUG else './logs/'

if args.use_crop:
    back_path = os.path.join(log_dir, time.strftime(
    "%y%m%d-%H%M%S-" + f'{args.expname}-{args.H}-crop{args.train_size}-dim{args.dim}-zbuf{args.points_per_pixel}-pix{args.pix_mask}-xyznear{args.xyznear}'))
else:
    back_path = os.path.join(log_dir, time.strftime(
    "%y%m%d-%H%M%S-" + f'{args.expname}-{args.H}-nocrop-dim{args.dim}-zbuf{args.points_per_pixel}-pix{args.pix_mask}-xyznear{args.xyznear}'))


os.makedirs(back_path)
backup_terminal_outputs(back_path)
backup_code(back_path, ignored_in_current_folder=[
            'debug_logs', 'back', 'pointcloud', 'data', '.git', 'pytorch_rasterizer.egg-info', 'build', 'logs', '__pycache__', '.sh', 'dev_scripts'])
print(back_path)
logger = SummaryWriter(back_path)
video_path = os.path.join(back_path, 'video')
os.makedirs(video_path)

formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


stdlog = setup_logger('bpcr', os.path.join(back_path, 'metric.txt'))

if __name__ == '__main__':
    def log_string(str):
        stdlog.info(str)
        print(str)

    log_string(args)

    if args.dataset == 'nerf':
        train_set = nerfDataset(args, 'train', 'render')
        test_set = nerfDataset(args, 'test', 'render')
    elif args.dataset == 'scan':
        train_set = ScanDataset(args, 'train', 'render')
        test_set = ScanDataset(args, 'test', 'render')
    elif args.dataset == 'dtu':
        train_set = DTUDataset(args, 'train', 'render')
        test_set = DTUDataset(args, 'test', 'render')
    else:
        assert False

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)

    renderer = Renderer(args)
    edge = args.edge_mask

    # Optimizer
    opt_para = []
    opt_para.append({"params": renderer.mpn.parameters(), "lr": 1.5e-4})
    opt_para.append({"params": renderer.unet.parameters(), "lr": args.u_lr})
    opt_para.append({"params": renderer.mlp.parameters(), "lr": args.mlp_lr})
    opt = optim.Adam(opt_para)

    fn_psnr = PSNR().to(args.device)
    fn_lpips = LPIPS('vgg').to(args.device)
    loss_lpips = lpips.LPIPS(net='vgg').to(args.device)
    fn_ssim = SSIM().to(args.device)

    # load buf
    if args.xyznear:
        train_buf, test_buf = load_fragments(args)  # cpu 100 800 800 k
        xyz_o = None
    else:
        train_buf, test_buf = load_idx(args)  # cpu 100 800 800 k
        xyz_o = train_set.get_pc().xyz  # n 3

    log_string(f'zbuf shape: {train_buf.shape}')



    if args.ckpt is not None:
        print(f'load model from {args.ckpt}')
        renderer.load_state_dict(torch.load(args.ckpt))

    it = 0
    epoch = 0
    training_time = 0.

    best_lpips = 1
    best_psnr = 0
    best_ssim = 0

    log_string(f"\n 🚀 [START Training]\n")
    while True:
        renderer.train()
        t1 = time.time()
        epoch += 1
        for batch in train_loader:
            it += 1
            idx = int(batch['idx'][0])
            ray = batch['ray'][0]  # h w 7
            img_gt = batch['rgb'][0]  # h w 3
            if args.dataset == 'dtu':
                mask_gt = batch['mask'][0][..., :1]  # h w 1
            else:
                mask_gt = None

            zbuf = train_buf[idx].to(args.device)  # h w 1

            output = renderer(zbuf, ray, img_gt, mask_gt,
                              isTrain=True, xyz_o=xyz_o)

            if args.dataset == 'dtu':
                img_pre = output['img'] * \
                    output['mask_gt'] + 1 - output['mask_gt']
                # img_pre = output['img']
            else:
                img_pre = output['img']

            if output['gt'].min() == 1:
                print('None img, skip')
                torch.cuda.empty_cache()
                continue

            opt.zero_grad()
            # if edge > 0:
            #     loss_l2 = torch.mean((img_pre - output['gt'])[edge:-edge, edge:-edge] ** 2)
            # else:
            loss_l2 = torch.mean((img_pre - output['gt']) ** 2)

            if args.vgg_l > 0:
                loss_vgg = loss_lpips(img_pre.permute(2, 0, 1).unsqueeze(
                    0), output['gt'].permute(2, 0, 1).unsqueeze(0), normalize=True)
                loss = loss_l2 + args.vgg_l * loss_vgg
            else:
                loss = loss_l2

            loss.backward()
            opt.step()

            if it % 50 == 0:
                psnr = mse2psnr(loss_l2)
                logger.add_scalar('train/psnr', psnr.item(), it)

            if it % 200 == 0:
                if args.vgg_l > 0:
                    print('[{}]-it:{}, psnr:{:.4f}, l2_loss:{:.4f}, vgg_loss:{:.4f}'.format(
                        back_path, it, psnr.item(), loss_l2.item(), loss_vgg.item()))
                else:
                    print('[{}]-it:{}, psnr:{:.4f}, l2_loss:{:.4f}'.format(back_path,
                          it, psnr.item(), loss.item()))
                img_pre[img_pre > 1] = 1.
                img_pre[img_pre < 0] = 0.
                # logger.add_image(
                #     'train/fea', output['fea_map'], global_step=it, dataformats='HWC')
                logger.add_image('train/predict', img_pre,
                                 global_step=it, dataformats='HWC')
                logger.add_image(
                    'train/gtimg', output['gt'], global_step=it, dataformats='HWC')
                logger.add_image('train/flip_error',  flip_error_map(
                    output['gt'], img_pre), global_step=it, dataformats='HWC')

            del output
            torch.cuda.empty_cache()

        lr_decay(opt)
        t2 = time.time()
        training_time += (t2 - t1) / 3600

        # test
        if epoch % args.test_freq == 0:
            print('TEST BEGIN!!!')
            if epoch % args.vid_freq == 0:
                video_it_path = os.path.join(video_path, str(it))
                os.makedirs(video_it_path)

            test_psnr = 0
            test_lpips = 0
            test_ssim = 0

            renderer.eval()
            with torch.autograd.no_grad():
                for i, batch in enumerate(test_loader):
                    idx = int(batch['idx'][0])
                    ray = batch['ray'][0]
                    img_gt = batch['rgb'][0]
                    zbuf = test_buf[idx].to(args.device)
                    output = renderer(zbuf, ray, gt=None,
                                      mask_gt=None, isTrain=False, xyz_o=xyz_o)

                    if args.dataset == 'dtu':
                        mask_gt = batch['mask'][0][..., :1]
                        img_pre = output['img'].detach(
                        )[..., :3] * mask_gt + 1 - mask_gt
                    else:
                        img_pre = output['img']

                    img_pre[img_pre > 1] = 1.
                    img_pre[img_pre < 0] = 0.

                    # save test 200 images
                    # logger.add_image('test/gtimg', img_gt, global_step=i, dataformats='HWC')
                    # logger.add_image('test/predict', img_pre, global_step=i, dataformats='HWC')
                    # logger.add_image('test/flip_error',  flip_error_map(img_gt, img_pre)  , global_step=i, dataformats='HWC')

                    img_pre = img_pre.permute(2, 0, 1).unsqueeze(0)
                    img_gt = img_gt.permute(2, 0, 1).unsqueeze(0)

                    if edge > 0:
                        psnr = fn_psnr(
                            img_pre[..., edge:-edge, edge:-edge], img_gt[..., edge:-edge, edge:-edge])
                        ssim = fn_ssim(
                            img_pre[..., edge:-edge, edge:-edge], img_gt[..., edge:-edge, edge:-edge])
                        lpips_ = fn_lpips(
                            img_pre[..., edge:-edge, edge:-edge], img_gt[..., edge:-edge, edge:-edge])
                    else:
                        psnr = fn_psnr(img_pre, img_gt)
                        ssim = fn_ssim(img_pre, img_gt)
                        lpips_ = fn_lpips(img_pre, img_gt)
                    test_lpips += lpips_.item()
                    test_psnr += psnr.item()
                    test_ssim += ssim.item()

                    # save at logs/*/video/
                    if epoch % args.vid_freq == 0:
                        img_pre = img_pre.squeeze(0).permute(1, 2, 0)
                        img_pre = img_pre.cpu().numpy()
                        plt.imsave(os.path.join(video_it_path, str(
                            i).rjust(3, '0') + '.png'), img_pre)

                    del output
                    torch.cuda.empty_cache()

            test_lpips = test_lpips / len(test_set)
            test_psnr = test_psnr / len(test_set)
            test_ssim = test_ssim / len(test_set)
            logger.add_scalar('test/psnr', test_psnr, it)
            logger.add_scalar('test/lpips', test_lpips, it)
            logger.add_scalar('test/ssim', test_ssim, it)

            if test_psnr > best_psnr:
                print(f'update PSNR, better: {test_psnr-best_psnr}')
                best_psnr = test_psnr
                ckpt = os.path.join(back_path, 'model.pkl')
                torch.save(renderer.state_dict(), ckpt)

            if test_lpips < best_lpips:
                print(f'update lpips, better: {best_lpips-test_lpips}')
                best_lpips = test_lpips
                ckpt = os.path.join(back_path, 'model.pkl')
                torch.save(renderer.state_dict(), ckpt)

            if test_ssim > best_ssim:
                print(f'update SSIM, better: {test_ssim-best_ssim}')
                best_ssim = test_ssim
                ckpt = os.path.join(back_path, 'model.pkl')
                torch.save(renderer.state_dict(), ckpt)

            log_string('------------------------------------------------')
            log_string(f'Training_time: {training_time:{4}.{4}} hours.')
            log_string(f'Test phrase! Epoch:{epoch}, it: {it}')
            log_string(
                f'Current test metric: PSNR: {test_psnr}, LPIPS: {test_lpips}, SSIM: {test_ssim}')
            log_string(
                f'Best test metric: PSNR: {best_psnr}, LPIPS: {best_lpips}, SSIM: {best_ssim}')
            log_string('------------------------------------------------')

            if (args.epochs > 0) and (args.epochs <= epoch):
                print(
                    f"\nReach preset target epoch: {args.epochs}, current epoch: {epoch}")
                print(
                    f"Current test psnr {test_psnr:{4}.{4}}, Best psnr: {best_psnr:{4}.{4}}, \n")
                exit(0)
