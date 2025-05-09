import configargparse
import os
import numpy as np
import torch
import cv2
from flip.data import *
from flip.flip_api import *

mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument("--expname", type=str, help='the name of experiment')
    parser.add_argument("--logdir", type=str, help='log directory')
    parser.add_argument("--datadir", type=str, help='data directory')
    parser.add_argument("--pcdir", type=str, help='point cloud directory')

    parser.add_argument("--version", type=int, help='render version', default=1)

    parser.add_argument("--points_per_pixel", type=int, default=8, help='rasterize points_per_pixel')
    parser.add_argument("--use_crop", action='store_true', default=False, help='whether crop when training')
    parser.add_argument("--mpn_tiny", action='store_true', default=False, help='enable mpn_tiny')
    parser.add_argument("--af_mlp", action='store_true', default=False, help='enable frebpcr af_mlp')

    parser.add_argument("--epochs", type=int, default=-1, help='train epochs. Keep training until CTRL-C, if less than 0')
    parser.add_argument("--radius", type=float, help='the radius of points when rasterizing')
    parser.add_argument("--frag_path", type=str, help='directory of saving fragments')
    parser.add_argument("--H", type=int)
    parser.add_argument("--W", type=int)
    parser.add_argument("--train_size", type=int, help='window size of training')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--device", type=str)
    parser.add_argument("--scale_min", type=float, help='the minimum area ratio when random resize and crop')
    parser.add_argument("--scale_max", type=float, help='the maximum area ratio when random resize and crop')

    parser.add_argument("--dim", type=int, default=8, help='feature dimension of radiance mapping output')
    parser.add_argument("--u_lr", type=float, help='learning rate of unet')
    parser.add_argument("--mlp_lr", type=float, help='learning rate of mlp')
    parser.add_argument("--xyznear", action='store_true', default=False, help='corrdinates rectification or not')
    parser.add_argument("--pix_mask", action='store_true', default=False, help='using pixel mask or not')
    parser.add_argument("--U", type=int, help='down sampling times of unet')
    parser.add_argument("--every_k_points", default=1, type=int, help='open3d uniform downsample, param: every_k_points')
    parser.add_argument("--udim", type=str, help='layers dimension of unet')
    parser.add_argument("--vgg_l", type=float, help='the weight of perceptual loss')
    parser.add_argument("--edge_mask", default=0, type=int, help='used in ScanNet 0000_00')
    parser.add_argument("--test_freq", default=10, type=int, help='test every ${test_freq} epoch')
    parser.add_argument("--vid_freq", default=10, type=int, help='save test result every ${vid_freq} epochs')
    parser.add_argument("--pad", type=int, help='num of padding')

    parser.add_argument("--use_fourier", action='store_true', default=False, help='whether to enable fourier_encoding')
    return parser

def load_fragments(args):
    train_name = str(args.radius) + '-z-' + str(args.H) + '-' + \
        str(args.points_per_pixel) + '-train.npy'
    test_name = str(args.radius) + '-z-' + str(args.H) + '-' + \
        str(args.points_per_pixel) +'-test.npy'
    train_path = os.path.join(args.frag_path, train_name) 
    test_path = os.path.join(args.frag_path, test_name)
    train_buf = np.load(train_path)
    test_buf = np.load(test_path)
    print('Load fragments from', train_name, test_name)
    return torch.tensor(train_buf), torch.tensor(test_buf)

def load_color_fragments(args):
    train_name = str(args.radius) + '-color-' + str(args.H) + '-' + \
        str(args.points_per_pixel) + '-train.npy'
    test_name = str(args.radius) + '-color-' + str(args.H) + '-' + \
        str(args.points_per_pixel) +'-test.npy'
    train_path = os.path.join(args.frag_path, train_name) 
    test_path = os.path.join(args.frag_path, test_name)
    train_buf = np.load(train_path)
    test_buf = np.load(test_path)
    print('Load fragments from', train_name, test_name)
    return torch.tensor(train_buf), torch.tensor(test_buf)

def load_idx(args):
    train_name = str(args.radius) + '-idx-' + str(args.H) + '-' + \
        str(args.points_per_pixel) +'-train.npy'
    test_name = str(args.radius) + '-idx-' + str(args.H) + '-' + \
        str(args.points_per_pixel) +'-test.npy'
    train_path = os.path.join(args.frag_path, train_name) 
    test_path = os.path.join(args.frag_path, test_name)
    train_buf = np.load(train_path)
    test_buf = np.load(test_path)
    print('Load fragments from', train_name, test_name)
    return torch.tensor(train_buf), torch.tensor(test_buf)

def lr_decay(opt):
    for p in opt.param_groups:
        p['lr'] = p['lr'] * 0.9999

def write_video(path, savepath, size):
    file_list = sorted(os.listdir(path))
    fps = 20
    four_cc = cv2.VideoWriter_fourcc(*'MJPG')
    save_path = savepath
    video_writer = cv2.VideoWriter(save_path, four_cc, float(fps), size)
    for item in file_list:
        if item.endswith('.jpg') or item.endswith('.png'):
            item = path + '/' + item
            img = cv2.imread(item)
            video_writer.write(img)

    video_writer.release()
    cv2.destroyAllWindows()

def check_nans(reference, test, verbosity=3):
    """
    Code from nvidia-flip
    Checks reference and test images for NaNs and sets NaNs to 0. Depending on verbosity level, warns if NaNs occur

    :param reference: float tensor
    :param test: float tensor
    :param verbosity: (optional) integer describing level of verbosity.
                      0: no printed output,
                      1: print mean FLIP error,
                      "2: print pooled FLIP errors, PPD, and evaluation time and (for HDR-FLIP) start and stop exposure and number of exposures"
                      3: print pooled FLIP errors, PPD, warnings, and evaluation time and (for HDR-FLIP) start and stop exposure and number of exposures
    :return: two float tensors
    """
    if (np.isnan(reference)).any() or (np.isnan(test)).any():
        reference = np.nan_to_num(reference)
        test = np.nan_to_num(test)
        if verbosity == 3:
            print('=====================================================================')
            print('WARNING: either reference or test (or both) images contain NaNs.')
            print('Those values have been set to 0.')
            print('=====================================================================')
    return reference, test

def color_space_transform(input_color, fromSpace2toSpace):
    """
    Transforms inputs between different color spaces

    :param input_color: tensor of colors to transform (with CxHxW layout)
    :param fromSpace2toSpace: string describing transform
    :return: transformed tensor (with CxHxW layout)
    """
    dim = input_color.shape

    # Assume D65 standard illuminant
    reference_illuminant = np.array([[[0.950428545]], [[1.000000000]], [[1.088900371]]]).astype(np.float32)
    inv_reference_illuminant = np.array([[[1.052156925]], [[1.000000000]], [[0.918357670]]]).astype(np.float32)

    if fromSpace2toSpace == "srgb2linrgb":
        limit = 0.04045
        transformed_color = np.where(input_color > limit, np.power((input_color + 0.055) / 1.055, 2.4), input_color / 12.92)

    elif fromSpace2toSpace == "linrgb2srgb":
        limit = 0.0031308
        transformed_color = np.where(input_color > limit, 1.055 * (input_color ** (1.0 / 2.4)) - 0.055, 12.92 * input_color)

    elif fromSpace2toSpace == "linrgb2xyz" or fromSpace2toSpace == "xyz2linrgb":
        # Source: https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
        # Assumes D65 standard illuminant
        if fromSpace2toSpace == "linrgb2xyz":
            a11 = 10135552 / 24577794
            a12 = 8788810  / 24577794
            a13 = 4435075  / 24577794
            a21 = 2613072  / 12288897
            a22 = 8788810  / 12288897
            a23 = 887015   / 12288897
            a31 = 1425312  / 73733382
            a32 = 8788810  / 73733382
            a33 = 70074185 / 73733382
        else:
            # Constants found by taking the inverse of the matrix
            # defined by the constants for linrgb2xyz
            a11 = 3.241003275
            a12 = -1.537398934
            a13 = -0.498615861
            a21 = -0.969224334
            a22 = 1.875930071
            a23 = 0.041554224
            a31 = 0.055639423
            a32 = -0.204011202
            a33 = 1.057148933
        A = np.array([[a11, a12, a13],
                      [a21, a22, a23],
                      [a31, a32, a33]]).astype(np.float32)

        input_color = np.transpose(input_color, (2, 0, 1)) # C(H*W)
        transformed_color = np.matmul(A, input_color)
        transformed_color = np.transpose(transformed_color, (1, 2, 0))

    elif fromSpace2toSpace == "xyz2ycxcz":
        input_color = np.multiply(input_color, inv_reference_illuminant)
        y = 116 * input_color[1:2, :, :] - 16
        cx = 500 * (input_color[0:1, :, :] - input_color[1:2, :, :])
        cz = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])
        transformed_color = np.concatenate((y, cx, cz), 0)

    elif fromSpace2toSpace == "ycxcz2xyz":
        y = (input_color[0:1, :, :] + 16) / 116
        cx = input_color[1:2, :, :] / 500
        cz = input_color[2:3, :, :] / 200

        x = y + cx
        z = y - cz
        transformed_color = np.concatenate((x, y, z), 0)

        transformed_color = np.multiply(transformed_color, reference_illuminant)

    elif fromSpace2toSpace == "xyz2lab":
        input_color = np.multiply(input_color, inv_reference_illuminant)
        delta = 6 / 29
        delta_square = delta * delta
        delta_cube = delta * delta_square
        factor = 1 / (3 * delta_square)

        input_color = np.where(input_color > delta_cube, np.power(input_color, 1 / 3), (factor * input_color + 4 / 29))

        l = 116 * input_color[1:2, :, :] - 16
        a = 500 * (input_color[0:1,:, :] - input_color[1:2, :, :])
        b = 200 * (input_color[1:2, :, :] - input_color[2:3, :, :])

        transformed_color = np.concatenate((l, a, b), 0)

    elif fromSpace2toSpace == "lab2xyz":
        y = (input_color[0:1, :, :] + 16) / 116
        a =  input_color[1:2, :, :] / 500
        b =  input_color[2:3, :, :] / 200

        x = y + a
        z = y - b

        xyz = np.concatenate((x, y, z), 0)
        delta = 6 / 29
        factor = 3 * delta * delta
        xyz = np.where(xyz > delta,  xyz ** 3, factor * (xyz - 4 / 29))

        transformed_color = np.multiply(xyz, reference_illuminant)

    elif fromSpace2toSpace == "srgb2xyz":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color,'linrgb2xyz')
    elif fromSpace2toSpace == "srgb2ycxcz":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == "linrgb2ycxcz":
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2ycxcz')
    elif fromSpace2toSpace == "srgb2lab":
        transformed_color = color_space_transform(input_color, 'srgb2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == "linrgb2lab":
        transformed_color = color_space_transform(input_color, 'linrgb2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    elif fromSpace2toSpace == "ycxcz2linrgb":
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
    elif fromSpace2toSpace == "lab2srgb":
        transformed_color = color_space_transform(input_color, 'lab2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2linrgb')
        transformed_color = color_space_transform(transformed_color, 'linrgb2srgb')
    elif fromSpace2toSpace == "ycxcz2lab":
        transformed_color = color_space_transform(input_color, 'ycxcz2xyz')
        transformed_color = color_space_transform(transformed_color, 'xyz2lab')
    else:
        sys.exit('Error: The color transform %s is not defined!' % fromSpace2toSpace)

    return transformed_color

def compute_ldrflip(reference, test, pixels_per_degree=(0.7 * 3840 / 0.7) * np.pi / 180):
    """
    Computes the FLIP error map between two LDR images,
    assuming the images are observed at a certain number of
    pixels per degree of visual angle

    :param reference: reference image (with CxHxW layout on float32 format with values in the range [0, 1] in the sRGB color space)
    :param test: test image (with CxHxW layout on float32 format with values in the range [0, 1] in the sRGB color space)
    :param pixels_per_degree: (optional) float describing the number of pixels per degree of visual angle of the observer,
                              default corresponds to viewing the images on a 0.7 meters wide 4K monitor at 0.7 meters from the display
    :return: matrix (with 1xHxW layout on float32 format) containing the per-pixel FLIP errors (in the range [0, 1]) between LDR reference and test image
    """
    # Set color and feature exponents
    qc = 0.7
    qf = 0.5

    # Transform reference and test to opponent color space
    reference = color_space_transform(reference, 'srgb2ycxcz')
    test = color_space_transform(test, 'srgb2ycxcz')

    # --- Color pipeline ---
    # Spatial filtering
    s_a = generate_spatial_filter(pixels_per_degree, 'A')
    s_rg = generate_spatial_filter(pixels_per_degree, 'RG')
    s_by = generate_spatial_filter(pixels_per_degree, 'BY')
    filtered_reference = spatial_filter(reference, s_a, s_rg, s_by)
    filtered_test = spatial_filter(test, s_a, s_rg, s_by)

    # Perceptually Uniform Color Space
    preprocessed_reference = hunt_adjustment(color_space_transform(filtered_reference, 'linrgb2lab'))
    preprocessed_test = hunt_adjustment(color_space_transform(filtered_test, 'linrgb2lab'))

    # Color metric
    deltaE_hyab = hyab(preprocessed_reference, preprocessed_test)
    hunt_adjusted_green = hunt_adjustment(color_space_transform(np.array([[[0.0]], [[1.0]], [[0.0]]]).astype(np.float32), 'linrgb2lab'))
    hunt_adjusted_blue = hunt_adjustment(color_space_transform(np.array([[[0.0]], [[0.0]], [[1.0]]]).astype(np.float32), 'linrgb2lab'))
    cmax = np.power(hyab(hunt_adjusted_green, hunt_adjusted_blue), qc)
    deltaE_c = redistribute_errors(np.power(deltaE_hyab, qc), cmax)

    # --- Feature pipeline ---
    # Extract and normalize achromatic component
    reference_y = (reference[0:1, :, :] + 16) / 116
    test_y = (test[0:1, :, :] + 16) / 116

    # Edge and point detection
    edges_reference = feature_detection(reference_y, pixels_per_degree, 'edge')
    points_reference = feature_detection(reference_y, pixels_per_degree, 'point')
    edges_test = feature_detection(test_y, pixels_per_degree, 'edge')
    points_test = feature_detection(test_y, pixels_per_degree, 'point')

    # Feature metric
    deltaE_f = np.maximum(abs(np.linalg.norm(edges_reference, axis=0) - np.linalg.norm(edges_test, axis=0)),
                          abs(np.linalg.norm(points_test, axis=0) - np.linalg.norm(points_reference, axis=0)))
    deltaE_f = np.power(((1 / np.sqrt(2)) * deltaE_f), qf)

    # --- Final error ---
    return np.power(deltaE_c, 1 - deltaE_f)

def flip_error_map(reference, pre):
    """
    Computes the FLIP error map between reference and pre for tensorboard
    assuming the images are observed at a certain number of
    pixels per degree of visual angle

    :param reference: reference image (with HxWxC layout on float32 format with values in the range [0, 1] in the sRGB color space)
    :param test: test image (with HxWxC layout on float32 format with values in the range [0, 1] in the sRGB color space)
    :return: matrix (with HxWxC layout on float32 format) containing the per-pixel FLIP errors (in the range [0, 1]) between LDR reference and pre image
    """
    gt_img = reference.detach().cpu().permute(2, 0 ,1)
    test_img = pre.detach().cpu().permute(2, 0, 1)
    flip_img = compute_ldrflip(gt_img, test_img).squeeze(0)
    error_map = CHWtoHWC(index2color(np.round(flip_img * 255.0), get_magma_map()))
    return error_map