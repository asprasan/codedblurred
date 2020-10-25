'''
HELPER FUNCTIONS

'''
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from skimage.measure import compare_psnr, compare_ssim

from matplotlib import cm
import matplotlib.pyplot as plt
from math import exp


if torch.cuda.is_available():
    print('GPU available!')
    tensor_type = torch.cuda.FloatTensor
else:
    print('GPU not available!')
    tensor_type = torch.FloatTensor


def read_image(path, height, width):
    '''Read image from path'''
    img = Image.open(path).convert(mode='L')
    img = img.resize((width, height), Image.ANTIALIAS)
    img = torch.from_numpy(np.array(img)).type(tensor_type)
    return img/255


def save_image(img, path, normalize=False):
    '''Save image to path'''
    img = img.squeeze(0).cpu().numpy()
    if normalize:
        img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    img = img.clip(0,1)
    img = Image.fromarray((img*255.).astype('uint8'))
    img.save(path)


def save_gif(arr, path):
    '''Save array of frames as gif to path'''
    arr = arr.cpu().numpy()
    frames = []
    for sf in range(arr.shape[0]):
        img = arr[sf].clip(0,1)
        img = Image.fromarray((img*255).astype('uint8'))
        frames.append(img)
    frame1 = frames[0]
    frame1.save(path, save_all=True, append_images=frames[1:], duration=300, loop=0)


def reverse_pixel_shuffle(img, factor):
    '''Perform reverse of pixel shuffle operation'''
    # shuffle (N,1,H,W) image to (N,f,H/f,W/f)
    shuffle_filter = torch.eye(factor**2).type(tensor_type)
    shuffle_filter = shuffle_filter.view(factor**2, 1, factor, factor)
    shuffled = F.conv2d(img, shuffle_filter, stride=factor)
    return shuffled


def compute_psnr_ssim(pred, target):
    '''Compute PSNR and SSIM'''
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    psnr = compare_psnr(pred, target)
    ssim = 0
    for sf in range(target.shape[0]):
        ssim += compare_ssim(pred[sf], target[sf], gaussian_weights=True, 
                                sigma=1.5, use_sample_covariance=False)
    ssim /= target.shape[0]
    return psnr, ssim


def solve_constraints(b1, b0, code):
    '''Solve for low-res video from coded exposure image and complement exposure image'''
    csize = code.shape[-1]
    vec_filter = torch.eye(csize**2).type(tensor_type)
    vec_filter = vec_filter.view(csize**2, 1, csize, csize)

    N, _, H, W = b1.size()
    # reshaping/shuffling b1 and b0
    vec_b1 = F.conv2d(b1, vec_filter, stride=csize)
    vec_b0 = F.conv2d(b0, vec_filter, stride=csize)
    b1_b0 = torch.cat([vec_b1, vec_b0], dim=1).view(N, 1, 2*(csize**2), -1)
    
    code_mat = code.contiguous().view(csize**2, csize**2).transpose(0,1)
    comp_mat = 1 - code_mat
    code_concat = torch.cat([code_mat, comp_mat], dim=0)
    code_concat = code_concat / torch.sum(code_concat, dim=1, keepdim=True)
    code_pinv = torch.pinverse(code_concat, rcond=1e-3)
    inverse_filter = code_pinv.unsqueeze(1).unsqueeze(3)  
    lowres_vid = F.conv2d(b1_b0, inverse_filter, stride=1)
    lowres_vid = lowres_vid.view(N, csize**2, H//csize, W//csize)
    return lowres_vid


def solve_constraints_single(b1, code):
    '''Solve for low-res video from coded exposure image'''
    csize = code.shape[-1]
    vec_filter = torch.eye(csize**2).type(tensor_type)
    vec_filter = vec_filter.view(csize**2, 1, csize, csize)

    N, _, H, W = b1.size()
    # reshaping b1
    vec_b1 = F.conv2d(b1, vec_filter, stride=csize)
    vec_b1 = vec_b1.view(N, 1, csize**2, -1)
    
    code_mat = code.contiguous().view(csize**2, csize**2).transpose(0,1)
    code_mat = code_mat / torch.sum(code_mat, dim=1, keepdim=True)
    code_pinv = torch.pinverse(code_mat)
    inverse_filter = code_pinv.unsqueeze(1).unsqueeze(3)
    lowres_vid = F.conv2d(vec_b1, inverse_filter, stride=1)
    lowres_vid = lowres_vid.view(N, csize**2, H//csize, W//csize)
    return lowres_vid

