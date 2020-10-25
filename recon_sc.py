'''
INFERENCE CODE FOR VIDEO RECONSTRUCTION FROM A SINGLE CODED EXPOSURE IMAGE

'''
import os 
import torch
import glob
import argparse
import numpy as np

from sensor import C2B
from unet import UNet
import utils

 
# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--savepath', type=str, default=None, help='path to save results')
args = parser.parse_args()

# create directory to save results
if args.savepath is not None:
    if not os.path.exists(args.savepath):
        os.makedirs(os.path.join(args.savepath, 'frames'))
        os.makedirs(os.path.join(args.savepath, 'gifs'))
        os.makedirs(os.path.join(args.savepath, 'images'))

# input params
im_height = 720
im_width = 1200
subframes = 9

# load test videos
data_path = os.path.join(os.getcwd(), 'data', 'test_videos', 'seq*', '*.png')
image_paths = sorted(glob.glob(data_path))
print('Test videos: %d'%(len(image_paths)//subframes))
assert len(image_paths) % subframes == 0

# load weights
c2b = C2B(code_size=int(np.sqrt(subframes)))
unet = UNet(in_channel=subframes, out_channel=subframes)

if torch.cuda.is_available():
    print('Running inference on GPU...')
    c2b = c2b.cuda()
    unet = unet.cuda()
    weights = torch.load(os.path.join('weights', 'single-coded-inp.pth'))
else:
    print('Running inference on CPU...')
    weights = torch.load(os.path.join('weights', 'single-coded-inp.pth'), 
                        map_location=lambda storage, loc: storage)
unet.load_state_dict(weights['unet_state_dict'])
unet.eval()

# inference
# image size must be divisible by code size
csize = np.sqrt(subframes)
if im_height % csize != 0:
    im_height -= im_height % csize
if im_width % csize != 0:
    im_width -= im_width % csize

psnr_sum = 0.
ssim_sum = 0.
psnr_mid_sum = 0
ssim_mid_sum = 0
if args.savepath is not None:
    log = open(os.path.join(args.savepath, 'log.txt'), 'w')

with torch.no_grad():
    for seq in range(len(image_paths)//subframes):
        vid = []
        for sf in range(subframes):
            sframe = utils.read_image(image_paths[seq*subframes+sf], im_height, im_width)
            vid.append(sframe)
        vid = torch.stack(vid, dim=0).unsqueeze(0)

        b1, _, _ = c2b(vid)
        lowres_vid = utils.solve_constraints_single(b1, c2b.code)
        highres_vid = unet(lowres_vid).clamp(0,1)
        
        psnr, ssim = utils.compute_psnr_ssim(highres_vid[0], vid[0])
        psnr_sum += psnr
        ssim_sum += ssim
        print('Test video: %d PSNR: %.2f SSIM: %.3f'%(seq+1, psnr, ssim))        
        mid_idx = (subframes-1)//2
        psnr_mid, ssim_mid = utils.compute_psnr_ssim(highres_vid[0,mid_idx:mid_idx+1,:,:], 
                                                    vid[0,mid_idx:mid_idx+1,:,:])
        psnr_mid_sum += psnr_mid
        ssim_mid_sum += ssim_mid
        print('Mid frame PSNR: %.2f SSIM: %.3f'%(psnr_mid, ssim_mid))

        # save results
        if args.savepath is not None:
            log.write('Test video: %d PSNR: %.2f SSIM: %.3f\n'%(seq+1, psnr, ssim))
            log.write('Mid frame PSNR: %.2f SSIM: %.3f\n'%(psnr_mid, ssim_mid))
            utils.save_image(b1[0], 
                os.path.join(args.savepath, 'images', 'seq_%.2d_codedInput.png'%(seq+1)))
            utils.save_gif(vid[0], 
                os.path.join(args.savepath, 'gifs', 'seq_%.2d_groundTruth.gif'%(seq+1)))
            utils.save_gif(highres_vid[0], 
                os.path.join(args.savepath, 'gifs', 'seq_%.2d_recon.gif'%(seq+1)))
            for sf in range(subframes):
                utils.save_image(highres_vid[0,sf,:,:], 
                    os.path.join(args.savepath, 'frames', 'seq_%.2d_recon%.1d.png'%(seq+1, sf+1)))
                utils.save_image(vid[0,sf,:,:], 
                    os.path.join(args.savepath, 'frames', 'seq_%.2d_groundTruth%.1d.png'%(seq+1, sf+1)))

    print('\nAverage PSNR: %.2f'%(psnr_sum/(len(image_paths)//subframes)))
    print('Average SSIM: %.3f'%(ssim_sum/(len(image_paths)//subframes)))
    print('Mid frame average PSNR: %.2f'%(psnr_mid_sum/(len(image_paths)//subframes)))
    print('Mid frame average SSIM: %.3f'%(ssim_mid_sum/(len(image_paths)//subframes)))
    if args.savepath is not None:
        log.write('\nAverage PSNR: %.2f\n'%(psnr_sum/(len(image_paths)//subframes)))
        log.write('Average SSIM: %.3f\n'%(ssim_sum/(len(image_paths)//subframes)))
        log.write('Mid frame average PSNR: %.2f\n'%(psnr_mid_sum/(len(image_paths)//subframes)))
        log.write('Mid frame average SSIM: %.3f\n'%(ssim_mid_sum/(len(image_paths)//subframes)))  
        log.close()
        print('Saved results to %s'%args.savepath)      
