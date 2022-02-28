import os
import time
import glob
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
from skimage import measure
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
cudnn.fastest = True
cudnn.benchmark = True
import warnings
warnings.filterwarnings("ignore") 

from HrGAN import Generator
from data_png import get_test  
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--haze_path',         type=str,   default='./data/REAL/URHI/1'    )
parser.add_argument('--clean_path',        type=str,   default='./data/REAL/URHI/1'    )
parser.add_argument('--netG',              type=str,   default='./HrGAN2/RESIDE.pth'   )
parser.add_argument('--workers',           type=int,   default=0                       )
parser.add_argument('--BN',                type=int,   default=1                       )
parser.add_argument('--exp',               type=str,   default='./HrGAN2/Real1/'       )
opt = parser.parse_args()
        
test_dataset = get_test(opt.clean_path, opt.haze_path)
test_loader = DataLoader(dataset=test_dataset, batch_size=opt.BN, shuffle=False, num_workers=opt.workers, drop_last=False, pin_memory=True)

image_name_list = glob.glob(os.path.join(opt.clean_path, '*.png'))

#netG = Generator().cuda()
netG = nn.DataParallel(Generator())
netG = netG.cuda()

netG.load_state_dict(torch.load(opt.netG))
netG.eval()

directory = opt.exp
if not os.path.exists(directory):
    os.makedirs(directory)

print('Model test')
netG.eval()
psnrs = []
ssims = []
with torch.no_grad():
    start_time = time.time()
    
    for i, data_test in enumerate(test_loader, 0):
        
        key = image_name_list[i].split(opt.clean_path)[-1].split('\\')[-1]
        print(key)
        
        test_haze, test_clean = data_test
        
        test_haze, test_clean = test_haze.cuda(), test_clean.cuda()
        test_dehaze = netG(test_haze)
        
        # 保存测试图像
        out_img = test_dehaze.data
        vutils.save_image(out_img, directory+key, normalize=False, scale_each=False)
        
        test_dehaze = torch.clamp(test_dehaze, min=0, max=1)

        clean_image  = test_clean.view(test_clean.shape[1], test_clean.shape[2], test_clean.shape[3]).cpu().numpy().astype(np.float32)
        dehaze_image = test_dehaze.view(test_dehaze.shape[1], test_dehaze.shape[2], test_dehaze.shape[3]).cpu().numpy().astype(np.float32)
        clean_image  = np.transpose(clean_image, (1,2,0))
        dehaze_image = np.transpose(dehaze_image, (1,2,0))
               
        psnr = measure.compare_psnr(clean_image, dehaze_image, data_range=1) 
        psnrs.append(psnr)
                                                    
        ssim = measure.compare_ssim(clean_image, dehaze_image, data_range=1, multichannel=True)
        ssims.append(ssim) 
        print('PSNR: %f | SSIM: %f' % (psnr, ssim))

psnr_avg = np.mean(psnrs)
ssim_avg = np.mean(ssims) 
print('Eval Result: PSNR: %f | SSIM: %f' % (psnr_avg, ssim_avg))
total_time = time.time() - start_time
print('Eval Result:  Avg-Time: {:.6f}  \n'.format((total_time/len(image_name_list)))) 


