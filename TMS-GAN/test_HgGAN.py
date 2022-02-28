import os
import time
import glob
import torch
import argparse
import torch.utils.data
import torch.nn.parallel
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

cudnn.fastest = True
cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore") 

from HgGAN import Generator
from data_HgGAN import get_test  
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--haze_path',         type=str,   default='./data/HgGAN/train/synth')
parser.add_argument('--clean_path',        type=str,   default='./data/HgGAN/train/clean')
parser.add_argument('--netG',              type=str,   default='./HgGAN/netG_epoch47.pth')
parser.add_argument('--workers',           type=int,   default=0                         )
parser.add_argument('--BN',                type=int,   default=1                         )
parser.add_argument('--exp',               type=str,   default='./HgGAN/result/'         )
opt = parser.parse_args()
        
test_dataset = get_test(opt.haze_path, opt.clean_path)
test_loader = DataLoader(dataset=test_dataset, batch_size=opt.BN, shuffle=False, num_workers=opt.workers, drop_last=False, pin_memory=True)

image_name_list = glob.glob(os.path.join(opt.clean_path, '*.png'))

netG = Generator().cuda()

netG.load_state_dict(torch.load(opt.netG))
netG.eval()

directory = opt.exp
if not os.path.exists(directory):
    os.makedirs(directory)

print('Model test')
with torch.no_grad():
    start_time = time.time()
    for i, data_test in enumerate(test_loader, 0):
        key = image_name_list[i].split(opt.clean_path)[-1].split('\\')[-1]
        print(key)
        test_haze, test_clean = data_test
        
        test_haze, test_clean = test_haze.cuda(), test_clean.cuda()
        fake_haze = netG(test_haze, test_clean)
        
        # 保存测试图像
        out_img = fake_haze.data
        vutils.save_image(out_img, directory+key, normalize=False, scale_each=False)
        
total_time = time.time() - start_time
print('Eval Result:  Avg-Time: {:.6f}  \n'.format((total_time/len(image_name_list)))) 



