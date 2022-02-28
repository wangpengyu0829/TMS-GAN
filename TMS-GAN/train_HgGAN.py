import os
import sys
import time
import torch
#import numpy
import random
import argparse
#import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.fastest = True
from myutils.vgg16 import Vgg16
from torch.utils.data import DataLoader
from data_HgGAN import get_train, get_test
from HgGAN import Discriminator, Generator
import warnings
warnings.filterwarnings("ignore")  
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--real_path',         type=str,   default='./data/HgGAN/train/real' )
parser.add_argument('--haze_path',         type=str,   default='./data/HgGAN/train/synth')
parser.add_argument('--clean_path',        type=str,   default='./data/HgGAN/train/clean')
parser.add_argument('--thaze_path',        type=str,   default='./data/HgGAN/test/synth' )
parser.add_argument('--tclean_path',       type=str,   default='./data/HgGAN/test/clean' )
parser.add_argument('--netD',              type=str,   default=''                  )
parser.add_argument('--netG',              type=str,   default=''                  )
parser.add_argument('--PP',                type=float, default=1.0                 )
parser.add_argument('--PA',                type=float, default=0.001               )
parser.add_argument('--lrG',               type=float, default=0.0001              )
parser.add_argument('--lrD',               type=float, default=0.0001              )
parser.add_argument('--annealStart',       type=int,   default=0                   )
parser.add_argument('--annealEvery',       type=int,   default=75                  )
parser.add_argument('--epochs',            type=int,   default=50                  )
parser.add_argument('--workers',           type=int,   default=0                   )
parser.add_argument('--BN',                type=int,   default=8                   )
parser.add_argument('--test_BN',           type=int,   default=4                   )
parser.add_argument('--exp',               type=str,   default='HgGAN'             ) 
parser.add_argument('--display',           type=int,   default=50                  )
opt = parser.parse_args()

#opt.manualSeed = random.randint(1, 10000)
opt.manualSeed = 501
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

'''参数量'''
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

'''创建路径'''
def create_exp_dir(exp):
    try:
        os.makedirs(exp)
        print('Creating exp dir: %s' % exp)
    except OSError:
        pass
    return True
          
'''学习率衰减'''        
def adjust_learning_rate(optimizer, init_lr, epoch):
    lrd = init_lr / epoch 
    old_lr = optimizer.param_groups[0]['lr']
    lr = old_lr - lrd
    if lr < 0: lr = 0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
# 读取数据      
create_exp_dir(opt.exp)
train_dataset = get_train(opt.real_path, opt.haze_path, opt.clean_path, 256)
test_dataset  = get_test(opt.thaze_path, opt.tclean_path,)

train_loader = DataLoader(dataset=train_dataset, batch_size=opt.BN, shuffle=False, num_workers=opt.workers, drop_last=True, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=opt.test_BN, shuffle=False, num_workers=opt.workers, drop_last=False, pin_memory=True)

# 记录文件
trainLogger = open('%s/train.log' % opt.exp, 'w')

netD = Discriminator().cuda()
netG = Generator().cuda()

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

netD.train()
netG.train()

# 损失
criterionMSE = nn.MSELoss().cuda()
criterionBCE = nn.BCEWithLogitsLoss(reduction='mean').cuda()
lambdaPP = opt.PP
lambdaPA = opt.PA

# 初始化 VGG-16
vgg = Vgg16()
model_dict = vgg.state_dict()
vgg16 = models.vgg16(pretrained=True)
pretrained_dict = vgg16.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict) 
vgg.load_state_dict(model_dict)
vgg.cuda()
for p2 in vgg.parameters():
    p2.requires_grad = False

# 优化器
optimizerD = optim.Adam(netD.parameters(), lr = opt.lrD, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (0.5, 0.999))
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD , step_size=1, gamma=0.98)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG , step_size=1, gamma=0.98)

'''模型训练'''
print('Total-parameter-NetG: %s ' % (get_parameter_number(netG)) )
print('Total-parameter-NetD: %s ' % (get_parameter_number(netD)) )
best_epoch = {'epoch':0, 'psnr':0}

for epoch in range(opt.epochs):
    print()
    start_time = time.time()
    ganIterations = 0
    Loss_D = 0.0  
    Loss_con = 0.0
    Loss_adv = 0.0
    real_dis_D = 0.0
    fake_dis_D = 0.0
    
#    if epoch+1 > opt.annealStart:  # 调整学习率
#        adjust_learning_rate(optimizerD, opt.lrD, opt.annealEvery)
#        adjust_learning_rate(optimizerG, opt.lrG, opt.annealEvery)
        
    for i, data_train in enumerate(train_loader):
        real, haze, clean = data_train 
        # gt 
        real, haze, clean = real.cuda(), haze.cuda(), clean.cuda()
        
        # 更新判别器网络参数
        for p in netD.parameters():
            p.requires_grad = True      
        netD.zero_grad()
        # 真
        real_D = netD(real)  
        # 假
        fake_haze = netG(haze, clean)
        fake_D = netD(fake_haze.detach())
        # 损失
        real_logit_D = real_D - torch.mean(fake_D)
        fake_logit_D = fake_D - torch.mean(real_D)
        real_dis_D += F.sigmoid(real_logit_D).mean().item()*(1/opt.display)
        fake_dis_D += F.sigmoid(fake_logit_D).mean().item()*(1/opt.display)
        real_loss_D = criterionBCE(real_logit_D, torch.ones_like(real_logit_D))
        fake_loss_D = criterionBCE(fake_logit_D, torch.zeros_like(fake_logit_D))
        D_loss = real_loss_D + fake_loss_D
        Loss_D += D_loss.item()
        # 反向传播  
        D_loss.backward()
        optimizerD.step()  
        
        # 固定判别器，更新生成器网络参数 
        for p in netD.parameters():
            p.requires_grad = False
        netG.zero_grad()
        
        # 动态判别损失
        real_G = netD(real)
        fake_G = netD(fake_haze)
        real_logit_G = real_G - torch.mean(fake_G)
        fake_logit_G = fake_G - torch.mean(real_G)
        real_loss_G = criterionBCE(real_logit_G, torch.zeros_like(real_logit_G))
        fake_loss_G = criterionBCE(fake_logit_G, torch.ones_like(fake_logit_G))
        adv_loss = real_loss_G + fake_loss_G
        
        # 图像损失 256
#        features_dehaze = vgg(fake_haze)
#        features_clean  = vgg(haze)                                  
#        con_loss_1 = criterionMSE(features_dehaze[4], features_clean[4].detach())
#        con_loss_2 = criterionMSE(features_dehaze[5], features_clean[5].detach())
#        con_loss_3 = criterionMSE(features_dehaze[6], features_clean[6].detach())
#        con_loss   = con_loss_1 + con_loss_2 + con_loss_3
        
        con_loss = criterionMSE(fake_haze, haze)
  
        # 总损失
        Loss_con += con_loss.item()
        Loss_adv += adv_loss.item()
        total_loss = lambdaPP*con_loss + lambdaPA*adv_loss

        total_loss.backward()    
        optimizerG.step()
    
        ganIterations += 1 # 迭代次数加1
        
        # 损失展示
        if ganIterations % opt.display == 0:
            print('[%d/%d][%d/%d] con: %f G: %f D: %f real: %f fake: %f' % (epoch+1, opt.epochs, i+1, len(train_loader), Loss_con*opt.BN, lambdaPA*Loss_adv*opt.BN, lambdaPA*Loss_D*opt.BN, real_dis_D, fake_dis_D) )
            sys.stdout.flush()
            trainLogger.write('[%d/%d][%d/%d] con: %f Gd: %f\n' % (epoch+1, opt.epochs, i+1, len(train_loader), Loss_con*opt.BN, lambdaPA*Loss_adv*opt.BN) )
            trainLogger.flush()
            
            Loss_D = 0.0
            Loss_con = 0.0
            Loss_adv = 0.0
            real_dis_D = 0.0
            fake_dis_D = 0.0
    
    print("Epoch: %d Learning rate: D: %f G: %f" % (epoch+1, optimizerD.param_groups[0]['lr'], optimizerG.param_groups[0]['lr']))
    schedulerD.step() 
    schedulerG.step() 
    # 模型测试, SSIM and PSNR
    print('Model eval')
    netG.eval()
    with torch.no_grad():
        for j, data_test in enumerate(test_loader):
            test_haze, test_clean = data_test
            test_haze, test_clean = test_haze.cuda(), test_clean.cuda()
            test_fake_haze = netG(test_haze, test_clean)
            vutils.save_image(test_fake_haze, '%s/fake_epoch%d.png' % (opt.exp, epoch+1), normalize=False, scale_each=False)
            if epoch == 0:
                vutils.save_image(test_haze, '%s/synthetic_haze.png' % (opt.exp), normalize=False, scale_each=False)
    netG.train()
    
    # 保存最佳模型
    torch.save(netD.state_dict(), '%s/netD_epoch%d.pth' % (opt.exp, epoch+1))
    torch.save(netG.state_dict(), '%s/netG_epoch%d.pth' % (opt.exp, epoch+1))
    total_time = time.time() - start_time
    print('Total-Time: {:.6f} '.format(total_time))  
    
trainLogger.close()



