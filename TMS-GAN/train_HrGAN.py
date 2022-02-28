import os
import sys
import time
import torch
import numpy
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.parallel
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.models as models
import torch.backends.cudnn as cudnn
cudnn.fastest = True
cudnn.benchmark = True
#from skimage import measure
from math import log10
from myutils.vgg16 import Vgg16
from torchvision import transforms
from torch.utils.data import DataLoader
from data_png import get_train, get_test
from data import get_val
from HrGAN import Discriminator, Generator

import warnings
warnings.filterwarnings("ignore")  
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--haze_path',         type=str,   default='./data/HrGAN/train/hazy')
parser.add_argument('--clean_path',        type=str,   default='./data/HrGAN/train/gt'  )
parser.add_argument('--thaze_path',        type=str,   default='./data/HrGAN/test/hazy' )
parser.add_argument('--tclean_path',       type=str,   default='./data/HrGAN/test/gt'   )
parser.add_argument('--vhaze_path',        type=str,   default='./data/HrGAN/val/hazy'  )
parser.add_argument('--vclean_path',       type=str,   default='./data/HrGAN/val/gt'    )
parser.add_argument('--netD',              type=str,   default='' )
parser.add_argument('--netG',              type=str,   default='./HrGAN/last.pth'  )
parser.add_argument('--lambdaIMG',         type=float, default=1.0                 )
parser.add_argument('--lambdaPER',         type=float, default=1.0                 )
parser.add_argument('--lambdaGRA',         type=float, default=0.1                 )
parser.add_argument('--lambdaADV',         type=float, default=0.001               )
parser.add_argument('--lrD',               type=float, default=0.00010             )
parser.add_argument('--lrG',               type=float, default=0.00010             )
parser.add_argument('--annealStart',       type=int,   default=0                   )
parser.add_argument('--annealEvery',       type=int,   default=105                 )
parser.add_argument('--epochs',            type=int,   default=100                 )
parser.add_argument('--workers',           type=int,   default=0                   )
parser.add_argument('--BN',                type=int,   default=6                   )
parser.add_argument('--VBN',               type=int,   default=4                   )
parser.add_argument('--TBN',               type=int,   default=1                   )
parser.add_argument('--exp',               type=str,   default='HrGAN'             ) 
parser.add_argument('--display',           type=int,   default=100                 )
parser.add_argument('--evalIter',          type=int,   default=1000                )
opt = parser.parse_args()

#opt.manualSeed = random.randint(1, 10000)
opt.manualSeed = 101
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
        
'''梯度损失函数'''
def Gradient_loss(y):
    gradient_h=torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])
    gradient_y=torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])
    return gradient_h, gradient_y

# 读取数据      
create_exp_dir(opt.exp)

data_transform = transforms.Compose([transforms.Resize([320, 480]), transforms.ToTensor()])

train_dataset = get_train(opt.clean_path, opt.haze_path, 256)
test_dataset  = get_test(opt.tclean_path, opt.thaze_path)
val_dataset   = get_val(opt.vclean_path, opt.vhaze_path, data_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=opt.BN, shuffle=False, num_workers=opt.workers, drop_last=True, pin_memory=True)
test_loader  = DataLoader(dataset=test_dataset, batch_size=opt.TBN, shuffle=False, num_workers=opt.workers, drop_last=False, pin_memory=True)
val_loader   = DataLoader(dataset=val_dataset, batch_size=opt.VBN, shuffle=False, num_workers=opt.workers, drop_last=False, pin_memory=True)

# 记录文件
trainLogger = open('%s/train.log' % opt.exp, 'w')

netD = Discriminator().cuda()
netG = nn.DataParallel(Generator())
netG = netG.cuda()

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

netD.train()  
netG.train()

# 损失
criterionMAE = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()
criterionPSNR = nn.MSELoss(size_average=True).cuda()
criterionBCE = nn.BCEWithLogitsLoss(reduction='mean').cuda()

# 参数
lambdaIMG = opt.lambdaIMG
lambdaPER = opt.lambdaPER
lambdaGRA = opt.lambdaGRA
lambdaADV = opt.lambdaADV

# 初始化 VGG-16
vgg = Vgg16()
model_dict = vgg.state_dict()
vgg16 = models.vgg16(pretrained=True)
pretrained_dict = vgg16.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict) 
vgg.load_state_dict(model_dict)
vgg.cuda()
for param in vgg.parameters():
    param.requires_grad = False

# 优化器
optimizerD = optim.Adam(netD.parameters(), lr = opt.lrD, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lrG, betas = (0.9, 0.999))
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD , step_size=1, gamma=0.98)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG , step_size=1, gamma=0.98)

'''模型训练'''
print('Total-parameter-NetG: %s ' % (get_parameter_number(netG)) )
print('Total-parameter-NetD: %s ' % (get_parameter_number(netD)) )
best_epoch = {'epoch':0, 'psnr':0}

for epoch in range(opt.epochs):
    print()
    Loss_D = 0.0
    Loss_adv = 0.0
    Loss_img = 0.0
    Loss_con = 0.0
    Loss_gra = 0.0
    real_dis_D = 0.0
    fake_dis_D = 0.0
    start_time = time.time()
    ganIterations = 0

#    if epoch+1 > opt.annealStart:  # 调整学习率
#        adjust_learning_rate(optimizerD, opt.lrD, opt.annealEvery)
#        adjust_learning_rate(optimizerG, opt.lrG, opt.annealEvery)
        
    for i, data_train in enumerate(train_loader):
        haze, clean = data_train 
        haze, clean = haze.cuda(), clean.cuda()
        # 更新判别器网络参数
        for p in netD.parameters():
            p.requires_grad = True      
        netD.zero_grad()
        # 真假
        real_D = netD(clean)  
        dehaze = netG(haze)
        fake_D = netD(dehaze.detach())
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
        # 对抗损失
        real_G = netD(clean)
        fake_G = netD(dehaze)
        real_logit_G = real_G - torch.mean(fake_G)
        fake_logit_G = fake_G - torch.mean(real_G)
        real_loss_G = criterionBCE(real_logit_G, torch.zeros_like(real_logit_G))
        fake_loss_G = criterionBCE(fake_logit_G, torch.ones_like(fake_logit_G))
        adv_loss = lambdaADV * (real_loss_G + fake_loss_G)
        # 图像损失
        img_loss = criterionMAE(dehaze, clean)
        img_loss = lambdaIMG * img_loss
        # 感知损失
        features_clean  = vgg(clean) 
        features_dehaze = vgg(dehaze)                      
        con_loss_1 = criterionMAE(features_dehaze[4], features_clean[4].detach())
        con_loss_2 = criterionMAE(features_dehaze[5], features_clean[5].detach())
        con_loss_3 = criterionMAE(features_dehaze[6], features_clean[6].detach())
        con_loss = lambdaPER * (con_loss_1 + con_loss_2 + con_loss_3)
#        # 梯度损失
#        gradie_h_clean,  gradie_v_clean  = Gradient_loss(clean) 
#        gradie_h_detail, gradie_v_detail = Gradient_loss(detail)
#        gra_loss_1 = criterionMAE(gradie_h_detail, gradie_h_clean) 
#        gra_loss_2 = criterionMAE(gradie_v_detail, gradie_v_clean) 
#        gra_loss = lambdaGRA * (gra_loss_1 + gra_loss_2)
        
        # 总损失
#        G_loss = img_loss + con_loss + gra_loss + adv_loss
        G_loss = img_loss + con_loss + adv_loss
        
        G_loss.backward() 
        Loss_adv += adv_loss.item()
        Loss_img += img_loss.item()
        Loss_con += con_loss.item()
#        Loss_gra += gra_loss.item()
        
        optimizerG.step()
    
        ganIterations += 1 # 迭代次数加1
        
        # 损失展示
        if ganIterations % opt.display == 0:
            print('[%d/%d][%d/%d]|img:%f|con:%f|gra:%f|adv:%f|real:%f|fake:%f' % (epoch+1,opt.epochs,i+1,len(train_loader),Loss_img*opt.BN,Loss_con*opt.BN,Loss_gra*opt.BN,Loss_adv*opt.BN,real_dis_D,fake_dis_D))
            sys.stdout.flush()
            trainLogger.write('[%d/%d][%d/%d]|img:%f|con:%f|gra:%f|adv:%f|real:%f|fake:%f\n' % (epoch+1,opt.epochs,i+1,len(train_loader),Loss_img*opt.BN,Loss_con*opt.BN,Loss_gra*opt.BN,Loss_adv*opt.BN,real_dis_D,fake_dis_D))
            trainLogger.flush()
            Loss_D = 0.0
            Loss_adv = 0.0
            Loss_img = 0.0
            Loss_con = 0.0
            Loss_gra = 0.0
            real_dis_D = 0.0
            fake_dis_D = 0.0
            
        if ganIterations % opt.evalIter == 0:
            netG.eval()
            with torch.no_grad():
                for k, data_val in enumerate(val_loader):
                    val_haze, val_clean = data_val
                    val_haze, val_clean = val_haze.cuda(), val_clean.cuda()
                    val_dehaze = netG(val_haze)
#                    vutils.save_image(val_detail, '%s/d_epoch_%d_iter%06d.png' % (opt.exp, epoch+1, ganIterations), normalize=False, scale_each=False)
                    vutils.save_image(val_dehaze, '%s/h_epoch_%d_iter%06d.png' % (opt.exp, epoch+1, ganIterations), normalize=False, scale_each=False)
            netG.train()  
            
    print("Epoch: %d Learning rate: D: %f G: %f" % (epoch+1, optimizerD.param_groups[0]['lr'], optimizerG.param_groups[0]['lr']))
    schedulerD.step() 
    schedulerG.step() 
    
    # 模型测试
    print('Model test')
    netG.eval()
    psnrs = []
    with torch.no_grad():
        for j, data_test in enumerate(test_loader):
            test_haze, test_clean = data_test
            test_haze, test_clean = test_haze.cuda(), test_clean.cuda()
            test_dehaze = netG(test_haze)
            dehaze = torch.clamp(test_dehaze, min=0, max=1)
            mse = criterionPSNR(dehaze, test_clean)
            psnr = 10 * log10(1 / mse)
            psnrs.append(psnr)                                                     
    psnr_avg = np.mean(psnrs)
    print('Eval Result: [%d/%d] | PSNR: %f' % (epoch+1, opt.epochs, psnr_avg))
    sys.stdout.flush()
    trainLogger.write('Eval Result: [%d/%d] | PSNR: %f\n' % (epoch+1, opt.epochs, psnr_avg) )
    trainLogger.flush()
    netG.train()
    
    # 保存最佳模型
    if psnr_avg > best_epoch['psnr']:
        torch.save(netD.state_dict(), '%s/netD_epoch%d.pth' % (opt.exp, epoch+1))
        torch.save(netG.state_dict(), '%s/netG_epoch%d.pth' % (opt.exp, epoch+1))
        best_epoch['psnr'] = psnr_avg
        best_epoch['epoch'] = epoch+1 

    total_time = time.time() - start_time
    print('Total-Time: {:.6f} '.format(total_time))  
    
trainLogger.close()

