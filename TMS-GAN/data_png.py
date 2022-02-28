import os
import cv2
import random
import torch
import numpy as np
import torch.utils.data as data
import math

def is_img1(x):
    if x.endswith('.png') and not(x.startswith('._')):
        return True
    else:
        return False
    
def is_img2(x):
    if x.endswith('.png') and not(x.startswith('._')):
        return True
    else:
        return False
    
def _np2Tensor(img):  
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    tensor = torch.from_numpy(np_transpose).float() # numpy 转化为 tensor
    return tensor

'''读取训练数据'''
class get_train(data.Dataset):
    def __init__(self, clean_path, haze_path, patch_size):
        self.patch_size = patch_size   
        self.haze_path  = haze_path 
        self.clean_path = clean_path 
        self._set_filesystem(self.haze_path, self.clean_path)   # 训练图像路径 / groudtruth 路径
        self.images_h, self.images_c = self._scan()             # 获得训练和 GT 图像
        self.repeat = 12
        
    '''打印路径'''        
    def _set_filesystem(self, dir_h, dir_c):
        self.dir_h = dir_h
        self.dir_c = dir_c
        print('********* Train dir *********')
        print(self.dir_h)
        print(self.dir_c)
        
    '''遍历图像，获取名称集合'''
    def _scan(self):
        list_c = sorted([os.path.join(self.dir_c, x) for x in os.listdir(self.dir_c) if is_img1(x)])  # 遍历 groudtruth 路径中的图像，其名字形成列表
        random.shuffle(list_c)
        list_h = [os.path.splitext(x)[0]+'.png' for x in list_c]
        list_h = [os.path.join(self.dir_h, os.path.split(x)[-1]) for x in list_h]  # 根据list_c中的图像名+.png 遍历训练图像路径中的图像，其名字形成列表
        return list_h, list_c                  

    def __getitem__(self, idx):
        img_h, img_c, filename_h, filename_c = self._load_file(idx)      # 获取图像
        assert img_h.shape==img_c.shape # 大小相等 # 如果可训练
        x = random.randint(0, img_h.shape[0] - self.patch_size)          # img_n.shape = (321, 481, 3)
        y = random.randint(0, img_h.shape[1] - self.patch_size)
        img_h = img_h[x : x+self.patch_size, y : y+self.patch_size, :]   # 随机裁剪一个 patch
        img_c = img_c[x : x+self.patch_size, y : y+self.patch_size, :]
        img_h = _np2Tensor(img_h)                                        # 转化为 tensor
        img_c = _np2Tensor(img_c)
        return img_h, img_c

    def __len__(self):
        return len(self.images_h) * self.repeat
        
    '''依次读取每个 patch 的图像的索引'''
    def _get_index(self, idx):
        return idx % len(self.images_h)   # 余数

    '''依次读取每个 patch 的图像'''
    def _load_file(self, idx):
        idx = self._get_index(idx)    # 选取 idx
        file_h = self.images_h[idx]   # 选取训练图像名
        file_c = self.images_c[idx]   # 选取 GT 图像名
        img_h = cv2.cvtColor(cv2.imread(file_h),   cv2.COLOR_BGR2RGB)    # 读取训练图像
        if np.max(img_h)>1: img_h = img_h/255.0   # 归一化
        if img_h.shape[0]<256: img_h = cv2.resize(img_h, (img_h.shape[1], 256), interpolation=cv2.INTER_CUBIC)
        if img_h.shape[1]<256: img_h = cv2.resize(img_h, (256, img_h.shape[0]), interpolation=cv2.INTER_CUBIC)      
        
        img_c = cv2.cvtColor(cv2.imread(file_c),   cv2.COLOR_BGR2RGB)
        if np.max(img_c)>1: img_c = img_c/255.0    
        if img_c.shape[0]<256: img_c = cv2.resize(img_c, (img_c.shape[1], 256), interpolation=cv2.INTER_CUBIC)
        if img_c.shape[1]<256: img_c = cv2.resize(img_c, (256, img_c.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        filename_h = os.path.splitext(os.path.split(file_h)[-1])[0]   # 训练图像每个图像的图像名
        filename_c = os.path.splitext(os.path.split(file_c)[-1])[0]   # GT图像每个图像的图像名
        return img_h, img_c, filename_h, filename_c                   # 输出图像和图像名
    
    
'''读取训练数据'''
class get_test(data.Dataset):
    def __init__(self, clean_path, haze_path):
        self.haze_path  = haze_path 
        self.clean_path = clean_path 
        self._set_filesystem(self.haze_path, self.clean_path)   # 训练图像路径 / groudtruth 路径
        self.images_h, self.images_c = self._scan()             # 获得训练和 GT 图像
        
    '''打印路径'''        
    def _set_filesystem(self, dir_h, dir_c):
        self.dir_h = dir_h
        self.dir_c = dir_c
        print('********* Test dir *********')
        print(self.dir_h)
        print(self.dir_c)
        
    '''遍历图像，获取名称集合'''
    def _scan(self):
        list_c = sorted([os.path.join(self.dir_c, x) for x in os.listdir(self.dir_c) if is_img2(x)])  # 遍历 groudtruth 路径中的图像，其名字形成列表
        list_h = [os.path.splitext(x)[0]+'.png' for x in list_c]
        list_h = [os.path.join(self.dir_h, os.path.split(x)[-1]) for x in list_h]  # 根据list_c中的图像名.png 遍历训练图像路径中的图像，其名字形成列表
        return list_h, list_c                  

    def __getitem__(self, idx):
        img_h, img_c, filename_h, filename_c = self._load_file(idx)  # 获取图像
        assert img_h.shape==img_c.shape # 大小相等 # 如果可训练
        img_h = _np2Tensor(img_h)       # 转化为 tensor
        img_c = _np2Tensor(img_c)
        return img_h, img_c

    def __len__(self):
        return len(self.images_h)
        
    '''依次读取每个 patch 的图像的索引'''
    def _get_index(self, idx):
        return idx   # 余数

    '''依次读取每个 patch 的图像'''
    def _load_file(self, idx):
        idx = self._get_index(idx)    # 选取 idx
        file_h = self.images_h[idx]   # 选取训练图像名
        file_c = self.images_c[idx]   # 选取 GT 图像名
        
        img_h = cv2.cvtColor(cv2.imread(file_h), cv2.COLOR_BGR2RGB)    # 读取训练图像
        if np.max(img_h)>1: img_h = img_h/255.0   # 归一化
        
        xh = math.floor(img_h.shape[0]/16)*16
        yh = math.floor(img_h.shape[1]/16)*16
        img_h = cv2.resize(img_h, (yh, xh), interpolation=cv2.INTER_CUBIC)
        
        img_c = cv2.cvtColor(cv2.imread(file_c), cv2.COLOR_BGR2RGB)
        if np.max(img_c)>1: img_c = img_c/255.0   # 归一化
        
        xc = math.floor(img_c.shape[0]/16)*16
        yc = math.floor(img_c.shape[1]/16)*16
        img_c = cv2.resize(img_c, (yc, xc), interpolation=cv2.INTER_CUBIC)   
        
        filename_h = os.path.splitext(os.path.split(file_h)[-1])[0]   # 训练图像每个图像的图像名
        filename_c = os.path.splitext(os.path.split(file_c)[-1])[0]   # GT图像每个图像的图像名
        return img_h, img_c, filename_h, filename_c                   # 输出图像和图像名





   