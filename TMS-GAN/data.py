import os
import cv2
import glob
import torch
import random
from PIL import Image


class get_train(torch.utils.data.Dataset):
    def __init__(self, clean_root, haze_root, transforms):
        self.clean_root = clean_root
        self.haze_root = haze_root
        self.image_name_list = glob.glob(os.path.join(self.clean_root, '*.png'))  # 读取所有 clean目录下的 png 
        self.file_list = []
        self.get_image_pair_list()                                                
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        clean_image_name, haze_image_name = self.file_list[item]
        clean_image   = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(clean_image_name),   cv2.COLOR_BGR2RGB)))
        haze_image    = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(haze_image_name),    cv2.COLOR_BGR2RGB)))
        return clean_image, haze_image
    
    def __len__(self):
        return len(self.file_list)

    def get_image_pair_list(self):
        for image in self.image_name_list:
            key = image.split(self.clean_root)[-1]                                # 示例 \4_3.png
            self.file_list.append([self.clean_root+key, self.haze_root+key])
        random.shuffle(self.file_list)



class get_test(torch.utils.data.Dataset):
    def __init__(self, clean_root, haze_root, transforms):
        self.clean_root = clean_root
        self.haze_root = haze_root
        self.image_name_list = glob.glob(os.path.join(self.clean_root, '*.png'))  # 读取所有 clean目录下的 png 
        self.file_list = []
        self.get_image_pair_list()                                                
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        clean_image_name, haze_image_name = self.file_list[item]
        clean_image   = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(clean_image_name),   cv2.COLOR_BGR2RGB)))
        haze_image    = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(haze_image_name),    cv2.COLOR_BGR2RGB)))
        return haze_image, clean_image
    
    def __len__(self):
        return len(self.file_list)

    def get_image_pair_list(self):
        for image in self.image_name_list:
            key = image.split(self.clean_root)[-1]                                # 示例 \4_3.png
            self.file_list.append([self.clean_root+key, self.haze_root+key])



class get_val(torch.utils.data.Dataset):
    def __init__(self, clean_root, haze_root, transforms):
        self.clean_root = clean_root
        self.haze_root = haze_root
        self.image_name_list = glob.glob(os.path.join(self.clean_root, '*.png'))  # 读取所有 clean目录下的 png 
        self.file_list = []
        self.get_image_pair_list()                                                
        self.transforms = transforms
        print("Total data examples:", len(self.file_list))

    def __getitem__(self, item):
        clean_image_name, haze_image_name = self.file_list[item]
        clean_image   = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(clean_image_name),   cv2.COLOR_BGR2RGB)))
        haze_image    = self.transforms(Image.fromarray(cv2.cvtColor(cv2.imread(haze_image_name),    cv2.COLOR_BGR2RGB)))
        return haze_image, clean_image
    
    def __len__(self):
        return len(self.file_list)

    def get_image_pair_list(self):
        for image in self.image_name_list:
            key = image.split(self.clean_root)[-1]                                # 示例 \4_3.png
            self.file_list.append([self.clean_root+key, self.haze_root+key])





