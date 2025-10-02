# -*- coding: utf-8 -*-
# @Time    : 21/6/9 17:01
# @Author  : qgking
# @Email   : qgking@tju.edu.cn
# @Software: PyCharm
# @Desc    : SarcopeniaDataLoader.py
import sys

from torchvision.transforms import transforms

sys.path.extend(["../../", "../", "./"])
from torch.utils.data import Dataset
from skimage.transform import resize
import nibabel as nib
from commons.constant import *
from commons.utils import *
import pandas as pd
import SimpleITK as sitk
import numpy as np
import os
from monai.transforms import Compose, RandAffined, RandSpatialCropSamplesd, RandRotated, \
    RandAdjustContrastd, RandGaussianSmoothd, RandHistogramShiftd, \
    RandFlipd, RandScaleIntensityd, RandRotate90d, NormalizeIntensityd, ScaleIntensityd, \
    ScaleIntensityRangePercentilesd, RandGridPatchd, RandCoarseDropoutd, RandGaussianNoised, RandZoomd, \
    NormalizeIntensity, ScaleIntensity

TMP_DIR = "./tmp"
if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)
TEXT_COLS = ['AGE', 'Gender(M:0,F:1)', '身高cm', '體重kg', 'BMI']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


# 通过图片路径读取图片, 返回img[h, w, c*3], width, height
def open_image_file(filepath):
    if basename(filepath).endswith('.dcm'):
        img = read_dicom(filepath)
        nor_image = normalize_scale(img)
        img = (nor_image * 255).astype(np.uint8)  # 归一化后，像素值再分散到 [0, 255]
        size_x, size_y, _ = img.shape  # height, width, channel*3
    elif basename(filepath).endswith('.png'):
        img_pil = Image.open(filepath)
        size_y, size_x = img_pil.size  # width, height
        img = np.asarray(img_pil.convert('RGB'))  # height, width
    else:
        raise Exception('Unknown file extension')
    return img, size_y, size_x  # img[h, w, c*3], width, height


class SarcopeniaBasePTHDataSet(Dataset):
    def __init__(self, sarcopenia_paths_dfs, input_size=(256, 256), augment=None, text_only=False):
        self.input_x = input_size[0]
        self.input_y = input_size[1]
        self.sarcopeniaidx = []
        self.sarcopenialines = []
        self.augment = augment
        self.img_final_dfs = sarcopenia_paths_dfs
        self.text_only = text_only
        self.transform = Compose([
            RandRotated(keys=['img', 'mask'], range_x=10, range_y=10, prob=0.5, allow_missing_keys=True),
            RandZoomd(keys=['img', 'mask'], min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            RandFlipd(keys=['img', 'mask'], prob=0.5, spatial_axis=0, allow_missing_keys=True),
            RandFlipd(keys=['img', 'mask'], prob=0.5, spatial_axis=1, allow_missing_keys=True),
        ])

        self.normalizeD = Compose([
            NormalizeIntensityd(keys=['img', 'mask'], nonzero=True, allow_missing_keys=True),
            ScaleIntensityd(keys=['img', 'mask'], minv=0.0, maxv=1.0, allow_missing_keys=True)
        ])
        self.pos_dfs = []
        self.neg_dfs = []

        for dfs in sarcopenia_paths_dfs:
            cat = dfs[CAT]
            if cat == 0:
                self.neg_dfs.append(dfs)  # 负样本
            elif cat == 1:
                self.pos_dfs.append(dfs)  # 正样本
            else:
                raise Exception('Unknown CAT')

        print('Load images: %d,positive: %d, negative: %d' % (
            len(self.img_final_dfs), len(self.pos_dfs), len(self.neg_dfs)))

    def __len__(self):
        return int(len(self.img_final_dfs))

    def __getitem__(self, index):
        return


class SarcopeniaClsContrastivePTHDataSet(SarcopeniaBasePTHDataSet):
    def __init__(self, sarcopenia_paths_dfs, input_size, augment, text_only):
        super(SarcopeniaClsContrastivePTHDataSet, self).__init__(sarcopenia_paths_dfs, input_size, augment, text_only)
        self.norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        dfs = self.img_final_dfs[index]
        img = dfs[IMG]
        path = dfs[PATH]
        name = dfs[NAME]
        cat = dfs[CAT]
        nums = dfs[NUMS]

        mask = np.repeat(self.load_mask(index)[0].unsqueeze(0).numpy(), 3, 0)
        samples = {'img': img, 'mask': mask}
        if self.augment is not None:
            samples_xi = self.normalizeD(self.transform(samples))
            samples_xj = self.normalizeD(self.transform(samples))
        else:
            samples_xi = self.normalizeD(samples)
            samples_xj = self.normalizeD(samples)
        mask_xi = samples_xi['img'].as_tensor() + samples_xi['mask'].as_tensor()
        mask_xj = samples_xj['img'].as_tensor() + samples_xj['mask'].as_tensor()
        numerical_tensor = torch.from_numpy(rearrange(nums, 'b -> () b'))
        return {
            "image_patch": (samples_xi['img'].as_tensor(), samples_xj['img'].as_tensor()),
            "image_cat": cat,  # 是否肌小症(0/1)
            "image_name": name,  # 人名
            "image_path": path,  # .dcm图片路径
            "image_text": numerical_tensor,  # nums数组转化为tensor张量
            "mask": (mask_xi, mask_xj)
        }

    def load_mask(self, index):
        mask_name = self.img_final_dfs[index]['path'].split('/')[-2]
        mask_path = f'/root/disk1/qgking/czx/data/Sarcopenia/AdaptSAM/x512/{mask_name}.jpg'
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
        ])
        try:
            img = Image.open(mask_path)
            img_tensor = transform(img)
        except FileNotFoundError:
            print(mask_path, ' NOT FOUNT')
            img_tensor = None
        assert img_tensor is not None
        return img_tensor


class SarcopeniaClsPTHDataSet(SarcopeniaBasePTHDataSet):
    def __init__(self, sarcopenia_paths_dfs, input_size, augment, text_only):
        super(SarcopeniaClsPTHDataSet, self).__init__(sarcopenia_paths_dfs, input_size, augment, text_only)

    def __getitem__(self, index):
        dfs = self.img_final_dfs[index]
        img = dfs[IMG]
        path = dfs[PATH]
        name = dfs[NAME]
        cat = dfs[CAT]
        nums = dfs[NUMS]

        mask = np.repeat(self.load_mask(index)[0].unsqueeze(0).numpy(), 3, 0)
        samples = {'img': img, 'mask': mask}
        if self.augment is not None:
            samples = self.normalizeD(self.transform(samples))
        else:
            samples = self.normalizeD(samples)
        numerical_tensor = torch.from_numpy(rearrange(nums, 'b -> () () b'))

        mask = samples['img'].as_tensor() + samples['mask'].as_tensor()

        return {
            "image_patch": samples['img'].as_tensor(),  # 经过数据增强后的图片
            "image_cat": cat,  # 是否肌小症(0/1)
            "image_name": name,  # 人名
            "image_path": path,  # .dcm图片路径
            "image_text": numerical_tensor,  # nums数组转化为tensor张量
            "mask": mask,
        }

    def load_mask(self, index):
        mask_name = self.img_final_dfs[index]['path'].split('/')[-2]
        mask_path = f'/root/disk1/qgking/czx/data/Sarcopenia/AdaptSAM/x512/{mask_name}.jpg'
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x)
        ])
        try:
            img = Image.open(mask_path)
            img_tensor = transform(img)
        except FileNotFoundError:
            print(mask_path, ' NOT FOUNT')
            img_tensor = None
        assert img_tensor is not None
        return img_tensor


# (h, w, c*3)
def read_dicom(scan_path):
    try:
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(scan_path))
        # c h w
    except Exception as er:
        print(er)
    # print(os.path.basename(scan_path))
    # return height,width,channel
    img_len = len(image_array.shape)
    if img_len == 4:
        image_array = image_array.squeeze()
    if img_len == 3:
        image_array = np.transpose(image_array, (1, 2, 0))  # h, w, c
        image_array = np.tile(image_array, 3)  # 通道层重复三次
    return image_array
