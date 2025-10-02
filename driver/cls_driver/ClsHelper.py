import sys

sys.path.extend(["../../", "../", "./"])
from commons.utils import *
import torch
from driver.base_train_helper import BaseTrainHelper
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sarcopenia_data.SarcopeniaDataLoader import SarcopeniaClsPTHDataSet, SarcopeniaClsContrastivePTHDataSet
# from driver import transform_local, transform_test
from models import MODELS

plt.rcParams.update({'figure.max_open_warning': 20})


class ClsHelper(BaseTrainHelper):
    def __init__(self, criterions, config):
        super(ClsHelper, self).__init__(criterions, config)

    def init_params(self):
        return

    def create_model(self):
        mm = MODELS[self.config.model](backbone=self.config.backbone, n_channels=self.config.n_channels,
                                       num_classes=self.config.classes, pretrained=True)
        return mm

    def generate_batch(self, batch):
        images = batch['image_patch'].to(self.equipment).float()
        # segment = batch['image_label'].to(self.equipment).long()
        image_cat = batch['image_cat'].to(self.equipment).long()
        image_text = batch['image_text'].to(self.equipment).float()
        image_path = batch['image_path']
        image_name = batch['image_name']
        mask = batch['mask'].to(self.equipment).float()
        return images, image_cat, image_path, image_name, image_text, mask

    def read_data(self, dataloader):
        while True:
            for batch in dataloader:
                images = [inst.to(self.equipment).float() for inst in batch["image_patch"]]
                image_cat = batch["image_cat"]
                image_name = batch["image_name"]
                image_path = batch["image_path"]
                image_text = batch['image_text'].to(self.equipment).float()
                images_with_mask = [inst.to(self.equipment).float() for inst in batch["mask"]]
                if len(image_cat) == self.config.train_batch_size:
                    yield images, image_cat, image_path, image_name, image_text, images_with_mask

    def merge_batch(self, batch):
        image_patch = [torch.unsqueeze(inst["image_patch"], dim=0) for inst in batch]
        image_patch = torch.cat(image_patch, dim=0)
        image_cat = [inst["image_cat"] for inst in batch]
        image_cat = torch.tensor(image_cat)
        image_name = [inst["image_name"] for inst in batch]
        image_path = [inst["image_path"] for inst in batch]
        image_text = [torch.unsqueeze(text, dim=0) for inst in batch for text in inst["image_text"]]
        image_text = torch.cat(image_text, dim=0)
        mask = torch.cat([torch.unsqueeze(inst["mask"], dim=0) for inst in batch], dim=0)
        return {"image_patch": image_patch,
                "image_path": image_path,
                "image_name": image_name,
                "image_cat": image_cat,
                "image_text": image_text,
                "mask": mask,
                }

    def get_data_loader_pth(self, fold, seed=666, text_only=False):

        image_dfs = torch.load(join(self.config.data_path, self.config.data_name))
        # image_dfs is a list of dict, where each dict is formed like
        # {
        #  'img':a np.ndarray with the shape of (c,h,w),#ndarray
        #  'INDEX': idx, # int
        #  'NUMS':numerical tabular data, # ndarray
        #  'cat': class label, # int
        #  'name': name of patient, # str
        #  'path': path of original image, # str
        # }

        train_index, test_index = self.get_n_fold(image_paths=image_dfs, fold=fold, seed=seed)

        train_image_dfs = [image_dfs[index] for index in range(len(image_dfs)) if index in train_index]
        test_image_dfs = [image_dfs[index] for index in range(len(image_dfs)) if index in test_index]

        print("Train images %d: " % (len(train_image_dfs)))
        print("Test images %d: " % (len(test_image_dfs)))

        train_dataset = SarcopeniaClsPTHDataSet(train_image_dfs,
                                                input_size=(self.config.patch_x, self.config.patch_y), augment=True,
                                                text_only=text_only)
        valid_dataset = SarcopeniaClsPTHDataSet(test_image_dfs,
                                                input_size=(self.config.patch_x, self.config.patch_y), augment=None,
                                                text_only=text_only)

        train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size, shuffle=True,
                                  num_workers=self.config.workers,
                                  collate_fn=self.merge_batch,
                                  drop_last=True if len(train_image_dfs) % self.config.train_batch_size == 1 else False)
        valid_loader = DataLoader(valid_dataset, batch_size=self.config.test_batch_size, shuffle=False,
                                  num_workers=self.config.workers,
                                  collate_fn=self.merge_batch)
        test_loader = DataLoader(valid_dataset, batch_size=self.config.test_batch_size, shuffle=False,
                                 num_workers=self.config.workers,
                                 collate_fn=self.merge_batch)

        return train_loader, valid_loader, test_image_dfs, test_loader

    def get_constra_data_loader_pth(self, fold, seed=666, text_only=False):
        image_dfs = torch.load(join(self.config.data_path, 'sarcopenia_all_new_data.pth'))
        train_index, test_index = self.get_n_fold(image_paths=image_dfs, fold=fold, seed=seed)

        train_image_dfs = [image_dfs[index] for index in range(len(image_dfs)) if index in train_index]
        train_dataset = SarcopeniaClsContrastivePTHDataSet(train_image_dfs,
                                                           input_size=(self.config.patch_x, self.config.patch_y),
                                                           augment=True, text_only=text_only)

        train_loader = DataLoader(train_dataset, batch_size=self.config.train_batch_size, shuffle=True,
                                  num_workers=self.config.workers,
                                  drop_last=True if len(train_image_dfs) % self.config.train_batch_size == 1 else False)
        return train_loader
