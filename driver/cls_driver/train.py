# -*- coding: utf-8 -*-
import shutil
import sys
from os import makedirs
from os.path import exists

sys.path.extend(["../../", "../", "./"])
from commons.utils import *
import torch
from torch.cuda import empty_cache
import random
from driver.cls_driver.ClsHelper import ClsHelper
from driver.Config import Configurable
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.utils import clip_grad_norm_
from commons.evaluation import calculate_metrics
import numpy as np
from commons.visualization_utils import t_sne, auc_roc
import matplotlib
from module.losses import InstanceLoss
import torch.nn.functional as F
matplotlib.use("Agg")
import configparser

config = configparser.ConfigParser()
import argparse
import ssl


def main(config, seed=666, writer=None):
    assert writer is not None
    criterion = {
        'loss': InstanceLoss(config.train_batch_size, 0.5),  # temperature[0.07,0.2],一般取0.1
        'cls_loss': CrossEntropyLoss(),
        'bce_loss': BCEWithLogitsLoss(),
    }
    cls_help = ClsHelper(criterion, config)
    cls_help.move_to_cuda()
    print("data name ", cls_help.config.data_name)
    print("data patch x ", cls_help.config.patch_x)
    print("Random dataset Seed: %d" % seed)
    start_fold = 0
    best_acc_history = []
    avg_metrics = np.zeros(6)
    for fold in range(start_fold, cls_help.config.nfold):
        print("**********Start train fold %d: **********" % (fold))
        train_dataloader, vali_loader, _, test_loader = cls_help.get_data_loader_pth(fold=fold, seed=seed)
        train_con_dataloader = cls_help.get_constra_data_loader_pth(fold=fold, seed=seed)
        bad_step = 0
        best_accuracy = 0
        test_acc = 0
        best_test_acc = 0
        cls_help.log.flush()
        cls_help.reset_model()
        optimizer = cls_help.reset_optim()

        for epoch in range(cls_help.config.epochs):
            train_losses = train(cls_help, train_dataloader, train_con_dataloader, optimizer, epoch)
            vali_critics = valid(cls_help, vali_loader)
            print(" Vali acc epoch %d: current = %.4f" % (epoch, vali_critics['vali/acc']))
            writer.add_scalar(f'fold_{fold}/loss/train_cls_loss', train_losses['train/cls_loss'], epoch)
            writer.add_scalar(f'fold_{fold}/loss/train_ins_loss', train_losses['train/ins_loss'], epoch)
            writer.add_scalar(f'fold_{fold}/acc/valid_acc', vali_critics['vali/acc'], epoch)
            cls_help.log.flush()
            opt_s = ''
            for g in optimizer.param_groups:
                opt_s += "optimizer current_lr to %.8f \t" % (g['lr'])
            print(opt_s)
            if vali_critics['vali/acc'] >= best_accuracy:
                print(" * Best vali acc at epoch %d: history = %.4f, current = %.4f" % (
                epoch, best_accuracy, vali_critics['vali/acc']))
                best_metrics = test(cls_help, test_loader, epoch, fold, thres=0.5)

                best_test_acc = best_metrics[0]
                best_accuracy = vali_critics['vali/acc']
                cls_help.save_best_checkpoint(model_optimizer=optimizer, save_model=True, fold=fold)
                bad_step = 0
            else:
                bad_step += 1
                if bad_step >= cls_help.config.patience:
                    cls_help.save_last_checkpoint(model_optimizer=optimizer, save_model=True, fold=fold)
            cls_help.save_last_checkpoint(model_optimizer=optimizer, save_model=True, fold=fold)
            writer.add_scalar(f'fold_{fold}/acc/test_acc', test_acc, epoch)

        avg_metrics = avg_metrics + best_metrics
        best_acc_history.append(best_test_acc)
        print("\n**********Finish train fold %d, best acc: %.4f: **********" % (fold, best_acc_history[fold]))
        print(
            "best metrics: ACC %0.4f, precision %0.4f, F1score %0.4f, sensitivity %0.4f, specificity %0.4f, AUC %0.4f" % (
                best_metrics[0], best_metrics[1], best_metrics[2], best_metrics[3], best_metrics[4], best_metrics[5]))
    avg_metrics = avg_metrics / cls_help.config.nfold
    for i in range(5):
        print("\nAcc of fold %d: %.4f" % (i, best_acc_history[i]))
    print("Average of best acc: %.4f" % np.mean(best_acc_history))
    print(
        "Metrics: ACC %0.4f, precision %0.4f, f1score %0.4f, sensitivity %0.4f, spc %0.4f, auc %0.4f" % (
            avg_metrics[0], avg_metrics[1], avg_metrics[2], avg_metrics[3], avg_metrics[4], avg_metrics[5]))
    cls_help.log.flush()
    cls_help.summary_writer.close()


def train(cls_help, train_dataloader, train_con_dataloader, optimizer, epoch):
    cls_help.model.train()

    results = None
    optimizer.zero_grad()
    batch_num = int(np.ceil(len(train_dataloader.dataset) / float(cls_help.config.train_batch_size)))
    total_iter = batch_num * cls_help.config.epochs
    con_data_reader = cls_help.read_data(train_con_dataloader)
    for i, batch in enumerate(train_dataloader):
        cls_help.adjust_learning_rate_g(optimizer, epoch * batch_num + i, total_iter, istuning=False)
        images, image_cat, image_path, image_name, image_text, images_with_mask = cls_help.generate_batch(batch)
        con_data, con_cat, con_path, con_name, con_text, con_masks = next(con_data_reader)
        logits, _ = cls_help.model(images, images_with_mask, text=image_text, text_included=True)
        _, con0 = cls_help.model(con_data[0], con_masks[0], text=con_text, text_included=True)
        _, con1 = cls_help.model(con_data[1], con_masks[1], text=con_text, text_included=True)

        loss_instance = cls_help.criterions['loss'](con0, con1)
        loss_cls = cls_help.criterions['cls_loss'](logits, image_cat)

        loss = loss_cls + 0.001 * loss_instance
        loss.backward()
        result = [loss_cls.item(), loss_instance.item()]

        if results is None:
            results = Averagvalue(len([*result]))
        results.update(result, images.size(0))
        if (i + 1) % cls_help.config.update_every == 0 or i == batch_num - 1:
            clip_grad_norm_(filter(lambda p: p.requires_grad, cls_help.model.parameters()), \
                            max_norm=cls_help.config.clip)
            optimizer.step()
            optimizer.zero_grad()
            empty_cache()
    print('[Epoch {:d}/{:d}] Train Avg:'
          ' [Loss Cls {r[0]:.4f}]'
          ' [Loss Ins {r[1]:.4f}]'.format(epoch, cls_help.config.epochs, r=results.avg))
    empty_cache()
    return {
        'train/cls_loss': results.avg[0],
        'train/ins_loss': results.avg[1]
    }


def valid(cls_help, vali_loader):
    cls_help.model.eval()
    results = None
    for i, batch in enumerate(vali_loader):
        images, image_cat, image_path, image_name, image_text, images_with_mask = cls_help.generate_batch(batch)

        with torch.no_grad():
            logits, _ = cls_help.model(images, images_with_mask, text=image_text, text_included=True)
        probs = F.softmax(logits, dim=1)

        loss_cls = cls_help.criterions['cls_loss'](logits, image_cat)
        acc = cls_help.correct_predictions(probs, image_cat)
        result = [acc.item(), loss_cls.item()]
        if results is None:
            results = Averagvalue(len([*result]))
        results.update(result, images.size(0))
    empty_cache()
    return {
        'vali/acc': results.avg[0],
        'vali/loss': results.avg[1],
    }


def test(cls_help, test_image_dfs, epoch, fold, thres=0.5):
    cls_help.model.eval()

    save_dir = join(cls_help.config.submission_dir, 'test_fold_%d_epoch_%d' % (fold, epoch))
    if not exists(save_dir):
        makedirs(save_dir)
    shutil.rmtree(save_dir)
    makedirs(save_dir)
    cls_imgs_gt = []
    cls_imgs_pred = []
    feats = []
    labels = []
    num = 0
    for i, batch in enumerate(test_image_dfs):
        images, image_cat, image_path, image_name, image_text, images_with_mask = cls_help.generate_batch(batch)
        with torch.no_grad():
            logits, _ = cls_help.model(images, images_with_mask, text=image_text, text_included=True)
        gt = image_cat.cpu().numpy()
        _, pred_cls = logits.max(dim=1)
        pred_cls = pred_cls.cpu().numpy()
        cls_imgs_gt = np.concatenate([cls_imgs_gt, gt], axis=0)
        cls_imgs_pred = np.concatenate([cls_imgs_pred, pred_cls], axis=0)
        probs = F.softmax(logits, dim=1)

        if num == 0:
            feats = cls_help.model.get_feats()
            probs_ = probs
            num = 1
        else:
            feats = torch.cat((feats, cls_help.model.get_feats()), dim=0)
            probs_ = torch.cat((probs_, probs), dim=0)
        labels.append(image_cat[0].detach().cpu())
    feats = feats.detach().cpu().numpy()
    labels = np.array(labels)
    probs_ = probs_.detach().cpu().numpy()

    t_sne(feats, labels, cls_help.config, fold, epoch)

    model_name = 'SAM+ResNet18+SA+CA'
    auc_roc(probs_, labels, cls_help.config, model_name, fold, epoch)


    total_pred_list = cls_imgs_pred
    gt_list = cls_imgs_gt
    [accuracy, precision, _, f1score, sensitivity, _, _, specificity, auc] = calculate_metrics(gt_list, total_pred_list >= thres, probs_)
    result = [accuracy, precision, f1score, sensitivity, specificity, auc]
    zipDir(save_dir, save_dir + '.zip')
    shutil.rmtree(save_dir)
    empty_cache()
    print(
        " * Test fold %d epoch %d: ACC %0.4f, precision %0.4f, F1score %0.4f, sensitivity %0.4f, specificity %0.4f, AUC %0.4f" % (
            fold, epoch, result[0], result[1], result[2], result[3], result[4], result[5]))
    return result


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context

    torch.manual_seed(6666)
    torch.cuda.manual_seed(6666)
    random.seed(6666)
    np.random.seed(6666)
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    torch.backends.cudnn.benchmark = True  # cudn
    print("CuDNN: ", torch.backends.cudnn.enabled)

    import time
    current_time = time.strftime("%Y%m%d_%H%M%S")

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config-file', default='./config/sarcopenia/fusioncls_configuration.txt')
    argparser.add_argument('--use-cuda', action='store_true', default=True)
    argparser.add_argument('--train', help='running mode', default=True)
    argparser.add_argument('--gpu', help='GPU id', default='2')
    argparser.add_argument('--gpu-count', help='GPU ids for parallel training', default='0')
    argparser.add_argument('--run-num', help='run num', default=f"{current_time}")
    argparser.add_argument('--ema-decay', help='ema decay', default="0")
    argparser.add_argument('--active-iter', help='active iter', default="1")
    argparser.add_argument('--seed', help='random seed that controls dataset', default=666, type=int)
    argparser.add_argument('--model', help='model name', default="ResNetFusionTextNet")
    argparser.add_argument('--backbone', help='backbone name', default="resnet18")
    argparser.add_argument('--data_name',help='name of .pth data file',default='')
    argparser.add_argument('--data_path',help='path of data file',default='')
    args, extra_args = argparser.parse_known_args()
    config = Configurable(args, extra_args)
    torch.set_num_threads(config.workers + 1)
    config.train = args.train
    config.use_cuda = False
    if gpu and args.use_cuda:
        config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=config.tensorboard_dir)
    main(config, seed=args.seed, writer=writer)
