from torch.nn import BCELoss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MInstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(MInstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, fetures, labels):
        N = self.batch_size * 2
        # N = fetures.size()[0] * 2
        sim = torch.matmul(fetures, fetures.T) / self.temperature
        # self.mask = (labels * labels.T)
        self.mask = torch.matmul(labels, labels.T)
        sim_i_j = torch.triu(sim, diagonal=1)[torch.triu(self.mask.bool(), diagonal=1)]
        sim_j_i = torch.tril(sim, diagonal=-1)[torch.tril(self.mask.bool(), diagonal=-1)]
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask_neg = 1 - self.mask
        mask_neg = mask_neg - torch.diag_embed(torch.diag(mask_neg))
        negative_samples = sim[mask_neg.bool()].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.to(fetures.get_device())).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = torch.matmul(z, z.T) / self.temperature
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.to(z_i.get_device())).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class L1LossMean(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='sum')

    def forward(self, x, y, masked):
        l1_sum = self.l1(x, y)
        return l1_sum


class LossVariance(nn.Module):
    """ The instances in target should be labeled
    """

    def __init__(self):
        super(LossVariance, self).__init__()

    def forward(self, input, target):
        B = input.size(0)

        loss = torch.zeros(1).float().to(input.get_device())
        for k in range(B):
            unique_vals = target[k].unique()
            unique_vals = unique_vals[unique_vals != 0]

            sum_var = torch.zeros(1).float().to(input.get_device())
            for val in unique_vals:
                instance = input[k][:, target[k] == val]
                if instance.size(1) > 1:
                    sum_var += instance.var(dim=1).sum()

            loss += sum_var / (len(unique_vals) + 1e-8)
            # print(len(unique_vals))
        loss /= B
        # print(loss)
        # print(B)
        return loss


def make_one_hot(labels, classes):
    one_hot = torch.cuda.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    return target


class WeightedSoftDiceLoss(nn.Module):
    '''
    from
    https://kaggle2.blob.core.windows.net/forum-message-attachments/212369/7045/weighted_dice_2.png
    '''

    def __init__(self):
        super(WeightedSoftDiceLoss, self).__init__()

    def forward(self, inputs, targets, weights):
        num = targets.size(0)
        m1 = inputs.view(num, -1)
        m2 = targets.view(num, -1)
        w = weights.view(num, -1)
        w2 = w * w
        intersection = (m1 * m2)

        score = 2. * ((w2 * intersection).sum(1) + 1) / ((w2 * m1).sum(1) + (w2 * m2).sum(1) + 1)
        score = 1 - score.sum() / num
        return max(score, 0)


class DC_and_Focal_loss(nn.Module):
    def __init__(self, gamma=0.25):
        super(DC_and_Focal_loss, self).__init__()
        self.fl = FocalLoss(gamma=gamma)
        self.dc = DiceLoss()

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        fl_loss = self.fl(net_output, target)
        result = fl_loss + dc_loss
        return result


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, probs, targets):
        num = targets.size(0)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + 1) / (m1.sum(1) + m2.sum(1) + 1)
        score = 1 - score.sum() / num
        return max(score, 0)


class DC_and_BCE_loss(nn.Module):
    def __init__(self, ):
        super(DC_and_BCE_loss, self).__init__()
        self.ce = BCELoss()
        self.dc = SoftDiceLoss()

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        result = ce_loss + dc_loss
        return result


class DiceLoss(nn.Module):
    def __init__(self, smooth=1., ignore_index=255):
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, output, target):
        if self.ignore_index not in range(target.min(), target.max()):
            if (target == self.ignore_index).sum() > 0:
                target[target == self.ignore_index] = target.min()
        target = make_one_hot(target.unsqueeze(dim=1), classes=output.size()[1])
        output = F.softmax(output, dim=1)
        output_flat = output.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (output_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (output_flat.sum() + target_flat.sum() + self.smooth))
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, ignore_index=255, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=alpha)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class CE_DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean', ignore_index=255, weight=None):
        super(CE_DiceLoss, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction=reduction, ignore_index=ignore_index)

    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        return CE_loss + dice_loss


class EntropyLoss(nn.Module):
    """
        Entropy loss for probabilistic prediction vectors
        input: batch_size x channels x h x w
        output: batch_size x 1 x h x w
    """

    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, v):
        assert v.dim() == 4
        n, c, h, w = v.size()
        return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * h * w * np.log2(c))


# def softmax_mse_loss(input_logits, target_logits):
#     """Takes softmax on both sides and returns MSE loss
#
#     Note:
#     - Returns the sum over all examples. Divide by the batch size afterwards
#       if you want the mean.
#     - Sends gradients to inputs but not the targets.
#     """
#     assert input_logits.size() == target_logits.size()
#     input_softmax = F.softmax(input_logits, dim=1)
#     target_softmax = F.softmax(target_logits, dim=1)
#     # num_classes = input_logits.size()[1]
#     # print (num_classes)
#     mse_loss = (input_softmax - target_softmax) ** 2
#     return mse_loss
#     # print (F.mse_loss(input_softmax, target_softmax, size_average=False))
#     # exit(0)
#     # return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss


def softmax_inter_extreme_lr_mse_loss(input_logits, input_logits2, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    input_softmax2 = F.softmax(input_logits2, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    tea_uncert = - (target_softmax * torch.log(target_softmax + 1e-12))
    tt = (1 - tea_uncert) * target_softmax + tea_uncert * input_softmax
    mse_loss = (input_softmax - tt) ** 2 + (input_softmax - input_softmax2) ** 2
    # 0.1 weight before
    return mse_loss / 2.0


def softmax_mse_loss_three(input_logits, input_logits2, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    input_softmax2 = F.softmax(input_logits2, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2 + (input_softmax2 - target_softmax) ** 2 + (
            input_softmax - input_softmax2) ** 2

    return mse_loss / 3.0


def softmax_extreme_lr_mse_loss(input_logits, input_logits2, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    input_softmax2 = F.softmax(input_logits2, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    tea_uncert = - (target_softmax * torch.log(target_softmax + 1e-12))
    tt = (1 - tea_uncert) * target_softmax + tea_uncert * input_softmax
    tt2 = (1 - tea_uncert) * target_softmax + tea_uncert * input_softmax2
    mse_loss = (input_softmax - tt) ** 2 + (input_softmax2 - tt2) ** 2
    # 0.5 weight before
    return mse_loss / 2.0


def softmax_aug_extreme_lr_mse_loss(input_logits, input_logits2, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    input_softmax2 = F.softmax(input_logits2, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    tea_uncert = - (target_softmax * torch.log(target_softmax + 1e-12))
    tt = (1 - tea_uncert) * target_softmax + tea_uncert * input_softmax
    tt2 = (1 - tea_uncert) * target_softmax + tea_uncert * input_softmax2
    mse_loss = (input_softmax - tt) ** 2 + 0.5 * (input_softmax2 - tt2) ** 2 + 0.5 * (
            input_softmax - input_softmax2) ** 2
    return mse_loss / 2.0


def softmax_lr_mse_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    tea_uncert = - (target_softmax * torch.log(target_softmax + 1e-12))
    tt = (1 - tea_uncert) * target_softmax + tea_uncert * input_softmax
    mse_loss = (input_softmax - tt) ** 2
    return mse_loss


def softmax_l1_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    l1_loss = F.l1_loss(input_softmax, target_softmax)
    return l1_loss


def softmax_consist_loss_two(input_logits, input_logits2, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    tea_uncert = - (target_softmax * torch.log(target_softmax + 1e-12))
    tt = (1 - tea_uncert) * target_softmax + tea_uncert * input_softmax

    consist_loss = -(torch.log(tt + 1e-12) * input_softmax + torch.log(1 - tea_uncert))
    return consist_loss


def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2) ** 2) / num_classes


def update_variance(labels, pred1, pred2):
    criterion = nn.CrossEntropyLoss(reduction='none')
    kl_distance = nn.KLDivLoss(reduction='none')
    loss = criterion(pred1, labels)

    # n, h, w = labels.shape
    # labels_onehot = torch.zeros(n, self.num_classes, h, w)
    # labels_onehot = labels_onehot.cuda()
    # labels_onehot.scatter_(1, labels.view(n,1,h,w), 1)
    sm = torch.nn.Softmax(dim=1)
    log_sm = torch.nn.LogSoftmax(dim=1)

    variance = torch.sum(kl_distance(log_sm(pred1), sm(pred2)), dim=1)
    exp_variance = torch.exp(-variance)
    # variance = torch.log( 1 + (torch.mean((pred1-pred2)**2, dim=1)))
    # torch.mean( kl_distance(self.log_sm(pred1),pred2), dim=1) + 1e-6
    # print(variance.shape)
    # print('variance mean: %.4f' % torch.mean(exp_variance[:]))
    # print('variance min: %.4f' % torch.min(exp_variance[:]))
    # print('variance max: %.4f' % torch.max(exp_variance[:]))
    # loss = torch.mean(loss/variance) + torch.mean(variance)
    loss = torch.mean(loss * exp_variance) + torch.mean(variance)
    return loss


def softmax_kl_loss_mask(inputs, targets):
    input_log_softmax = F.log_softmax(inputs, dim=1)
    targets = F.softmax(targets, dim=1)
    return F.kl_div(input_log_softmax, targets, reduction='mean')


def softmax_js_loss(inputs, targets, **_):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    epsilon = 1e-5

    M = (F.softmax(inputs, dim=1) + targets) * 0.5
    kl1 = F.kl_div(F.log_softmax(inputs, dim=1), M, reduction='mean')
    kl2 = F.kl_div(torch.log(targets + epsilon), M, reduction='mean')
    return (kl1 + kl2) * 0.5


class MultiboxLoss(nn.Module):
    def __init__(self, num_classes, alpha=1.0, neg_pos_ratio=3.0,
                 background_label_id=0, negatives_for_hard=100.0):
        self.num_classes = num_classes
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        if background_label_id != 0:
            raise Exception('Only 0 as background label id is supported')
        self.background_label_id = background_label_id
        self.negatives_for_hard = torch.FloatTensor([negatives_for_hard])[0]

    def _l1_smooth_loss(self, y_true, y_pred):
        abs_loss = torch.abs(y_true - y_pred)
        sq_loss = 0.5 * (y_true - y_pred) ** 2
        l1_loss = torch.where(abs_loss < 1.0, sq_loss, abs_loss - 0.5)
        return torch.sum(l1_loss, -1)

    def _softmax_loss(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, min=1e-7)
        softmax_loss = -torch.sum(y_true * torch.log(y_pred),
                                  axis=-1)
        return softmax_loss

    def forward(self, y_true, y_pred):
        # --------------------------------------------- #
        #   y_true batch_size, 8732, 4 + self.num_classes + 1
        #   y_pred batch_size, 8732, 4 + self.num_classes
        # --------------------------------------------- #
        num_boxes = y_true.size()[1]
        y_pred = torch.cat([y_pred[0], nn.Softmax(-1)(y_pred[1])], dim=-1)

        # --------------------------------------------- #
        #   分类的loss
        #   batch_size,8732,21 -> batch_size,8732
        # --------------------------------------------- #
        conf_loss = self._softmax_loss(y_true[:, :, 4:-1], y_pred[:, :, 4:])

        # --------------------------------------------- #
        #   框的位置的loss
        #   batch_size,8732,4 -> batch_size,8732
        # --------------------------------------------- #
        loc_loss = self._l1_smooth_loss(y_true[:, :, :4],
                                        y_pred[:, :, :4])

        # --------------------------------------------- #
        #   获取所有的正标签的loss
        # --------------------------------------------- #
        pos_loc_loss = torch.sum(loc_loss * y_true[:, :, -1],
                                 axis=1)
        pos_conf_loss = torch.sum(conf_loss * y_true[:, :, -1],
                                  axis=1)

        # --------------------------------------------- #
        #   每一张图的正样本的个数
        #   num_pos     [batch_size,]
        # --------------------------------------------- #
        num_pos = torch.sum(y_true[:, :, -1], axis=-1)

        # --------------------------------------------- #
        #   每一张图的负样本的个数
        #   num_neg     [batch_size,]
        # --------------------------------------------- #
        num_neg = torch.min(self.neg_pos_ratio * num_pos, num_boxes - num_pos)
        # 找到了哪些值是大于0的
        pos_num_neg_mask = num_neg > 0
        # --------------------------------------------- #
        #   如果所有的图，正样本的数量均为0
        #   那么则默认选取100个先验框作为负样本
        # --------------------------------------------- #
        has_min = torch.sum(pos_num_neg_mask)

        # --------------------------------------------- #
        #   从这里往后，与视频中看到的代码有些许不同。
        #   由于以前的负样本选取方式存在一些问题，
        #   我对该部分代码进行重构。
        #   求整个batch应该的负样本数量总和
        # --------------------------------------------- #
        num_neg_batch = torch.sum(num_neg) if has_min > 0 else self.negatives_for_hard

        # --------------------------------------------- #
        #   对预测结果进行判断，如果该先验框没有包含物体
        #   那么它的不属于背景的预测概率过大的话
        #   就是难分类样本
        # --------------------------------------------- #
        confs_start = 4 + self.background_label_id + 1
        confs_end = confs_start + self.num_classes - 1

        # --------------------------------------------- #
        #   batch_size,8732
        #   把不是背景的概率求和，求和后的概率越大
        #   代表越难分类。
        # --------------------------------------------- #
        max_confs = torch.sum(y_pred[:, :, confs_start:confs_end], dim=2)

        # --------------------------------------------------- #
        #   只有没有包含物体的先验框才得到保留
        #   我们在整个batch里面选取最难分类的num_neg_batch个
        #   先验框作为负样本。
        # --------------------------------------------------- #
        max_confs = (max_confs * (1 - y_true[:, :, -1])).view([-1])

        _, indices = torch.topk(max_confs, k=int(num_neg_batch.cpu().numpy().tolist()))

        neg_conf_loss = torch.gather(conf_loss.view([-1]), 0, indices)

        # 进行归一化
        num_pos = torch.where(num_pos != 0, num_pos, torch.ones_like(num_pos))
        total_loss = torch.sum(pos_conf_loss) + torch.sum(neg_conf_loss) + torch.sum(self.alpha * pos_loc_loss)
        total_loss = total_loss / torch.sum(num_pos)
        return total_loss
