import torch
from einops import rearrange
import numpy as np
from torchvision import models as models
from collections import OrderedDict
from torch import nn
if __name__ == '__main__':
    # nums = np.asarray([1.2, 3, 0.5, 2.3, -1.9, 7.7], dtype=np.float32)
    # t1 = torch.from_numpy(rearrange(nums, 'b -> ()()() b'))
    # print(t1.size())
    # print(t1)
    # t2 = t1.view(1,-1)
    # print(t2.size())
    # print(t2)
    # print(torch.from_numpy(rearrange(nums, 'b -> () b ()')))

    # batch = torch.Tensor(4,3,2,3)
    # print(batch[1])
    # print("****************************************************")
    # b0 = torch.unsqueeze(batch[0], dim = 0)
    # print(b0)
    # b1 = torch.unsqueeze(batch[1], dim = 0)
    # print(b1)
    # b01 = torch.cat((b0, b1), dim = 0)
    # print(b01)
    # print(b01[0])
    # print(b01[1])

    # for b in batch:
    #     print(b.size())
    #     b = torch.unsqueeze(b, dim = 0)
    #     print(b.size())

    # i = int(1)
    # j = float(1.0)
    # print(i==j)

    # q = torch.ones(32,64)
    # print(q.size())
    # print(q.unsqueeze(-1).unsqueeze(-1).size())
    # print(q.unsqueeze(-1).unsqueeze(-1).expand(32,64,7,7).size())
    # p = torch.bmm(q.unsqueeze(2),q.unsqueeze(1))
    # print(p.size())

    # resnet18_state_dict = torch.load("resnet18-f37072fd.pth")
    # with open("resnet18.txt",'a') as file0:
    #     print(resnet18_state_dict,file=file0)

    # best_model_state_dict = torch.load("fold_4_best_model.pt")
    # with open("best_model.txt",'a') as file1:
    #     print(best_model_state_dict, file=file1)
    # print("resnet18_state_dict")
    # print(resnet18_state_dict)
    # print("***************************************************************************************")
    # print("best_model_state_dict")
    # print(best_model_state_dict)

    resnet18 = models.resnet18(pretrained=True)
    resnet = nn.Sequential(OrderedDict(list(resnet18.named_children())[:-2]))
    # resnet18.avgpool = nn.Sequential()
    # resnet18.fc = nn.Conv2d(resnet18.fc.in_features, 2, 1, bias=False)
    resnet.add_module('fc', nn.Conv2d(resnet18.fc.in_features, 2, 1, bias=False))
    resnet.add_module('dropout', nn.Dropout2d(0.5))
    print(resnet18)
    print("**********************************************************************************************")
    print(resnet)
