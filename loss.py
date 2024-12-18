import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, t, device):
        super(Loss, self).__init__()
        self.t = t
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def contrast_loss(self, h0, h1, we1, we2):
        mask_miss_inst = we1.mul(we2).bool() # mask the unavailable instances

        v1 = h0[mask_miss_inst]
        v2 = h1[mask_miss_inst]
        n = v1.size(0)
        N = 2 * n
        if n == 0:
            return 0
        v1 = F.normalize(v1, p=2, dim=1) #normalize two vectors
        v2 = F.normalize(v2, p=2, dim=1)

        similarity_mat = torch.matmul(v1, v2.T) / self.t

        pos = torch.exp(similarity_mat).diag()
        fm = torch.exp(similarity_mat).sum(1)

        loss = -torch.log(pos/fm).sum()/n





        # h0, h1 = nn.functional.normalize(h0, dim=1),nn.functional.normalize(h1, dim=1)
        # cos12 = torch.matmul(h0, h1.T)
        # cos11 = torch.matmul(h0, h0.T)
        # cos22 = torch.matmul(h1, h1.T)
        # cos21 = torch.matmul(h1, h0.T)
        # sim12 = (cos12).exp()
        # sim11 = (cos11).exp()
        # sim21 = (cos21).exp()
        # sim22 = (cos22).exp()
        # pos1 = sim12.diag()
        # pos2 = sim21.diag()
        # p1 = pos1 / (sim12 + sim11).sum(1)
        # p2 = pos2 / (sim21 + sim22).sum(1)
        # # loss1 = F.kl_div(F.log_softmax(p1, dim=0), F.softmax(p2, dim=0), reduction='batchmean')
        # # loss2 = F.kl_div(F.log_softmax(p2, dim=0), F.softmax(p1, dim=0), reduction='batchmean')
        # loss1 = (p1 * (torch.log(p1 / p2))).sum()
        # loss2 = (p2 * (torch.log(p2 / p1))).sum()
        #
        # # loss = -p.log().mean()
        # # loss1 = -(p1 * torch.log(p1)).mean()
        # # loss2 = -(p2 * torch.log(p2)).mean()
        # loss = (loss1 + loss2) / 2
        return loss

    def wmse_loss(self, input, target, weight, reduction='mean'):
        ret = (torch.diag(weight).mm(target - input)) ** 2
        ret = torch.mean(ret)
        return ret
    def weighted_BCE_loss(self,target_pre,sub_target,inc_L_ind,reduction='mean'):
        assert torch.sum(torch.isnan(torch.log(target_pre))).item() == 0
        assert torch.sum(torch.isnan(torch.log(1 - target_pre + 1e-5))).item() == 0
        res=torch.abs((sub_target.mul(torch.log(target_pre + 1e-5)) \
                                                + (1-sub_target).mul(torch.log(1 - target_pre + 1e-5))).mul(inc_L_ind))
        
        if reduction=='mean':
            return torch.sum(res)/torch.sum(inc_L_ind)
        elif reduction=='sum':
            return torch.sum(res)
        elif reduction=='none':
            return res


class My_loss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mean(torch.mm(x,y.t(), 2))


