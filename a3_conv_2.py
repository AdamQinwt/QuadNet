from os import O_TRUNC
from numpy import expand_dims
import torch
from torch._C import device
import torch.nn as nn
from tqdm import tqdm
import torch.autograd.functional as F
from torch.utils.data import Dataset,DataLoader
from nntrainer.trainer.am import AMGroup
from nntrainer.trainer.valid import accuracy
from nntrainer.trainer.optimizer import get_optimizer
from nntrainer.data_utils.dataset_loader import load_dataset
# from nntrainer.model_utils.loss import RMSE
from nntrainer.trainer.max_min_stat import MaxRecord
from nntrainer.model_utils.trivial import UnitLayer
from nntrainer.model_utils.view import View,Squeeze
from nntrainer.model_utils.convbase import ConvBaseBlock
from nntrainer.model_utils.fc import FCBlock_v2
from nntrainer.model_utils.weight_init import model_param_stat
from nntrainer.config_utils.args_updater import get_device,set_gpu
from nntrainer.config_utils.seed import fix_seed
from nntrainer.simple_datasets.mnist import load_mnist

from nntrainer.model_utils.quad_fc import QuadFCLayer

def get_square_indices(n):
    l=[]
    start_idx=0
    end_idx=n
    n1=n+1
    for i in range(n):
        l+=list(range(start_idx,end_idx))
        start_idx+=n1
        end_idx+=n
    return l

class ChannelQuadLayer(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(ChannelQuadLayer,self).__init__()
        n_fc_in=in_channel*(in_channel+3)
        n_fc_in>>=1
        self.idx=get_square_indices(in_channel)
        self.actual_channel=n_fc_in
        self.fc=nn.Conv2d(n_fc_in,out_channel,1,bias=False)
    def forward(self,x):
        bs,chn,width,height=x.size()
        area=width*height
        chn2=chn*chn
        a=x.permute(0,2,3,1).view(bs,area,chn)  # N(H*W)C
        x1=a.unsqueeze(-1)
        x2=a.unsqueeze(-2)
        y=torch.matmul(x1,x2).view(bs,area,chn2).permute(0,2,1).view(bs,chn2,width,height)[:,self.idx]
        y=torch.cat([x,y],1)
        out=self.fc(y)
        return out

class SpatialQuadLayer(nn.Module):
    def __init__(self,n_channel,ks=3,stride=1,padding=None,chn_share_weight=False,chn_expand=1):
        super(SpatialQuadLayer,self).__init__()
        qlen=ks*ks
        n_fc_in=qlen*(qlen+3)
        n_fc_in>>=1
        if padding is None:
            padding=ks>>1
        self.idx=get_square_indices(qlen)
        self.fc=nn.Conv1d(n_fc_in,chn_expand,kernel_size=1)
        self.qlen=qlen
        self.stride=stride
        self.padding=padding
        self.unfold=nn.Unfold(kernel_size=ks,dilation=1,padding=padding,stride=stride)
        self.ks=ks
        self.out_chn=n_channel*chn_expand
    def forward(self,x):
        bs,chn,width,height=x.size()
        area=width*height
        width_ = calcNewLength(width,self.padding,self.ks,self.stride)
        height_ = calcNewLength(height,self.padding,self.ks,self.stride)
        windows=self.unfold(x).transpose(1,2).view(bs,-1,chn,self.ks*self.ks).transpose(1,2)
        num_kernel=windows.size(2)
        x1=windows.unsqueeze(-1)
        x2=windows.unsqueeze(-2)
        y=torch.matmul(x1,x2).view(bs,chn,num_kernel,-1)[...,self.idx].permute(0,3,1,2) # N(IN)C(H*W)
        y=torch.cat([windows.permute(0,3,1,2),y],dim=1).view(bs,-1,chn*area)
        out=self.fc(y).view(bs,self.out_chn,height_,width_)
        return out

def calcNewLength(s,padding,kernel_size,stride=1):
    return ((s + 2 * padding - kernel_size )// stride) + 1

class QuadConvNet(UnitLayer):
    def __init__(self,nclass=10):
        super(QuadConvNet,self).__init__()
        self.main=nn.Sequential(
            # Squeeze(1,False),
            ConvBaseBlock([1,6],ks=[5],activate='relu',bn=False),
            ChannelQuadLayer(6,16),
            ConvBaseBlock([16,8],ks=[5],activate='relu',bn=False),
            View([-2,-1]),
            FCBlock_v2([392,16],activate=['relu']),
            QuadFCLayer(16,out_n=16,bn=False),
            QuadFCLayer(16,out_n=nclass,bn=False),
            nn.Softmax(-1)
        )

class QuadConvNet1(UnitLayer):
    def __init__(self,nclass=10):
        super(QuadConvNet1,self).__init__()
        self.main=nn.Sequential(
            # Squeeze(1,False),
            ConvBaseBlock([1,6],ks=[5],activate='none',bn=False),
            ChannelQuadLayer(6,16),
            ConvBaseBlock([16,8],ks=[5],activate='none',bn=False),
            ChannelQuadLayer(8,8),
            View([-2,-1]),
            FCBlock_v2([392,16],activate=['none']),
            QuadFCLayer(16,out_n=16,bn=False),
            QuadFCLayer(16,out_n=nclass,bn=False),
            nn.Softmax(-1)
        )

class QuadConvNet2(UnitLayer):
    def __init__(self,nclass=10):
        super(QuadConvNet2,self).__init__()
        self.main=nn.Sequential(
            # Squeeze(1,False),
            SpatialQuadLayer(n_channel=1,ks=5,chn_expand=32),
            ChannelQuadLayer(32,8),
            nn.AvgPool2d(2,2),
            SpatialQuadLayer(n_channel=8,ks=5,chn_expand=4),
            ChannelQuadLayer(32,8),
            nn.AvgPool2d(2,2),
            View([-2,-1]),
            nn.Linear(392,16,bias=False),
            nn.ReLU(inplace=True),
            QuadFCLayer(16,out_n=16,bn=False),
            QuadFCLayer(16,out_n=nclass,bn=False),
            nn.Softmax(-1)
        )

class LeNet5(UnitLayer):
    def __init__(self,nclass=10):
        super(LeNet5,self).__init__()
        self.main=nn.Sequential(
            # Squeeze(1,False),
            ConvBaseBlock([1,6],ks=[5],activate='relu',bn=False),
            nn.Conv2d(6,16,5,padding=0),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2,2),
            View([-2,-1]),
            FCBlock_v2([400,120,84,nclass],activate=['relu','relu','relu','none']),
            nn.Softmax(-1)
        )

if __name__=='__main__':
    fix_seed(1003)
    set_gpu('3')
    device=get_device()
    # dl_train,dl_valid=load_mnist('e:/data/mnist')
    dl_train,dl_valid=load_mnist('./mnist')

    # model=LeNet5(nclass=10).to(device)
    model=QuadConvNet1(nclass=10).to(device)
    print(model)
    cnt,param=model_param_stat(model)
    print(f'Model with {cnt} params. Total size: {param}')
    loss_func=nn.CrossEntropyLoss().to(device)

    opt=get_optimizer(model.parameters(),'adam',{'lr':0.001,'betas':(0.9,0.999)})

    crit=AMGroup(['train_loss','valid_loss','top1','top3'])
    best_top1=MaxRecord()
    best_top1.update(0,epoch=-1,top3=0)

    
    for epoch in range(200):
        model.train()
        crit.reset()
        tbar=tqdm(dl_train,desc=f'Training Epoch {epoch}',total=len(dl_train),dynamic_ncols=True)
        for x,y_ in tbar:
            x=x.to(device)
            y_=y_.to(device)
            opt.zero_grad()
            bs=x.size(0)
            y=model(x)
            loss=loss_func(y,y_)
            loss.backward()
            opt.step()
            l=crit['train_loss']+(loss.detach().cpu().item(),bs)
            tbar.set_postfix({'loss':crit['train_loss'].avg})
        
        model.eval()
        tbar=tqdm(dl_valid,desc=f'Validating Epoch {epoch}',total=len(dl_valid),dynamic_ncols=True)
        for x,y_ in tbar:
            x=x.to(device)
            y_=y_.to(device)
            opt.zero_grad()
            bs=x.size(0)
            y=model(x)
            loss=loss_func(y,y_)
            acc1,acc3=accuracy(y,y_,(1,3))
            l=crit['valid_loss']+(loss.detach().cpu().item(),bs)
            l=crit['top1']+(acc1,bs)
            l=crit['top3']+(acc3,bs)
            tbar.set_postfix({'loss':crit['valid_loss'].avg,'top 1/3':(crit['top1'].avg,crit['top3'].avg)})
        print(f'Epoch {epoch}:')
        for k,v in crit.t_avg():
            print(f'\t{k}:{v}')
        best_top1.update(crit['top1'].avg,epoch=epoch,top3=crit['top3'].avg)
        print(f'\tBest top1/3:{best_top1.value}/{best_top1.info["top3"]}@{best_top1.info["epoch"]}')