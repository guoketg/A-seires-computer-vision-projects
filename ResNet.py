import torch
from torch import nn,optim
from torch.nn import functional as F
from utils import load_data_cifar10,train


class Residual(nn.Module):
    def __init__(self,input_channels,num_channels,use_1x1conv=False,strides=1):
        #num_channels表示输出特征图的通道数
        super().__init__()
        self.conv1=nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1,stride=strides)
        self.conv2=nn.Conv2d(num_channels,num_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3=nn.Conv2d(input_channels,num_channels,kernel_size=1,stride=strides)
        else:
            self.conv3=None
        self.bn1=nn.BatchNorm2d(num_channels)
        self.bn2=nn.BatchNorm2d(num_channels)

    def forward(self,X):
        Y=F.relu(self.bn1(self.conv1(X)))
        Y=self.bn2(self.conv2(Y))
        if self.conv3:
            X=self.conv3(X)
        Y+=X
        return F.relu(Y)

in_channels=3
b1=nn.Sequential(nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),
                 nn.BatchNorm2d(64),nn.ReLU(),
                 nn.MaxPool2d(kernel_size=3,stride=2,padding=1))

def resnet_block(input_channels,num_channels,num_residuals,first_block=False):
    blk=[]
    for i in range(num_residuals):
        if i==0 and not first_block:
            blk.append(Residual(input_channels,num_channels,use_1x1conv=True,strides=2))
        else:
            blk.append(Residual(num_channels,num_channels))
    return blk

b2=nn.Sequential(*resnet_block(64,64,2,first_block=True))
b3=nn.Sequential(*resnet_block(64,128,2))
b4=nn.Sequential(*resnet_block(128,256,2))
b5=nn.Sequential(*resnet_block(256,512,2))

net=nn.Sequential(b1,b2,b3,b4,b5,
                  nn.AdaptiveAvgPool2d((1,1)),
                  nn.Flatten(),nn.Linear(512,10))


if __name__=='__main__':
    path="D:\code\python\DataMaluti\CIFAR10"
    lr,num_epochs,batch_size=0.001,10,256
    loss_fn=nn.CrossEntropyLoss()
    optimizer=optim.Adam(net.parameters(),lr=lr)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader,test_loader=load_data_cifar10(batch_size,path=path,resize=96)
    train(net,train_loader,test_loader,num_epochs,loss_fn,optimizer,device)
    torch.save(net.state_dict(),'D:\code\python\save_position\\trained_resnet.pth')


