import torch
from torch import nn,optim

from ModernCNN.utils import load_data_fashion_mnist, train, load_data_cifar10,load_data_mnist


def conv_block(input_channels,num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=3,padding=1)
    )

class DenseBlock(nn.Module):
    def __init__(self,num_convs,input_channels,num_channels):
        super(DenseBlock,self).__init__()
        layer=[]
        for i in range(num_convs):
            layer.append(conv_block(num_channels*i+input_channels,num_channels))
        self.net=nn.Sequential(*layer)

    def forward(self,x):
        for blk in self.net:
            y=blk(x)
            x=torch.cat((x,y),dim=1)
        return x

blk=DenseBlock(2,3,10)
x=torch.randn(4,3,8,8)
y=blk(x)
print(y.shape)

def transition_block(input_channels,num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels),nn.ReLU(),
        nn.Conv2d(input_channels,num_channels,kernel_size=1),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )
blk=transition_block(23,10)
print(blk(y).shape)

in_channels=3
b1=nn.Sequential(
    nn.Conv2d(in_channels,64,kernel_size=7,stride=2,padding=3),
    nn.BatchNorm2d(64),nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
)

#衔接DenseBlock和Transition_Block
num_channels,growth_rate=64,32
num_convs_in_dense_blocks=[4,4,4,4]
blks=[]
for i,num_convs in enumerate(num_convs_in_dense_blocks):
    blks.append(DenseBlock(num_convs,num_channels,growth_rate))
    num_channels+=num_convs*growth_rate
    if i!=len(num_convs_in_dense_blocks)-1:
        blks.append(transition_block(num_channels,num_channels//2))
        num_channels=num_channels//2

net=nn.Sequential(
    b1,*blks,
    nn.BatchNorm2d(num_channels),nn.ReLU(),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten(),
    nn.Linear(num_channels,10)
)

if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size=128
    num_epochs=10
    lr=0.01
    path="D:\code\python\data\CIFAR10"

    optimizer=optim.SGD(net.parameters(),lr=lr)
    loss_fn=nn.CrossEntropyLoss()

    train_loader,test_loader=load_data_cifar10(batch_size,path,resize=96)
    train(net,train_loader,test_loader,num_epochs,loss_fn,optimizer,device)
    torch.save(net.state_dict(),'D:\code\python\save_position\trained_DenseNet.pth')


