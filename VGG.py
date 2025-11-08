import torch
from torch import nn,optim
from utils import load_data_cifar10,train

def vgg_block(num_convs,in_channels,out_channels):
    layers=[]
    for _ in range(num_convs):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # 添加权重初始化
        nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(conv.bias, 0)

        layers.append(conv)
        layers.append(nn.ReLU())
        in_channels=out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

#vgg-11 8+3
def vgg(conv_arch):
    conv_blks=[]
    in_channels=1
    for (num_convs,out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs,in_channels,out_channels))
        in_channels=out_channels
    #每次池化都会将特征图的高度和宽度各缩小一半：224 → 112 → 56 → 28 → 14 → 7（经过 5 次池化后）
    #out_channels*7*7作为全连接层的输入层
    return nn.Sequential(*conv_blks,nn.Flatten(),nn.Linear(out_channels*7*7,4096),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(4096,4096),nn.ReLU(),nn.Dropout(0.5),
                         nn.Linear(4096,10))

conv_arch=[(1,64),(1,128),(2,256),(2,512),(2,512)]
net=vgg(conv_arch)

X=torch.randn((1,1,224,224))
for blk in net:
    X=blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)

ratio=4
small_conv_arch=[(pair[0],pair[1]//ratio) for pair in conv_arch]

net=vgg(small_conv_arch)
lr,num_epochs,batch_size=0.001,10,128
loss_fn=nn.CrossEntropyLoss()
optimizer=optim.Adam(net.parameters(),lr=lr)
path="D:\code\python\data\FashionMNIST\\raw"
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_loader,test_loader=load_data_mnist1(batch_size,path)
train_loader,test_loader=load_data_cifar10(batch_size,path,resize=224)
train(net,train_loader,test_loader,num_epochs,loss_fn,optimizer,device)
torch.save(net.state_dict(),'D:\code\python\save_position\\trained_vgg11.pth')
#D:\code\python\save_position

