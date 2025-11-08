import torch
from torch import nn,optim
from utils import load_data_fashion_mnist,train

net = nn.Sequential(
    nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),

    nn.Conv2d(96,256,kernel_size=5,padding=2),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),

    nn.Conv2d(256,384,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.Conv2d(384,384,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.Conv2d(384,256,kernel_size=3,padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3,stride=2),

    nn.Flatten(),

    nn.Linear(6400,4096),nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096,4096),nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096,10)
)


if __name__ == "__main__":
    X=torch.randn(1,1,224,224)
    for layer in net:
        X=layer(X)
        print(layer.__class__.__name__,'output shape:\t',X.shape)

    batch_size=128
    lr, num_epochs = 0.001, 10
    path="D:\code\python\data"
    loss_fn=nn.CrossEntropyLoss()
    optimizer=optim.SGD(net.parameters(),lr=lr)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, test_iter = load_data_fashion_mnist(batch_size, path, resize=224)
    train(net,train_iter,test_iter,num_epochs,loss_fn,optimizer,device)
    torch.save(net.state_dict(),"D:\code\python\save_position\\trained_AlexNet.pth")


